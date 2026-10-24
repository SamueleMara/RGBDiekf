"""Reference coherence hierarchy applied to free point trajectories."""
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from .surface_supernodes import connected_geodesic_assignment
from .reference_motion import supernode_trajectories,edge_strain
from .reference_coherence import extract_moving_component
from .group_visualization import serial
from .store import json_write


def run(partition,arrays,times,root,representative_occupancy=512,max_candidates=48,anchor_frame=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    edges=partition.edges;n=len(partition.xyz)
    weights=np.linalg.norm(partition.xyz[edges[:,0]]-partition.xyz[edges[:,1]],axis=1)
    graph=coo_matrix((np.r_[weights,weights],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    sid=connected_geodesic_assignment(graph,representative_occupancy)
    reduction=supernode_trajectories(arrays['xyz'],torch.as_tensor(sid));traj=reduction['trajectories']
    # Canonical here is the actual endpoint, even in chronological reverse runs.
    canonical=np.stack([partition.xyz[sid==s].mean(0) for s in range(traj.shape[1])])
    count=len(canonical);k=min(12,count-1)
    if k<1:
        return {'status':'NO_COHERENT_MOVING_COMPONENT','reason':'Fewer than two representative nodes','point_support':np.zeros(n,bool),'representative_membership':sid,'labels':['INCOHERENT_OR_AMBIGUOUS']*count}
    _,near=cKDTree(canonical).query(canonical,k=k+1,workers=1)
    pairs=np.unique(np.sort(np.c_[np.repeat(np.arange(count),k),near[:,1:].ravel()],axis=1),axis=0)
    a=torch.as_tensor(pairs[:,0]);b=torch.as_tensor(pairs[:,1])
    st=edge_strain(traj,a,b,times=torch.as_tensor(times))
    # Reference Kabsch uses traj[0] as its gauge. Reversing the array to the
    # actual endpoint changes only that gauge, never the saved chronological IDs.
    anchor=anchor_frame if anchor_frame is not None else (0 if np.allclose(arrays['xyz'][0],partition.xyz,atol=1e-12,rtol=0) else len(times)-1)
    if anchor not in (0,len(times)-1):raise ValueError('Coherence requires the actual first/last source endpoint')
    directed=traj if anchor==0 else traj.flip(0)
    result=extract_moving_component(directed,a,b,st['C'],count,max_candidates=max_candidates,refine=False)
    moving=result['moving_nodes'].numpy();support=np.isin(sid,moving)
    np.savez_compressed(root/'representatives.npz',membership=sid,canonical=canonical,trajectory=traj.numpy(),edges=pairs,C=st['C'].numpy(),log_strain=st['z'].numpy(),observed_count=np.stack([np.bincount(sid,weights=np.asarray(o,dtype=int),minlength=count) for o in arrays['point_observed']]),moving_support=support)
    result.update(point_support=support,representative_membership=sid,representative_occupancy=representative_occupancy,
        candidate_limit=max_candidates,reference_revision='481b2cf2d5602932286948cf14114f9caee4de3c',
        semantics='Reference held-out representative-trajectory selection; trajectories share grouped corrections and are not independent sensor samples. Original raw RGB-D validation is a separate downstream requirement.',
        computational_groups=partition.count,representative_nodes=count,anchor_frame=anchor)
    result['cross_validation_partition']='supernodes, never cameras; all-camera free trajectories'
    json_write(root/'result.json',serial({k:v for k,v in result.items() if k not in ('point_support','representative_membership','moving_nodes')}))
    return result
