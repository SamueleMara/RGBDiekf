# SUPERSEDED persistent-physical-supernode experiment; not used by the corrected free-point pipeline.
"""Versioned, evidence-selected connected partitions with complete fresh replay."""
from pathlib import Path
import hashlib
import numpy as np
from .phase1_grouped_controller import run_phase1
from .group_partition_validation import group_evidence,validate_group,split_partition,merge_partition,validate_merge
from .group_sensor_tracks import write_sensor_pool,index_sensor_pool
from .store import json_write


def run_partition_epochs(dataset,config,noise,backend,root,anchor='first',target_occupancy=64,partition=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);seen=set();ledger=[]
    while True:
        version=0 if partition is None else partition.version;directory=root/f'partition_v{version:03d}'
        partition,arrays=run_phase1(dataset,config,noise,backend,directory,anchor,target_occupancy,partition=partition)
        if partition.digest() in seen:raise ArithmeticError('Partition cycle; no evidence-stable result')
        seen.add(partition.digest());original=hashlib.sha256(np.ascontiguousarray(arrays['xyz']).tobytes()).hexdigest()
        pool=write_sensor_pool(partition,dataset,config,backend,root/'independent_raw',0 if anchor=='first' else len(dataset.times)-1)
        db=index_sensor_pool(pool,partition,directory,len(dataset.times));proposals=[];accepted=None
        for g in range(partition.count):
            rows=group_evidence(db,g,dataset);report=validate_group(partition,g,rows,len(dataset.cameras))
            json_write(directory/f'validation/{g:06d}.json',report);proposals.append({'group':g,'status':report['status']})
            if report['status']=='APPROVED_SPLIT':accepted=('split',g,report['membership_labels']);break
        if accepted is None:
            for a,b in partition.adjacency:
                ra=group_evidence(db,a,dataset);rb=group_evidence(db,b,dataset);combined=[]
                for k in sorted({r['frame'] for r in ra+rb}):
                    rows=[r for r in ra+rb if r['frame']==k];row={key:value for key,value in rows[0].items() if key.startswith('_')}
                    row.update({key:np.concatenate([r[key] for r in rows]) for key in rows[0] if not key.startswith('_') and key!='frame'});row['frame']=k;combined.append(row)
                report=validate_merge(partition,int(a),int(b),dataset,config,backend,0 if anchor=='first' else len(dataset.times)-1,combined)
                json_write(directory/f'merge_validation/{a:06d}_{b:06d}.json',report)
                if report['status']=='APPROVED_MERGE':accepted=('merge',int(a),int(b));break
        db.close()
        if original!=hashlib.sha256(np.ascontiguousarray(arrays['xyz']).tobytes()).hexdigest():raise AssertionError('Validation modified prior epoch trajectories')
        ledger.append({'epoch':partition.version,'partition_hash':partition.digest(),'groups':partition.count,'proposals':proposals,'accepted_operation':None if accepted is None else accepted[0],'accepted_group':None if accepted is None else accepted[1],'prior_covariance_reused':False})
        json_write(root/'epochs.json',{'epochs':ledger,'recovery_started':False,'canonical_sample_ids_immutable':True})
        if accepted is None:
            status='PARTITION_EVIDENCE_STABLE_WITH_UNRESOLVED_GROUPS' if any(p['status'] in ('UNRESOLVED_SPLIT','GROUP_UNRESOLVED') for p in proposals) else 'PARTITION_EVIDENCE_STABLE_CONDITIONAL_ON_RAW_IDENTITIES'
            json_write(root/'partition_result.json',{'status':status,'epochs':len(ledger),'physical_labels':'NOT_ESTIMATED','complete_joint_pipeline':False})
            return partition,arrays,ledger
        before=partition
        partition=split_partition(partition,*accepted[1:]) if accepted[0]=='split' else merge_partition(partition,*accepted[1:])
        if not np.array_equal(partition.source,before.source) or not np.array_equal(partition.xyz,before.xyz):raise AssertionError('An epoch changed canonical observations')
        # Next loop creates fresh endpoint gauge/twist/covariance. No previous
        # group's posterior belongs to this new membership hypothesis.
