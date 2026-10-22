"""Bounded time/view batches of the SAME sparse backward point likelihood.

Only projected raw samples are gathered from streamed frames. No all-sequence
image/XYZ tensor exists, and no source or target time is subsampled.
"""
import numpy as np
import torch
from .backends.torch_cuda import numpy
from .observation_model import covariance,gaussian_log_probability
from .progress import progress
from .projection_cuda import rigid_transform,transport_and_project,build_pixel_bins_and_zbuffer


def backward_evidence(xyz,color,group,states,dataset,target_times,config,backend,cache,model,batch_views=None):
    n=len(xyz);views=[(k,c) for k in target_times for c in range(len(dataset.cameras))]
    size=max(1,backend.chunk_size(2048)//max(1,n)) if batch_views is None else batch_views
    size=min(size,max(1,len(views)));device=backend.device
    evidence=torch.zeros(n,dtype=torch.float64,device=device);strength=torch.full_like(evidence,-torch.inf)
    infeasible=torch.zeros(n,dtype=torch.bool,device=device)
    sf=np.full(n,-1,dtype=int);sc=sf.copy();sp=np.full(n,-1,dtype=np.int64);so=sp.copy()
    point32=backend.tensor(xyz,backend.input_dtype);point=point32.double();colour=backend.tensor(np.asarray(color,np.float32))
    hidden=0;free=0
    for start in progress(range(0,len(views),size),desc=f'Backward CUDA g={group} ({size} views/batch)',unit='view-batch',leave=False):
        batch=views[start:start+size];poses=np.array([states[k].poses[group] for k,c in batch]);cameras=[dataset.cameras[c] for k,c in batch]
        ext=np.array([c.world_to_camera for c in cameras]);intrinsic=backend.tensor(np.array([c.intrinsics for c in cameras]),backend.input_dtype)
        # Keep float32 projection and float64 conditional h identical to scalar
        # matching, rather than quietly changing the selected operator.
        t=backend.tensor(poses,backend.input_dtype);cw=backend.tensor(ext,backend.input_dtype)
        world=rigid_transform(point32[None],t[:,None,:3,:3],t[:,None,:3,3])
        q=rigid_transform(world,cw[:,None,:3,:3],cw[:,None,:3,3])
        safe=torch.where(q[:,:,2]!=0,q[:,:,2],torch.ones_like(q[:,:,2]))
        uv=q[:,:,:2]/safe[:,:,None]*intrinsic[:,None,:2]+intrinsic[:,None,2:]
        pixel=torch.floor(uv+.5).long();width=backend.tensor([c.width for c in cameras],torch.long);height=backend.tensor([c.height for c in cameras],torch.long)
        near=backend.tensor([c.near for c in cameras],backend.input_dtype);far=backend.tensor([c.far for c in cameras],backend.input_dtype)
        valid=(q[:,:,2]>near[:,None])&(q[:,:,2]<far[:,None])&torch.isfinite(uv).all(2)
        valid&=(pixel[:,:,0]>=0)&(pixel[:,:,0]<width[:,None])&(pixel[:,:,1]>=0)&(pixel[:,:,1]<height[:,None])
        flat=torch.nonzero(valid.reshape(-1),as_tuple=True)[0];depth=q[:,:,2].reshape(-1)
        flat=flat[torch.argsort(depth[flat],stable=True)];view=flat//n;index=flat%n
        offset=torch.cat([width.new_zeros(1),(width*height).cumsum(0)])
        rawpixel=(pixel[:,:,1]*width[:,None]+pixel[:,:,0]).reshape(-1)[flat];key=offset[view]+rawpixel
        order=torch.argsort(key,stable=True);flat=flat[order];rawpixel=rawpixel[order];key=key[order]
        first=torch.ones(len(key),dtype=torch.bool,device=device)
        if len(key)>1:first[1:]=key[1:]!=key[:-1]
        flat=flat[first];rawpixel=rawpixel[first];view=flat//n;index=flat%n
        v=numpy(view);pix=numpy(rawpixel);selected=np.zeros(len(pix),bool);observed=np.empty((len(pix),6))
        owners=np.full(len(pix),-1,dtype=np.int64)
        # CPU reads only raw samples selected by the GPU projector. Frames are
        # released immediately; batch size comes from the measured free memory.
        for b,(k,c) in enumerate(batch):
            take=np.flatnonzero(v==b)
            if not len(take):continue
            frame=dataset.observation(k,c);p=pix[take];foreground=frame.mask.ravel()[p]
            valid=frame.valid.ravel()[p]
            if hasattr(cache,'front_depth_at') and np.isfinite(cache.score['energy']):
                known_depth=np.full(len(p),np.inf,dtype=np.float32)
                known_depth[valid]=cache.front_depth_at(k,c,p[valid])
                hidden_by_map=backend.tensor(known_depth,backend.input_dtype)<depth[flat[backend.tensor(take,torch.long)]]
            else:
                known=transport_and_project(backend.tensor(model.xyz,backend.input_dtype),backend.tensor(model.group,torch.long),backend.tensor(states[k].poses,backend.input_dtype),frame.camera)
                known_bins=build_pixel_bins_and_zbuffer(known,frame.camera,backend.tensor(model.point_id,torch.long))
                hidden_by_map=known_bins['first_depth'].reshape(-1)[backend.tensor(p,torch.long)]<depth[flat[backend.tensor(take,torch.long)]]
            exposed=~numpy(hidden_by_map)
            if getattr(frame,'object_only_no_external_occluders',None) is True:
                rejected=backend.tensor(take[(~foreground)&exposed],torch.long);infeasible[index[rejected]]=True
            valid&=exposed
            selected[take]=valid
            use=take[valid];target=p[valid]
            observed[use]=np.c_[target%frame.camera.width,target//frame.camera.width,np.asarray(frame.depth).ravel()[target],frame.rgb.reshape(-1,3)[target]]
            owners[use]=cache.owners_at(k,c,target,model)
        use=backend.tensor(selected,torch.bool);view=view[use];index=index[use];rawpixel=rawpixel[use];flat=flat[use]
        if not len(flat):continue
        obs=backend.tensor(observed[selected]);owner=owners[selected];t64=backend.tensor(poses)[view];cw64=backend.tensor(ext)[view]
        x=rigid_transform(point[index],t64[:,:3,:3],t64[:,:3,3])
        qq=rigid_transform(x,cw64[:,:3,:3],cw64[:,:3,3]);ii=backend.tensor(np.array([c.intrinsics for c in cameras]))[view]
        h=torch.cat([qq[:,:2]/qq[:,2,None]*ii[:,:2]+ii[:,2:],qq[:,2,None],colour[index]],dim=1)
        r=covariance(config,obs[:,2],backend)
        logf=gaussian_log_probability(obs-h,r)+np.log1p(-config.sensor['no_return'])+np.log(config.match_prior/max(1,n))
        background=gaussian_log_probability(torch.cat([obs[:,3:],obs[:,2,None]],dim=1)-backend.tensor(config.sensor['background_mean']),backend.tensor(config.sensor['background_covariance']))
        area=(width*height)[view].double()
        bg=background+np.log1p(-config.sensor['background_no_return'])-torch.log(area)+np.log1p(-config.match_prior)
        gain=logf-bg;matched=gain>0
        predicted=depth[flat].double();variance=config.rgbd_covariance[3,3]+predicted**2*config.rgbd_depth_squared[3,3]
        odds=torch.special.log_ndtr((predicted-obs[:,2])/torch.sqrt(variance))
        # Nearer FG with a losing match explains hiding, without a false
        # missing-point penalty. Free-space contradiction retains noise odds.
        hidden_fg=(predicted>obs[:,2])&~matched
        values=torch.zeros((len(batch),n),dtype=torch.float64,device=device);values[view,index]=torch.where(matched,gain,torch.where(hidden_fg,torch.zeros_like(odds),odds))
        evidence+=values.sum(0);hidden+=int((predicted>obs[:,2]).sum());free+=int((predicted<obs[:,2]).sum())
        gains=torch.full_like(values,-torch.inf);gains[view,index]=torch.where(matched,gain,-torch.inf)
        best,which=gains.max(0);better=best>strength
        ids=torch.nonzero(better,as_tuple=True)[0]
        if len(ids):
            # One unique sample per (view, source point); ties keep the first
            # target view in the prescribed temporal/camera order.
            positions=torch.full((len(batch),n),-1,dtype=torch.long,device=device)
            positions[view,index]=torch.arange(len(view),device=device)
            selected_positions=numpy(positions[which[ids],ids]);updated=numpy(ids);vv=numpy(which[ids])
            sf[updated]=np.array([batch[j][0] for j in vv]);sc[updated]=np.array([batch[j][1] for j in vv])
            sp[updated]=numpy(rawpixel)[selected_positions];so[updated]=owner[selected_positions];strength[ids]=best[ids]
    evidence[infeasible]=-torch.inf
    return numpy(evidence),numpy(strength),sf,sc,sp,so,{'camera_group_queries':len(views),'hidden_tests':hidden,'free_space_tests':free,'view_batch_size':size,'hard_mask_infeasible_point_hypotheses':int(infeasible.sum())}
