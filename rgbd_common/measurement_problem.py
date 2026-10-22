"""Fixed externally matched factors. Neither filter is imported here."""
import numpy as np
import torch
from .observation_model import output,gaussian_log_probability,select_covariance


class MeasurementProblem:
    def __init__(self,model,matches,backend,observation_ids=None):
        self.model=model;self.matches=matches;self.backend=backend;self.excluded_source_factors=0
        self.xyz=backend.tensor(model.xyz);self.rgb=backend.tensor(model.rgb)
        self.group=backend.tensor(model.group,torch.long)
        for m in matches:
            if m.mask_context is not None:
                frame,_=m.mask_context;pixels=m.pixel.detach().cpu().numpy()
                if not np.all(frame.valid.ravel()[pixels]):raise ValueError('Kalman factors must refer to valid masked foreground raw pixels')
        if observation_ids is not None:
            from .observation_model import Matches
            source=backend.tensor(model.source_id,torch.long);selected=[]
            for m in matches:
                raw=m.frame*observation_ids.frame_size+int(observation_ids.offsets[m.camera_index])+m.pixel
                use=raw!=source[m.point];self.excluded_source_factors+=int((~use).sum())
                selected.append(Matches(m.point[use],m.pixel[use],m.observed[use],select_covariance(m.covariance,use),m.camera,m.frame,m.camera_index,m.unmatched_score[use],m.assignment_score[use],m.association_poses,m.mask_context))
            self.matches=selected
    def for_group(self,group):
        from .observation_model import Matches
        result=object.__new__(MeasurementProblem);result.model=self.model;result.backend=self.backend
        result.xyz=self.xyz;result.rgb=self.rgb;result.group=self.group;result.matches=[]
        result.excluded_source_factors=self.excluded_source_factors
        for m in self.matches:
            use=self.group[m.point]==group
            result.matches.append(Matches(m.point[use],m.pixel[use],m.observed[use],select_covariance(m.covariance,use),m.camera,m.frame,m.camera_index,m.unmatched_score[use],m.assignment_score[use],m.association_poses,m.mask_context))
        return result
    def chunks(self,bytes_per_item=2048):
        size=self.backend.chunk_size(bytes_per_item)
        for matches in self.matches:yield from matches.chunks(size)
    def evaluate(self,state):
        if not self.feasible(state):return float('inf')
        value=0.
        for match in self.chunks():
            prediction,_,q=output(state,self.xyz[match.point],self.rgb[match.point],self.group[match.point],match.camera,self.backend)
            if not torch.isfinite(prediction).all() or torch.any(q[:,2]<=0):return float('inf')
            value-=float(gaussian_log_probability(match.observed-prediction,match.covariance).sum())
        return value
    def feasible(self,state):
        from .foreground import check_hard_mask_feasibility
        return all(m.mask_context is None or check_hard_mask_feasibility(self.model,state,*m.mask_context,self.backend,classify_foreground=False)['feasible'] for m in self.matches)
