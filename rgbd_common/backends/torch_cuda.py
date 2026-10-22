"""Shared Torch kernels; CPU mode is the numerical reference, never CUDA proof."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch


class Backend:
    def __init__(self,device='cuda',input_dtype='float32'):
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA_UNAVAILABLE: run in rigeuler3dgs with host GPU access; CPU fallback is explicit --device cpu')
        self.input_dtype=getattr(torch,input_dtype)
        if self.device.type=='cpu':torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
    def tensor(self,value,dtype=torch.float64):
        if hasattr(value,'__array__') and not isinstance(value,(np.ndarray,torch.Tensor)):value=np.asarray(value)
        if isinstance(value,np.ndarray) and not value.flags.writeable:return torch.tensor(value,device=self.device,dtype=dtype)
        return torch.as_tensor(value,device=self.device,dtype=dtype)
    def synchronize(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
    def chunk_size(self,bytes_per_item):
        free=torch.cuda.mem_get_info(self.device)[0] if self.device.type=='cuda' else 128*1024**2
        return max(1,int(free//4//bytes_per_item))
    def memory(self):
        self.synchronize()
        if self.device.type!='cuda':return {'device':str(self.device),'peak_allocated_bytes':0,'peak_reserved_bytes':0}
        free,total=torch.cuda.mem_get_info(self.device)
        allocated=torch.cuda.max_memory_allocated(self.device);reserved=torch.cuda.max_memory_reserved(self.device)
        if allocated>total or reserved>total:raise MemoryError('Torch CUDA peak exceeds reported usable device memory')
        return {'device':str(self.device),'name':torch.cuda.get_device_name(self.device),'peak_allocated_bytes':allocated,
                'peak_reserved_bytes':reserved,'usable_device_bytes':total,'free_bytes_at_report':free,'torch_peak_within_device_capacity':True}

    def scene_memory_preflight(self,audit,cameras):
        n0=sum(audit['foreground_valid_counts'][0]);pixels=max(c.width*c.height for c in cameras)
        return {'frame_zero_raw_point_upper_bound':n0,'total_valid_foreground_observations':audit['foreground_valid_observations'],
                'persistent_all_frame_xyz_catalogue_bytes':0,'initial_point_xyz_rgb_group_ids_bytes_upper_bound':n0*(3*4+3*4+4+8+8),
                'host_conditional_point_covariance_bytes_upper_bound':n0*6*6*8,
                'largest_camera_pixels':pixels,'per_camera_zbuffer_index_depth_bytes':pixels*(8+4),
                'iekf_group_information_bytes':12*12*8+12*8,'ukf_group_moment_bytes':25*25*8+25*8,
                'bounded_transient_item_budget_bytes':self.chunk_size(1),'policy':'Transient chunks use at most one quarter of measured currently free memory; no point count cap/subsampling. Full-map projection and solver peaks are measured separately during execution.',
                'device':self.memory()}


def numpy(value):return value.detach().cpu().numpy()


def stable_segment_sum(values,keys,count):
    """Sort/segment, avoiding nondeterministic floating atomic additions."""
    result=torch.zeros((count,*values.shape[1:]),dtype=values.dtype,device=values.device)
    if len(keys):
        order=torch.argsort(keys,stable=True);unique,lengths=torch.unique_consecutive(keys[order],return_counts=True)
        reduced=torch.segment_reduce(values[order],'sum',lengths=lengths)
        result[unique]=reduced
    return result
