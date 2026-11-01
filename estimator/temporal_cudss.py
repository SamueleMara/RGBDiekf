"""cuDSS 1.0 sparse direct factorization on Torch-owned CUDA buffers.

No pivot perturbation or matrix regularization is permitted. The binding uses
the installed NVIDIA headers; it does not copy matrices through NumPy/CPU.
API reference: https://docs.nvidia.com/cuda/cudss/1.0.0/functions.html
"""
import ctypes as ct
from importlib import metadata
from pathlib import Path
import time
import torch


class SparseNumericalFailure(ArithmeticError):
    status = 'FULL_SOLVER_CERTIFICATE_FAILURE'


class SparseResourceLimit(SparseNumericalFailure):
    status='FULL_RESOURCE_LIMIT'


def load_library():
    distribution = metadata.distribution('nvidia-cudss-cu12')
    if distribution.version != '1.0.0.35':
        raise RuntimeError(f'Unsupported cuDSS ABI: {distribution.version}; expected 1.0.0.35')
    lib = ct.CDLL(str(distribution.locate_file('nvidia/cu12/lib/libcudss.so.1')))
    pointer, integer, size, i64 = ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_int64
    signatures = {
        'cudssCreate': [ct.POINTER(pointer)], 'cudssDestroy': [pointer],
        'cudssConfigCreate': [ct.POINTER(pointer)], 'cudssConfigDestroy': [pointer],
        'cudssDataCreate': [pointer, ct.POINTER(pointer)], 'cudssDataDestroy': [pointer, pointer],
        'cudssSetStream': [pointer, pointer],
        'cudssConfigSet': [pointer, integer, pointer, size],
        'cudssDataGet': [pointer, pointer, integer, pointer, size, ct.POINTER(size)],
        'cudssMatrixCreateCsr': [ct.POINTER(pointer), i64, i64, i64, pointer, pointer, pointer, pointer,
                                 integer, integer, integer, integer, integer, integer],
        'cudssMatrixCreateDn': [ct.POINTER(pointer), i64, i64, i64, pointer, integer, integer],
        'cudssMatrixDestroy': [pointer],
        'cudssExecute': [pointer, integer, pointer, pointer, pointer, pointer, pointer],
    }
    for name, arguments in signatures.items():
        function = getattr(lib, name); function.argtypes = arguments; function.restype = integer
    return lib


def checked(lib, name, *args):
    result = getattr(lib, name)(*args)
    if result:
        error=SparseResourceLimit if result==2 else SparseNumericalFailure
        raise error(f'{name}: cuDSS status {result}')


class CudaSparseFactor:
    def __init__(self, matrix):
        if matrix.device.type != 'cuda' or matrix.dtype != torch.float64 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('cuDSS requires a square CUDA float64 sparse matrix')
        self.lib = load_library()
        self.handle, self.config, self.data, self.A = (ct.c_void_p() for _ in range(4))
        self.matrix = matrix.to_sparse_csr()
        self.row = self.matrix.crow_indices().long().contiguous()
        if matrix.shape[0] >= 2**31:
            raise ValueError('cuDSS 1.0 requires fewer than 2^31 matrix rows')
        self.column = self.matrix.col_indices().int().contiguous()
        self.value = self.matrix.values().contiguous()
        self.profile = {'backend': 'cuDSS-1.0.0.35', 'rows': matrix.shape[0], 'nnz': len(self.value),
                        'pivot_epsilon': 0., 'formed_normal_equations': False}
        self._closed = False
        started = time.perf_counter()
        try:
            checked(self.lib, 'cudssCreate', ct.byref(self.handle))
            checked(self.lib, 'cudssSetStream', self.handle, torch.cuda.current_stream(matrix.device).cuda_stream)
            checked(self.lib, 'cudssConfigCreate', ct.byref(self.config))
            checked(self.lib, 'cudssDataCreate', self.handle, ct.byref(self.data))
            # cuDSS COLAMD's default symbolic capacity estimates 100*nnz. Bound
            # its storage by actual free VRAM; exceeding it is an explicit failure,
            # never permission to truncate the mathematical factorization.
            torch.cuda.empty_cache()
            available,total=torch.cuda.mem_get_info(matrix.device)
            capacity=max(1,int(available//8//16))
            value=ct.c_int64(capacity)
            checked(self.lib,'cudssConfigSet',self.config,10,ct.byref(value),ct.sizeof(value))
            self.profile.update(VRAM_free_before_analysis=available,VRAM_total=total,max_LU_nnz_capacity=capacity,
                torch_allocated_before_analysis=torch.cuda.memory_allocated(matrix.device),
                torch_reserved_before_analysis=torch.cuda.memory_reserved(matrix.device))
            # Config enums are fixed by the pinned installed cudss_data_types.h.
            for param, value in [(0, ct.c_int(1)), (3, ct.c_int(0)), (7, ct.c_int(3)),
                                 (9, ct.c_double(0.)), (20, ct.c_int(0)), (24, ct.c_int(0))]:
                checked(self.lib, 'cudssConfigSet', self.config, param, ct.byref(value), ct.sizeof(value))
            n = matrix.shape[0]
            checked(self.lib, 'cudssMatrixCreateCsr', ct.byref(self.A), n, n, len(self.value),
                    self.row.data_ptr(), None,
                    self.column.data_ptr(), self.value.data_ptr(), 24, 10, 1, 0, 0, 0)
            dummy = torch.zeros((n, 1), device=matrix.device, dtype=torch.float64)
            x, b = self._dense(dummy), self._dense(dummy)
            try:
                checked(self.lib, 'cudssExecute', self.handle, 3, self.config, self.data, self.A, x, b)
                torch.cuda.synchronize(matrix.device)
                self.profile['analysis_seconds'] = time.perf_counter() - started
                self._memory_profile()
                started = time.perf_counter()
                checked(self.lib, 'cudssExecute', self.handle, 4, self.config, self.data, self.A, x, b)
                torch.cuda.synchronize(matrix.device)
                self.profile['factorization_seconds'] = time.perf_counter() - started
                info, pivots = self._info(0, ct.c_int), self._info(2, ct.c_int)
                self.profile.update(info=info, perturbed_pivots=pivots, factor_nnz=self._info(1, ct.c_int64))
                if info or pivots:
                    raise SparseNumericalFailure(f'Uncertified cuDSS factorization (device error is not proof of mathematical singularity): info={info}, perturbed pivots={pivots}')
            finally:
                checked(self.lib, 'cudssMatrixDestroy', x); checked(self.lib, 'cudssMatrixDestroy', b)
        except BaseException as error:
            self.profile['VRAM_free_on_failure']=torch.cuda.mem_get_info(matrix.device)[0]
            self._memory_profile()
            if isinstance(error,SparseNumericalFailure):
                error.profile=dict(self.profile)
                error.args=(str(error)+'; sparse profile='+str(self.profile),)
            self.close()
            raise

    def _info(self, parameter, dtype):
        value, written = dtype(), ct.c_size_t()
        checked(self.lib, 'cudssDataGet', self.handle, self.data, parameter, ct.byref(value), ct.sizeof(value), ct.byref(written))
        return value.value

    def _memory_profile(self):
        if not self.data.value:return
        estimate=(ct.c_int64*16)();written=ct.c_size_t()
        code=self.lib.cudssDataGet(self.handle,self.data,13,ct.byref(estimate),ct.sizeof(estimate),ct.byref(written))
        self.profile['memory_estimate_query_status']=code
        if code==0:self.profile['native_memory_estimates_bytes']=list(estimate)
        count=ct.c_int64()
        code=self.lib.cudssDataGet(self.handle,self.data,1,ct.byref(count),ct.sizeof(count),ct.byref(written))
        if code==0:self.profile['native_LU_nnz_reported']=count.value

    def _dense(self, value):
        handle = ct.c_void_p()
        checked(self.lib, 'cudssMatrixCreateDn', ct.byref(handle), value.shape[0], value.shape[1], value.shape[0], value.data_ptr(), 1, 0)
        return handle

    def solve(self, rhs, relative_tolerance=2e-12, absolute_tolerance=2e-13):
        if self._closed:
            raise RuntimeError('Factorization is closed')
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        if rhs.shape[0] != self.matrix.shape[0] or rhs.device != self.matrix.device or rhs.dtype != torch.float64:
            raise ValueError('RHS must match sparse CUDA factorization')
        rhs = rhs.T.contiguous().T
        answer = self._solve_raw(rhs)
        absolute, backward = float('inf'), float('inf')
        for iteration in range(6):
            error = torch.sparse.mm(self.matrix, answer) - rhs
            scale = torch.sparse.mm(self.matrix.abs(), answer.abs()) + rhs.abs()
            normalized = error.abs() / scale.clamp_min(torch.finfo(torch.float64).tiny)
            absolute = float(error.abs().max()) if error.numel() else 0.
            backward = float(normalized.max()) if normalized.numel() else 0.
            valid=not bool((error.abs()>absolute_tolerance+relative_tolerance*scale).any())
            if not bool(torch.isfinite(answer).all()):
                break
            if valid:
                self.profile['last_solve'] = {'absolute_residual': absolute, 'componentwise_backward_error': backward,
                                             'rhs_columns': rhs.shape[1], 'refinement_steps': iteration,
                                             'mixed_absolute_relative_certificate':True,
                                             'relative_tolerance':relative_tolerance,'absolute_tolerance':absolute_tolerance}
                return answer
            if iteration < 5:
                answer -= self._solve_raw(error)
        raise SparseNumericalFailure(f'Uncertified cuDSS solve: absolute={absolute}, componentwise={backward}')

    def _solve_raw(self, rhs):
        rhs=rhs.T.contiguous().T
        answer = torch.empty_like(rhs)
        x, b = self._dense(answer), self._dense(rhs)
        try:
            checked(self.lib, 'cudssExecute', self.handle, 1008, self.config, self.data, self.A, x, b)
            torch.cuda.current_stream(self.matrix.device).synchronize()
            return answer
        finally:
            checked(self.lib, 'cudssMatrixDestroy', x); checked(self.lib, 'cudssMatrixDestroy', b)

    def close(self):
        if self._closed:
            return
        for function, handles in [('cudssMatrixDestroy', (self.A,)), ('cudssDataDestroy', (self.handle, self.data)),
                                  ('cudssConfigDestroy', (self.config,)), ('cudssDestroy', (self.handle,))]:
            if all(handle.value for handle in handles):
                getattr(self.lib, function)(*handles)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
