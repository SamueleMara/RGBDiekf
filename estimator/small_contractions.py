"""STEP 2B-B1 (opt-in, default OFF): tiny batched float64 products as broadcast multiply + sum contractions.

On this GPU cuBLAS batched GEMM on [K,3,3]-sized float64 operands is slow (measured J^T P J over 130k contributions:
24.5 ms vs 0.78 ms; reports/phase1_gpu_profile.md). The contractions evaluate the same sums in another order, so results
agree to float64 round-off, not bitwise. OFF (the default) every helper evaluates exactly the original expression.
Enabled per frame update by IEKFConfig.small_contractions (image_iekf.frame_step)."""
import contextlib

_ON = False


@contextlib.contextmanager
def enabled(flag=True):
    global _ON
    old, _ON = _ON, bool(flag)
    try: yield
    finally: _ON = old


def is_on():
    return _ON


def matvec(A, x):
    """A [...,m,n], x [...,n] -> A x [...,m]."""
    return (A*x[..., None, :]).sum(-1) if _ON else (A@x[..., None])[..., 0]


def rmatvec(A, y):
    """A [...,m,n], y [...,m] -> A^T y [...,n]."""
    return (A*y[..., :, None]).sum(-2) if _ON else (A.transpose(-1, -2)@y[..., None])[..., 0]


def congruence(A, P):
    """A [...,m,n], P [m,m] or [...,m,m] -> A^T P A [...,n,n]."""
    if not _ON: return A.transpose(-1, -2)@P@A
    PA = (P[..., :, :, None]*A[..., None, :, :]).sum(-2)
    return (A[..., :, :, None]*PA[..., :, None, :]).sum(-3)


def gram(A, B):
    """A [...,m,n], B [...,m,k] -> A^T B [...,n,k]."""
    return (A[..., :, :, None]*B[..., :, None, :]).sum(-3) if _ON else A.transpose(-1, -2)@B
