"""Shared rigid-part motion over EXPLICIT member IDs (experimental; never activated without supplied membership).

Chart (world frame, fixed pivot o_g, here the members' reference centroid; the pivot does not move):
    X_i(t) - o = R (X_i(r) - o) + t,      [R t; 0 1] = Exp(dt xi^),   xi = [v_o ; omega]  (estimator/se3.py order)
xi is a VELOCITY twist about the pivot, expressed in world axes. With the left Jacobian of SE(3) (se3.left_jacobian:
Exp(a + d) = Exp(J_l(a) d) Exp(a) to first order) the exact chain rule is
    dX_i(t)/dxi = B_i(X_i(t)) dt J_l(dt xi),      B_i(X) = [I_3 | -[X - o]_x]
(centred left action; an uncentred world-left perturbation would give -[X]_x instead). At xi = 0 this is dt B_i(X_i(r)).
Translation-only arm: omega = 0, dX_i/dv = dt I_3 for every member.

The group posterior is ONE latent twist: P_xi = (sum_rows H_xi^T R^-1 H_xi)^-1 (likelihood only, no invented prior on
xi); the point positions are deterministic functions of xi, so Cov(X_i, X_j) = J_i P_xi J_j^T (cross-point
correlation). Replaying the group measurement as independent per-point observations would be wrong (tests show the
variance it understates).
"""
import torch
from estimator import se3


def transform(xi, dt):
    return se3.exp(dt*xi)


def apply(T, X_ref, o):
    return (X_ref-o)@T[:3, :3].T+T[:3, 3]+o


def point_jacobian(xi, dt, X_t, o, translation_only=False):
    """dX_t/dxi (M,3,6) or (M,3,3) for the translation-only arm."""
    M = len(X_t)
    if translation_only:
        return dt*torch.eye(3, dtype=X_t.dtype, device=X_t.device).expand(M, 3, 3).clone()
    B = torch.zeros((M, 3, 6), dtype=X_t.dtype, device=X_t.device)
    B[:, :, :3] = torch.eye(3, dtype=X_t.dtype, device=X_t.device)
    B[:, :, 3:] = -se3.skew(X_t-o)
    return B@(dt*se3.left_jacobian(dt*xi))


def solve_group(evaluate, X_ref, o, dt, translation_only=False, iters=15, backtracks=20, xi0=None):
    """Likelihood-only Gauss-Newton for one shared twist.

    evaluate(X_members) -> dict with per-row 'member' (index into X_ref), 'H' (M,4,3) d residual / d X_t,
    'r' (M,4) residual, 'R' (M,4,4), 'key' (M,) stable identity code and 'cost' (M,) per-row negative log-likelihood.
    Steps are accepted on the identities valid at both states (rows that disappear never create descent)."""
    dim = 3 if translation_only else 6
    xi = torch.zeros(6, dtype=X_ref.dtype, device=X_ref.device) if xi0 is None else xi0.clone()
    if translation_only:
        xi[3:] = 0

    def state(x):
        Xt = apply(transform(x, dt), X_ref, o)
        ev = evaluate(Xt)
        J = point_jacobian(x, dt, Xt, o, translation_only)[ev['member']]
        Hx = ev['H']@J
        L = torch.linalg.cholesky(ev['R'])
        A = (Hx.transpose(1, 2)@torch.cholesky_solve(Hx, L)).sum(0)
        g = (Hx.transpose(1, 2)@torch.cholesky_solve(ev['r'][..., None], L))[..., 0].sum(0)
        return ev, A, g, Xt

    history = []
    ev, A, g, Xt = state(xi)
    for it in range(iters):
        step = -torch.linalg.lstsq(A, g[:, None]).solution[:, 0]
        full = torch.zeros(6, dtype=xi.dtype, device=xi.device)
        full[:dim] = step
        accepted = False
        for ls in range(backtracks):
            trial = xi+(.5**ls)*full
            et, At, gt, Xtt = state(trial)
            common_b = torch.isin(ev['key'], et['key'])
            common_t = torch.isin(et['key'], ev['key'])
            jb, jt = float(ev['cost'][common_b].sum()), float(et['cost'][common_t].sum())
            if jt <= jb+1e-12*abs(jb):
                accepted = True
                break
        history.append({'iteration': it, 'accepted': accepted, 'backtracks': ls, 'rows': int(len(ev['key'])),
                        'common_rows': int(common_b.sum()), 'cost_before': jb, 'cost_after': jt if accepted else None,
                        'xi': trial.tolist() if accepted else xi.tolist()})
        if not accepted:
            break
        moved = float((trial-xi).norm())
        xi, ev, A, g, Xt = trial, et, At, gt, Xtt
        if moved < 1e-12:
            break
    P = torch.linalg.pinv(A[:dim, :dim])
    ev_A = torch.linalg.eigvalsh(A[:dim, :dim])
    return {'xi': xi, 'P_xi': P, 'information_eigenvalues': ev_A, 'X_t': Xt, 'history': history, 'rows': ev,
            'rank': int((ev_A > ev_A.max()*torch.finfo(ev_A.dtype).eps*dim).sum()) if len(ev_A) else 0}


def point_covariance(xi, dt, X_t, o, P_xi, translation_only=False):
    """Per-member 3x3 marginal J_i P_xi J_i^T (cross-point blocks are J_i P_xi J_j^T)."""
    J = point_jacobian(xi, dt, X_t, o, translation_only)
    return J@P_xi@J.transpose(1, 2), J


def membership_residual_test(rows, members, alpha):
    """Per-member normalized residual sum and its chi-square tail test at the DECLARED tail probability alpha
    (configs/image_joint_policy.json): a member inconsistent with the shared motion is reported; GROUP_IDENTITY_INVALID
    when any member fails. Diagnostic of a wrong merge, never a silent fix."""
    from scipy.stats import chi2
    L = torch.linalg.cholesky(rows['R'])
    w = torch.cholesky_solve(rows['r'][..., None], L)[..., 0]
    nis = (rows['r']*w).sum(1)
    n = members
    tot = torch.zeros(n, dtype=nis.dtype, device=nis.device).index_add_(0, rows['member'], nis)
    dof = torch.zeros(n, dtype=nis.dtype, device=nis.device).index_add_(0, rows['member'], torch.full_like(nis, 4.))
    thr = torch.as_tensor(chi2.isf(alpha, dof.clamp_min(1).cpu().numpy()), dtype=nis.dtype, device=nis.device)
    bad = (dof > 0) & (tot > thr)
    return {'member_nis': tot, 'member_dof': dof, 'failed_members': bad,
            'status': 'GROUP_IDENTITY_INVALID' if bool(bad.any()) else 'GROUP_RESIDUALS_CONSISTENT'}


def material_evaluator(rig, noise, policy, warp, frame_t, frames, refs, X_all_ref, members, ordinal=1, lag=3, extra_variance=None, channels=(0, 1, 2, 3)):
    """evaluate(X_members) for solve_group from the EXISTING material-following residual (same rows, same pixels).

    All tracked points participate in first-hit ownership; non-members stay at X_all_ref (the hypothesis being tested).
    Residual f, d f / d X_t (= H_t[:, :, :3]), row noise sum_a w_a^2 R_cur + R_ref(p_ref) (+ optional declared extra
    variance, e.g. the same-instant cross-camera discrepancy) + R_warp; current/reference raw errors at 0 (no local
    Gaussian here: the group latent is the only state)."""
    from estimator.temporal_zbuffer import first_hit_pixel_owner
    from estimator.temporal_material_terms import build_material_rows, material_prediction
    from estimator.temporal_rgbd_terms import sensor_covariance
    N = len(X_all_ref)
    mem_index = torch.full((N,), -1, dtype=torch.long, device=X_all_ref.device)
    mem_index[members] = torch.arange(len(members), device=X_all_ref.device)
    C = rig.count

    def evaluate(X_members):
        s = torch.zeros((N, 6), dtype=torch.float64, device=X_all_ref.device)
        s[:, :3] = X_all_ref
        s[members, :3] = X_members
        own = first_hit_pixel_owner(s[:, :3], rig, frame_t, dense=False)
        rows, reason, life = build_material_rows(s, own, frame_t, refs, frames, rig, noise, policy, ordinal, lag)
        keep = mem_index[rows.point] >= 0
        f, Ht, _, w = material_prediction(rows, rows.current_values, rows.reference_values)
        cur = ((w*w)[:, :, None]*sensor_covariance(rows.current_values, noise).diagonal(dim1=-2, dim2=-1)).sum(1)
        ref = sensor_covariance(rows.reference_values, noise).diagonal(dim1=-2, dim2=-1)
        var = cur+ref+(0 if extra_variance is None else torch.as_tensor(extra_variance, dtype=cur.dtype, device=cur.device))
        R = torch.diag_embed(var)+warp
        L = torch.linalg.cholesky(R)
        wr = torch.cholesky_solve(f[..., None], L)[..., 0]
        cost = ((f*wr).sum(1)+2*torch.log(L.diagonal(dim1=1, dim2=2)).sum(1))/2
        k = keep
        if tuple(channels) != (0, 1, 2, 3):
            # channel ablation: excluded channels removed exactly (zero Jacobian and residual, unit variance, no cost)
            off = torch.ones(4, dtype=torch.bool, device=f.device)
            off[list(channels)] = False
            f, Ht = f.clone(), Ht.clone()
            f[:, off] = 0
            Ht[:, off] = 0
            R = R.clone()
            R[:, off, :] = 0
            R[:, :, off] = 0
            R[:, off, off] = 1
            L = torch.linalg.cholesky(R)
            wr = torch.cholesky_solve(f[..., None], L)[..., 0]
            cost = ((f*wr).sum(1)+2*torch.log(L.diagonal(dim1=1, dim2=2)).sum(1))/2
        return {'member': mem_index[rows.point[k]], 'H': Ht[k, :, :3], 'r': f[k], 'R': R[k], 'cost': cost[k],
                'key': rows.point[k]*C+rows.camera[k], 'reason': reason, 'life': life, 'shift': rows.pixel_shift[k]}
    return evaluate
