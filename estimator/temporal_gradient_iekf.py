"""`--temporal-gradient-guide replace`: the material-following likelihood re-assembled from image-gradient velocity rows.

SAME physical model and SAME raw pixels as material-following (material anchor u_t = p_ref + pi(X_t) - pi(X_r)); only the
linearization is assembled by a separate code path:
    clean current support c_a = y_a - eta_a, bilinear value h(q) and exact interpolant gradient grad_u h(q) (own code),
    G_I(X) = grad_u I J_pi(X), a(X) = J_Z(X) - grad_u D J_pi(X)          (temporal_motion_geometry.velocity_rows)
    f      = h(q) - (y_ref - eta_ref) - e4 (Z(X_t) - Z(X_r))
    H_t    = [ G_I(X_t) ; -a(X_t) ]        H_ref = [ -G_I(X_r) ; +a(X_r) ]  (gradient at q_t for both)
It is a reparameterization, NOT new sensor information; tests prove parity with temporal_material_terms.material_prediction.
The u_t = pi(X_t) (subpixel reference anchor) model would be a different physical model and is not implemented here.
"""
import torch
from estimator.temporal_material_iekf import MaterialFrameUpdate
from estimator.temporal_motion_geometry import velocity_rows

EQUIVALENCE = 'EQUIVALENT_REPARAMETERIZATION_OF_MATERIAL_FOLLOWING (same pixels, same covariance; no new information)'


def clean_bilinear(values, q, cell):
    """h (M,4) and d h / d(u,v) (M,4,2) of the bilinear interpolant over a 2x2 support (00,10,01,11)."""
    u = q[:, 0]-cell[:, 0].to(q.dtype)
    v = q[:, 1]-cell[:, 1].to(q.dtype)
    F00, F10, F01, F11 = values.unbind(1)
    h = ((1-u)*(1-v))[:, None]*F00+(u*(1-v))[:, None]*F10+((1-u)*v)[:, None]*F01+(u*v)[:, None]*F11
    du = (1-v)[:, None]*(F10-F00)+v[:, None]*(F11-F01)
    dv = (1-u)[:, None]*(F01-F00)+u[:, None]*(F11-F10)
    w = torch.stack(((1-u)*(1-v), u*(1-v), (1-u)*v, u*v), -1)
    return h, torch.stack((du, dv), -1), w


def gradient_prediction(rig, rows, clean_current, clean_reference):
    h, grad, w = clean_bilinear(clean_current, rows.q, rows.cell)
    p, c = rows.point, rows.camera
    G_t, a_t, _, _ = velocity_rows(rig, c, rows.x_t, grad[:, :3], grad[:, 3])
    G_r, a_r, _, _ = velocity_rows(rig, c, rows.reference_state[:, :3], grad[:, :3], grad[:, 3])
    f = h-clean_reference
    f[:, 3] -= rows.z_t-rows.z_ref
    M = len(p)
    Ht = torch.zeros((M, 4, 6), dtype=f.dtype, device=f.device)
    Hr = torch.zeros_like(Ht)
    Ht[:, :3, :3], Ht[:, 3, :3] = G_t, -a_t
    Hr[:, :3, :3], Hr[:, 3, :3] = -G_r, a_r
    return f, Ht, Hr, w


class GradientMaterialFrameUpdate(MaterialFrameUpdate):
    def linearize(self, name='it', register=True):
        lr = super().linearize(name, register)
        rows = lr.rows
        rows.x_t = self.current_states(name)[rows.point, :3]
        eta_cur, _, _ = self._key_values(name, lr.cur_code)
        f, Ht, Hr, w = gradient_prediction(self.rig, rows, rows.current_values-eta_cur, lr.clean_reference)
        rows.f, rows.H_t, rows.H_ref, rows.weights = f, Ht, Hr, w
        lr.residual, lr.weights = -f, w
        return lr
