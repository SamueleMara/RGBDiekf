#!/usr/bin/env python3
"""OFFLINE ORACLE (Arm G): shared part motion with GT PART MEMBERSHIP ONLY; motion solved from measured RGB-D only.

GT privileges in this executable (and nowhere else): (1) which catalogue points form the moving part and the static
part (NN part-cloud label AND multi-frame on-surface vote agree; disagreeing points excluded), (2) post-hoc scoring.
The GT axis, origin, amplitude and per-frame transform are NEVER given to the estimator: the twist starts at zero and is
solved by temporal_part_twist.solve_group from the existing material-following RGB-D rows, frame 0->1 with the measured
catalogue as reference, then 1->2 with references RE-COMMITTED at the ESTIMATED frame-1 positions (non-GT).
Non-members stay at their reference position (the oracle hypothesis that everything outside the part is static).

Arms per scene and pair: translation-only (prismatic hypothesis), full 6D twist, true-static control group, camera
subsets (best single / pairs / 4 / 8), RGB-only / depth-only / RGB+D, a deliberately wrong merge (moving + static
members), and the objective at xi = 0, xi_GT and random twists of the same magnitude on identical support.
Noise: declared sensor noise, and the same-instant cross-camera discrepancy (no GT) of the measurement study.

    python -B tools/audit_rgbd_part_motion_oracle.py --scenes blade_103706,USB_100109 --output-root OUT
"""
import argparse
import datetime
import json
import math
import sys
import uuid
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io import json_write, serial

REPO = Path(__file__).resolve().parents[1]
DATA = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear')


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {'p50': float(np.median(x)), 'p90': float(np.quantile(x, .9)), 'p99': float(np.quantile(x, .99)), 'max': float(x.max()), 'n': int(len(x))} if len(x) else None


class Oracle:
    def __init__(self, scene, device):
        from scene.dataset import Dataset
        from scene.config import load_config
        from scene.endpoint import endpoint_samples
        from estimator.projective_camera import CameraRig
        from estimator.projective_sensor import MeasuredFrame, SensorNoise
        from estimator.temporal_noise import temporal_sensor_only
        from tools.audit_temporal_dynamic_correspondence import ground_truth, verified_labels
        from tools.evaluate_projective_gt import gt_joint, gt_transform
        self.dev = device
        self.policy = json.loads((REPO/'configs/projective_rgbd.json').read_text())
        self.alpha = float(json.loads((REPO/'configs/image_joint_policy.json').read_text())['tail_probability'])
        self.data = Dataset(scene, 3)
        cfg, self.pol, self.calib = temporal_sensor_only(self.data, load_config(), self.policy)
        self.noise = SensorNoise.from_config(cfg, self.pol, device, torch.float64)
        self.rig = CameraRig(self.data.cameras, device, torch.float64)
        _, _, xyz, _, _, self.source = endpoint_samples(self.data, 'first', cfg)
        self.X0 = torch.as_tensor(xyz, device=device)
        Xgt, nn_moving, _, self.evidence = ground_truth(scene, self.data, xyz)
        total = len(Dataset(scene).times)
        v, _, self.label_report = verified_labels(scene, self.rig, self.noise, self.pol, xyz, nn_moving, device, sorted({total//2, total-1}))
        self.Xgt = Xgt                                     # POST-HOC scoring only
        self.moving = torch.as_tensor(np.flatnonzero(v == 1), device=device)
        self.static = torch.as_tensor(np.flatnonzero(v == 0), device=device)
        self.frames = {k: MeasuredFrame(self.data, k, device, torch.float64) for k in range(3)}
        self.dt = float(self.data.times[1]-self.data.times[0])
        kind, axis, origin, _ = gt_joint(scene)
        vals = np.genfromtxt(scene/'metadata/frame_values.csv', delimiter=',', names=True)
        qv = np.asarray(vals[vals.dtype.names[1]], float)[list(self.data.frame_ids)]
        self.gt = {'kind': kind, 'axis': axis, 'origin': origin, 'q': qv, 'transform': lambda a, b: gt_transform(kind, axis, origin, qv[b]-qv[a])}

    def references(self, X_ref, frame_k):
        """Camera-local material references at frame_k from the given (measured or ESTIMATED) positions: first-hit winner
        on its own measured surface (commit_material_references); frame 0 also uses the source catalogue pixel."""
        from estimator.temporal_reference import ReferenceTable
        from estimator.temporal_zbuffer import first_hit_pixel_owner
        from estimator.temporal_material_terms import commit_material_references
        refs = ReferenceTable(self.rig.count, len(X_ref), self.dev, torch.float64)
        s = torch.zeros((len(X_ref), 6), dtype=torch.float64, device=self.dev)
        s[:, :3] = X_ref
        own = first_hit_pixel_owner(X_ref, self.rig, self.frames[frame_k])
        commit_material_references(refs, own, self.frames[frame_k], s, 0, torch.ones(len(X_ref), dtype=torch.bool, device=self.dev), self.rig, self.noise, self.pol)
        if frame_k == 0:
            refs.initialize_from_source_catalogue(self.source, 0, s)
        return refs

    def cross_camera_variance(self):
        from tools.study_material_measurement_information import Sampler, robust_var, allowance
        from estimator.temporal_zbuffer import first_hit_pixel_owner
        F = self.frames[0]
        own = first_hit_pixel_owner(self.X0, self.rig, F, dense=False)
        sc = torch.as_tensor(self.source[:, 1], device=self.dev)
        sp = torch.as_tensor(self.source[:, 2], device=self.dev)
        rgb_src = F.rgb[sc, sp//F.W, sp % F.W].double()
        sm = Sampler(F, self.rig, self.noise, self.pol)
        ds, rs = [], []
        for c in range(self.rig.count):
            j = torch.where((own.reason[c] == 0) & (sc != c))[0]
            cc = torch.full((len(j),), c, device=self.dev)
            u, z, _, _ = self.rig.project_rows(self.X0[j], cc)
            val, _, _, ok, _, _ = sm(cc, u)
            ok &= (val[:, 3]-z).abs() <= allowance(z, float(self.rig.K[c, :2].min()), self.noise, self.pol, math.sqrt(2))
            ds.append(val[ok, 3]-z[ok])
            rs.append(val[ok, :3]-rgb_src[j[ok]])
        d, r = torch.cat(ds), torch.cat(rs)
        return [robust_var(r[:, ch]) for ch in range(3)]+[robust_var(d)]

    def gt_twist(self, a, b, o):
        """GT velocity twist about pivot o (post-hoc only): Exp(dt xi) = [R, (I-R)(p-o)+t_gt]."""
        from estimator import se3
        T = torch.as_tensor(self.gt['transform'](a, b), device=self.dev)
        R, t = T[:3, :3], T[:3, 3]
        Tc = torch.eye(4, dtype=torch.float64, device=self.dev)
        Tc[:3, :3] = R
        Tc[:3, 3] = R@o+t-o
        return se3.log(Tc)/(self.dt*(b-a))

    def solve(self, X_ref_all, refs, members, frame_t, ordinal, extra=None, translation_only=False, channels=(0, 1, 2, 3), cams=None):
        from estimator.temporal_part_twist import material_evaluator, solve_group
        if cams is not None:
            refs = self._restrict(refs, cams)
        frames = {k: self.frames[k] for k in range(frame_t+1)}
        ev = material_evaluator(self.rig, self.noise, self.pol, torch.zeros((4, 4), dtype=torch.float64, device=self.dev), self.frames[frame_t],
                                frames, refs, X_ref_all, members, ordinal=ordinal, lag=3, extra_variance=extra, channels=channels)
        o = X_ref_all[members].mean(0)
        res = solve_group(ev, X_ref_all[members], o, self.dt, translation_only=translation_only)
        res['o'], res['evaluate'] = o, ev
        return res

    def _restrict(self, refs, cams):
        from estimator.temporal_reference import ReferenceTable
        r = ReferenceTable(*refs.frame.shape, self.dev, torch.float64)
        for att in ('frame', 'ordinal', 'pixel', 'state'):
            getattr(r, att).copy_(getattr(refs, att))
        drop = [c for c in range(self.rig.count) if c not in cams]
        r.frame[drop] = -1
        return r

    def score(self, res, members, a, b):
        from estimator.temporal_part_twist import point_covariance, membership_residual_test
        Xt = res['X_t']
        gt = torch.as_tensor(self.Xgt[b], device=self.dev)[members]
        e = (Xt-gt).norm(dim=1)
        P, _ = point_covariance(res['xi'], self.dt, Xt, res['o'], res['P_xi'], translation_only=len(res['P_xi']) == 3)
        d = (Xt-gt)
        nees = torch.einsum('ni,nij,nj->n', d, torch.linalg.pinv(P), d) if len(res['P_xi']) == 6 else None
        xi_gt = self.gt_twist(a, b, res['o'])
        test = membership_residual_test(res['rows'], len(members), self.alpha)
        return {'members': int(len(members)), 'rows': int(len(res['rows']['key'])), 'rank': res['rank'], 'iterations': len(res['history']),
                'xi_estimated': res['xi'].tolist(), 'xi_GT_posthoc': xi_gt.tolist(),
                'v_o_error_m_s': float((res['xi'][:3]-xi_gt[:3]).norm()), 'omega_error_rad_s': float((res['xi'][3:]-xi_gt[3:]).norm()),
                'xi_std': res['P_xi'].diagonal().clamp_min(0).sqrt().tolist(),
                'information_eigenvalues': res['information_eigenvalues'].tolist(),
                'condition_number': float(res['information_eigenvalues'].max()/res['information_eigenvalues'].min().clamp_min(1e-300)),
                'member_error_mm': q(1e3*e.cpu().numpy()), 'member_true_displacement_mm': q(1e3*(gt-torch.as_tensor(self.Xgt[a], device=self.dev)[members]).norm(dim=1).cpu().numpy()),
                'member_NEES_3d': q(nees.cpu().numpy()) if nees is not None else None,
                'membership_test': {'status': test['status'], 'failed_member_fraction': float(test['failed_members'].double().mean())}}


def objective_at(oracle, res, xis):
    """Total cost on identities valid at ALL evaluated twists (identical support)."""
    from estimator.temporal_part_twist import transform, apply
    X_ref = res['X_ref']
    evs = [res['evaluate'](apply(transform(x, oracle.dt), X_ref, res['o'])) for x in xis]
    common = evs[0]['key']
    for e in evs[1:]:
        common = common[torch.isin(common, e['key'])]
    return [float(e['cost'][torch.isin(e['key'], common)].sum()) for e in evs], int(len(common))


def run_scene(name, a, out):
    dev = torch.device(a.device)
    o = Oracle(a.data_root/name, dev)
    xvar = o.cross_camera_variance()
    rep = {'scene': name, 'oracle_privileges': 'GT part membership + post-hoc scoring only; motion from measured RGB-D; twist starts at 0',
           'label_verification': o.label_report, 'moving_members': int(len(o.moving)), 'static_members': int(len(o.static)),
           'gt_joint_kind_posthoc': o.gt['kind'], 'cross_camera_variance_rgb_depth': xvar, 'pairs': []}
    X_ref = o.X0.clone()
    refs = o.references(X_ref, 0)
    rng = np.random.default_rng(0)
    for (ta, tb) in ((0, 1), (1, 2)):
        pr = {'pair': [ta, tb], 'arms': {}}
        for noise_name, extra in (('declared', None), ('xcam', xvar)):
            for arm, tonly in (('translation_only', True), ('twist_6d', False)):
                res = o.solve(X_ref, refs, o.moving, tb, 1, extra, tonly)
                pr['arms'][f'moving/{arm}/{noise_name}'] = o.score(res, o.moving, ta, tb)
                if noise_name == 'declared' and arm == 'twist_6d':
                    keep6 = res
            st = o.solve(X_ref, refs, o.static, tb, 1, extra, False)
            pr['arms'][f'static_control/twist_6d/{noise_name}'] = o.score(st, o.static, ta, tb)
            print(f'[oracle] {name} {ta}->{tb} {noise_name} done', flush=True)
        # wrong merge: moving + 10% of the static members
        extra_static = o.static[torch.as_tensor(rng.choice(len(o.static), max(1, len(o.static)//10), replace=False), device=dev)]
        wrong = torch.cat((o.moving, extra_static))
        wr = o.solve(X_ref, refs, wrong, tb, 1, None, False)
        sc = o.score(wr, wrong, ta, tb)
        drag = (wr['X_t'][len(o.moving):]-X_ref[extra_static]).norm(dim=1)
        sc['static_members_false_motion_mm'] = q(1e3*drag.cpu().numpy())
        pr['arms']['wrong_merge_moving_plus_10pct_static/twist_6d/declared'] = sc
        # ablations (declared noise, 6D twist): camera subsets and channels
        abl = {}
        for label, cams in [(f'cam{c}', [c]) for c in range(o.rig.count)]+[('cams0,4', [0, 4]), ('cams2,6', [2, 6]), ('cams1,3,5,7', [1, 3, 5, 7]), ('all8', list(range(8)))]:
            r_ = o.solve(X_ref, refs, o.moving, tb, 1, None, False, cams=cams)
            s_ = o.score(r_, o.moving, ta, tb)
            abl[label] = {k: s_[k] for k in ('rows', 'rank', 'condition_number', 'v_o_error_m_s', 'omega_error_rad_s', 'member_error_mm', 'information_eigenvalues')}
        for label, ch in (('rgb_only', (0, 1, 2)), ('depth_only', (3,)), ('rgb_and_depth', (0, 1, 2, 3))):
            r_ = o.solve(X_ref, refs, o.moving, tb, 1, None, False, channels=ch)
            s_ = o.score(r_, o.moving, ta, tb)
            abl[label] = {k: s_[k] for k in ('rows', 'rank', 'condition_number', 'v_o_error_m_s', 'omega_error_rad_s', 'member_error_mm')}
        pr['ablations_moving_twist_6d_declared'] = abl
        # objective at zero, GT and random twists (identical support), moving 6D declared
        keep6['X_ref'] = X_ref[o.moving]
        xi_gt = o.gt_twist(ta, tb, keep6['o'])
        rand = []
        for _ in range(4):
            u = torch.as_tensor(rng.normal(size=6), device=dev)
            rand.append(u/u.norm()*xi_gt.norm())
        J, ncommon = objective_at(o, keep6, [torch.zeros(6, dtype=torch.float64, device=dev), xi_gt, keep6['xi']]+rand)
        pr['objective_identical_support'] = {'common_rows': ncommon, 'J_zero': J[0], 'J_GT': J[1], 'J_estimate': J[2], 'J_random_same_magnitude': J[3:],
                                             'GT_preferred_over_zero': J[1] < J[0], 'GT_preferred_over_all_random': all(J[1] < x for x in J[3:])}
        rep['pairs'].append(pr)
        # next pair: references re-committed at the ESTIMATED frame-1 positions (6D twist, declared noise) -- no GT
        X_next = X_ref.clone()
        X_next[o.moving] = keep6['X_t']
        X_ref = X_next
        refs = o.references(X_ref, tb)
    json_write(out/f'{name}.json', serial(rep))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes', default='blade_103706,USB_100109')
    ap.add_argument('--data-root', type=Path, default=DATA)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = a.output_root/f'part_oracle_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True)
    for name in [s for s in a.scenes.split(',') if s]:
        run_scene(name, a, out)
    json_write(out/'index.json', {'scenes': a.scenes.split(','), 'tool': 'tools/audit_rgbd_part_motion_oracle.py', 'ground_truth': 'membership + scoring only'})
    print(out)


if __name__ == '__main__':
    raise SystemExit(main())
