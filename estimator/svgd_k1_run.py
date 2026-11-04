"""Opt-in runner of the joint K = 1 static/moving-support + shared-rigid-motion posterior with SVGD over motion hypotheses
(CODEX_RGBDiekf_SVGD_JOINT_RIGID_SUPPORT_K1_2026-11-03.md). Independent of the earlier free-point svgd_support pipeline and of train.py.

Stages: LOAD (camera-aware materialization -> point-only boundary) -> CALIBRATE (representation scales, frozen b_0, held-out rho_t,
certified grid radius, effective evidence weights a_eff; no GT) -> STATIC (log L^S cached once) -> SVGD (M hypotheses of the ONE motion,
score = grad of the same F) -> MAP (control, same F) -> OUTPUTS. Every output is labelled UNCALIBRATED_APPROXIMATION until the
calibration gates pass; a --frame-limit run is PREFIX_DIAGNOSTIC.
"""
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import chi2
from estimator.svgd_k1_observation import Background, FrameKDE, representation_scales, calibrate_rho, choose_kappa, kernel_sum
from estimator.svgd_k1_motion import MotionPrior, compose
from estimator.svgd_k1_joint_posterior import JointPosterior
from estimator.svgd_k1_stein import svgd_direction, AdaGrad
from scene.svgd_k1_cloud_adapter import from_dataset

REPO = Path(__file__).resolve().parents[1]
SOURCES = ['scene/svgd_k1_cloud_adapter.py', 'estimator/svgd_k1_observation.py', 'estimator/svgd_k1_reference.py', 'estimator/svgd_k1_motion.py',
           'estimator/svgd_k1_joint_posterior.py', 'estimator/svgd_k1_stein.py', 'estimator/svgd_k1_run.py', 'tools/run_svgd_joint_k1.py',
           'scene/pointcloud_sequence.py', 'scene/endpoint.py', 'scene/dataset.py', 'scene/dataloader.py', 'scene/noise_calibration.py']


def V(cfg, dotted):
    node = cfg
    for k in dotted.split('.'): node = node[k]
    return node['value'] if isinstance(node, dict) and 'value' in node else node


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jdump(path, obj): Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def run(a, data=None):
    if a.num_moving_parts != 1: raise SystemExit('NOT_IMPLEMENTED_K_GT_1: this version supports exactly one moving part')
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads(Path(a.config).read_text()); dev = torch.device(a.device); dt = torch.float64
    if dev.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA requested but unavailable (no silent CPU fallback)')
    if data is None:
        from scene.dataset import Dataset
        data = Dataset(a.scene, a.frame_limit, None, fps=a.fps, dt_seconds=a.dt_seconds, timestamps_file=a.timestamps_file)
    out = Path(a.output); ok_dir = out/'svgd_joint_k1'
    if out.exists() and any(out.iterdir()): raise SystemExit(f'output {out} exists and is not empty')
    if REPO in out.resolve().parents: raise SystemExit('output must be outside the repository')
    ok_dir.mkdir(parents=True); t0 = time.time()
    prefix = a.frame_limit is not None and hasattr(data, 'base') and a.frame_limit < len(data.base.times)
    prog = lambda **kw: jdump(out/'progress.json', {**kw, 'elapsed_s': time.time()-t0})
    status = {'stage': 'LOAD', 'probability_semantics': 'UNCALIBRATED_APPROXIMATION', 'sequence_scope': 'PREFIX_DIAGNOSTIC' if prefix else 'WHOLE_SEQUENCE',
              'num_moving_parts': 1, 'gates': {}}
    git = lambda *x: subprocess.run(['git', *x], cwd=REPO, capture_output=True, text=True).stdout.strip()
    man = {'run_kind': 'svgd_joint_k1', 'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': git('status', '--short').splitlines(), 'config': str(a.config),
           'config_sha256': sha(a.config), 'source_sha256': {s: sha(REPO/s) for s in SOURCES if (REPO/s).is_file()}, 'scene': str(a.scene),
           'arguments': {k: str(v) for k, v in vars(a).items()}, 'device': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu',
           'ground_truth_used': False}
    jdump(out/'run_manifest.json', man)
    perf = {'stages': {}}
    def stage(name): status['stage'] = name; jdump(ok_dir/'status.json', status); perf['stages'][name] = time.time()
    model = build_model(cfg, data, a, dev, ok_dir, prog, stage)
    post, prior, seq, a_eff, beta, T, N, F0_inputs = model['post'], model['prior'], model['seq'], model['a_eff'], model['beta'], model['T'], model['N'], None
    lS = post.lS; neff = model['n_eff']; P0 = seq.reference
    # ------------------------------------------------ SVGD over shared-motion hypotheses
    sv = V(cfg, 'svgd'); Mp, iters = int(sv['particles']), int(sv['iterations']); g = torch.Generator(device='cpu').manual_seed(int(sv['seed']))
    sig_h = float(sv['sigma_h']); h0 = math.log(beta[0]/beta[1])
    d6 = 6*T
    Vp = torch.randn(Mp, d6+1, generator=g, dtype=dt).to(dev)
    Vp[:, -1] = torch.randn(Mp, generator=g, dtype=dt).to(dev)
    unpack = lambda v: (prior.from_white(v[:d6]), h0+sig_h*v[-1])
    ada = AdaGrad(float(sv['step']), float(sv['alpha']), float(sv['fudge'])); trace = []; F0 = post.evaluate(torch.zeros(T, 6, dtype=dt, device=dev), torch.tensor(h0, dtype=dt, device=dev), grad=False)['F']
    stage('SVGD'); t_it = time.time(); last = None
    for it in range(iters+1):
        res = []
        for m in range(Mp):
            prog(stage='SVGD', iteration=it, iterations=iters, particle=m+1, particles=Mp, sec_per_iteration=last)
            nu, h = unpack(Vp[m]); res.append(post.evaluate(nu, h, grad=it < iters))
        Fs = torch.tensor([r['F'] for r in res]); gam = torch.stack([r['gamma'] for r in res])
        bad = [m for m, r in enumerate(res) if not (math.isfinite(r['F']) and (it == iters or bool(torch.isfinite(r['grad_nu']).all() and torch.isfinite(r['grad_h']))))]
        if bad:
            status.update(stage='NUMERICAL_FAILURE', failure={'iteration': it, 'particles': bad, 'reason': 'non-finite F or score'}); jdump(ok_dir/'status.json', status)
            raise FloatingPointError(f'NUMERICAL_FAILURE: non-finite F/score at iteration {it}, particles {bad}')
        row = {'iteration': it, 'F_mean': float(Fs.mean()), 'F_std': float(Fs.std()) if Mp > 1 else 0., 'F_max': float(Fs.max()), 'F_identity': F0,
               'gamma_mean': float(gam.mean()), 'gamma_particle_std_mean': float(gam.std(0).mean()) if Mp > 1 else 0., 'pi_mean': float(torch.sigmoid(torch.stack([unpack(v)[1] for v in Vp])).mean()),
               'qmc_fallback_max': max(r['diag']['qmc_fallback_points'] for r in res), 'certificate_max': max(r['diag']['certificate_max_sum_over_frames'] for r in res)}
        if it == iters: trace.append(row); break
        G = torch.stack([torch.cat([(prior.L.T@r['grad_nu'].reshape(-1)), (sig_h*r['grad_h']).reshape(1)]) for r in res])
        phi, kd = svgd_direction(Vp, G); Vp = Vp+ada.step(phi)
        row.update(kd, score_norm_mean=float(G.norm(dim=1).mean()), particle_diversity=float(torch.pdist(Vp).mean()) if Mp > 1 else 0., seconds=time.time()-t_it)
        trace.append(row); last = time.time()-t_it; t_it = time.time()
        print(f"[k1] SVGD {it+1}/{iters}: F mean {row['F_mean']:.6g} max {row['F_max']:.6g} (identity {F0:.6g}) pi {row['pi_mean']:.3f} |score| {row['score_norm_mean']:.3g} ({last:.1f} s)", flush=True)
    final = res
    # ------------------------------------------------ MAP control (same F)
    mapc = V(cfg, 'map'); map_trace = []; map_runs = []
    if mapc['enabled']:
        stage('MAP')
        starts = {'best_svgd_particle': Vp[int(torch.argmax(Fs))].clone(), 'identity': torch.cat([torch.zeros(d6, dtype=dt, device=dev), torch.zeros(1, dtype=dt, device=dev)])}
        for name in mapc.get('starts', ['best_svgd_particle']):
            v = starts[name].clone(); opt_m = torch.zeros_like(v); opt_v = torch.zeros_like(v); b1, b2, lr = .9, .999, float(mapc['lr'])
            for it in range(int(mapc['iterations'])):
                prog(stage='MAP', start=name, iteration=it, iterations=int(mapc['iterations']))
                nu, h = unpack(v); r = post.evaluate(nu, h)
                gv = torch.cat([prior.L.T@r['grad_nu'].reshape(-1), (sig_h*r['grad_h']).reshape(1)])
                opt_m = b1*opt_m+(1-b1)*gv; opt_v = b2*opt_v+(1-b2)*gv*gv
                v = v+lr*(opt_m/(1-b1**(it+1)))/(torch.sqrt(opt_v/(1-b2**(it+1)))+1e-12); map_trace.append({'start': name, 'iteration': it, 'F': r['F'], 'grad_norm': float(gv.norm())})
            nu, h = unpack(v); rr = post.evaluate(nu, h, grad=False); map_trace.append({'start': name, 'iteration': 'final', 'F': rr['F']}); map_runs.append((rr['F'], name, v.clone(), rr))
            print(f"[k1] MAP from {name}: F {rr['F']:.6g}", flush=True)
        _, map_start, v, map_res = max(map_runs, key=lambda z: z[0])
    # ------------------------------------------------ OUTPUTS (spec section 6)
    stage('OUTPUTS')
    gam = torch.stack([r['gamma'] for r in final]); B = torch.stack([r['B'] for r in final]); lM = torch.stack([r['logLM'] for r in final])
    poses = torch.stack([compose(unpack(v)[0], prior.dts) for v in Vp])
    num_err = max(r['diag']['certificate_max_sum_over_frames'] for r in final)*float(a_eff.max())
    ident = ((B.abs().max(0).values) > num_err).to(torch.int8)                  # 1 = informative beyond the certified numerical error, 0 = no information
    g_mean = gam.mean(0).cpu().numpy(); order = np.argsort(-g_mean, kind='stable'); alpha = float(V(cfg, 'seed_selection.alpha'))
    fdr = np.cumsum(1-g_mean[order])/np.arange(1, N+1); k_sel = int(np.nonzero(fdr <= alpha)[0].max()+1) if (fdr <= alpha).any() else 0
    np.save(ok_dir/'gamma_provisional.npy', g_mean); np.save(ok_dir/'evidence_static.npy', lS.cpu().numpy())
    np.save(ok_dir/'evidence_moving_conditional.npy', lM.mean(0).cpu().numpy()); np.save(ok_dir/'log_evidence_ratio.npy', B.mean(0).cpu().numpy())
    np.save(ok_dir/'log_evidence_ratio_particle_std.npy', B.std(0).cpu().numpy() if Mp > 1 else np.zeros(N)); np.save(ok_dir/'identifiability_status.npy', ident.cpu().numpy())
    np.save(ok_dir/'pose_particles.npy', poses.cpu().numpy()); np.save(ok_dir/'particle_F.npy', Fs.numpy())
    np.save(ok_dir/'moving_seed_indices.npy', order[:k_sel].astype(np.int64)); np.save(ok_dir/'reference_ids.npy', seq.reference_ids)
    np.savez_compressed(ok_dir/'reference_catalogue.npz', xyz=P0.xyz, rgb=P0.rgb, cov_xyz=P0.cov_xyz, a_eff=a_eff.cpu().numpy())
    if mapc['enabled']:
        np.save(ok_dir/'map_pose.npy', compose(unpack(v)[0], prior.dts).cpu().numpy()); np.save(ok_dir/'map_gamma.npy', map_res['gamma'].cpu().numpy())
        np.save(ok_dir/'map_log_evidence_ratio.npy', map_res['B'].cpu().numpy())
        gm = map_res['gamma'].cpu().numpy(); om = np.argsort(-gm, kind='stable'); fm = np.cumsum(1-gm[om])/np.arange(1, N+1)
        km = int(np.nonzero(fm <= alpha)[0].max()+1) if (fm <= alpha).any() else 0
        np.save(ok_dir/'map_moving_seed_indices.npy', om[:km].astype(np.int64)); status['map_seeds'] = km   # CONTROL: conditional gamma at the MAP pose
    jdump(ok_dir/'optimizer_diagnostics.json', {'svgd': trace, 'map': map_trace, 'F_identity': F0, 'semantics': 'F is a conditional Gibbs log-posterior, not a calibrated evidence'})
    status.update(stage='DONE', seeds=k_sel, seed_alpha=alpha, numerical_error_bound_nats=num_err, informative_points=int(ident.sum()),
                  F_identity=F0, F_svgd_best=float(Fs.max()), F_map=map_res['F'] if mapc['enabled'] else None, map_best_start=map_start if mapc['enabled'] else None,
                  pi_posterior_mean=float(torch.sigmoid(torch.stack([unpack(v)[1] for v in Vp])).mean()))
    perf['wall_seconds'] = time.time()-t0
    if dev.type == 'cuda': perf.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(dev), peak_reserved_bytes=torch.cuda.max_memory_reserved(dev))
    jdump(ok_dir/'performance.json', perf); jdump(ok_dir/'status.json', status); prog(stage='DONE')
    print(f"[k1] DONE seeds {k_sel} (alpha {alpha}), F identity {F0:.6g} SVGD best {status['F_svgd_best']:.6g} MAP {status['F_map']}, {perf['wall_seconds']:.0f} s", flush=True)
    return 0


def build_model(cfg, data, a, dev, ok_dir=None, prog=None, stage=None, verbose=True):
    """Everything up to the cached static evidence (shared by the runner and the post-hoc oracle; no GT anywhere)."""
    from scene.config import load_config
    from estimator.temporal_noise import temporal_sensor_only
    dt = torch.float64
    # ------------------------------------------------ LOAD + noise (same routine for synthetic and real)
    config = load_config(getattr(a, 'noise_config', None)); noise = {'sensor_noise': a.sensor_noise}
    if a.sensor_noise == 'measured': config, _, calib = temporal_sensor_only(data, config, {}); noise['temporal_sensor'] = calib
    (prog or (lambda **k: None))(stage='LOAD'); seq, counts = from_dataset(data, config)
    K = len(seq); T = K-1
    if T < 1: raise SystemExit('need at least two frames')
    P0 = seq.reference; N = P0.M
    if ok_dir: jdump(ok_dir/'loader_counts.json', counts)
    if verbose: print(f'[k1] LOAD N={N} reference points, {K} frames, M_t {min(f.M for f in seq.frames)}..{max(f.M for f in seq.frames)}', flush=True)
    # ------------------------------------------------ CALIBRATE (point clouds only; no GT)
    stage('CALIBRATE') if stage else None; tau = float(V(cfg, 'observation.certificate_tolerance_nats')); m2 = float(chi2.ppf(1-float(V(cfg, 'observation.domain_tail_probability')), 6))
    scales = [representation_scales(f.xyz, f.rgb, f.cov_rgb) for f in seq.frames]
    bg = Background(P0.xyz, P0.rgb, P0.cov_xyz, P0.cov_rgb, device=dev)
    rho_cfg = V(cfg, 'observation.rho')
    frames, rho_rep, kap = [], [], []
    for t in range(1, K):
        (prog or (lambda **k: None))(stage='CALIBRATE', frame=t, frames=T)
        fr = seq.frames[t]; s2, LC = scales[t]
        r = calibrate_rho(fr, s2, LC, bg, tau, m2, tuple(rho_cfg['prior_beta']), edge=float(rho_cfg['block_edge_spacing_factor'])*math.sqrt(s2),
                          n_folds=int(rho_cfg['folds']), device=dev)
        rho_rep.append(r); rho = r['rho']
        lw = None
        if V(cfg, 'observation.weights') == 'inverse_density':
            dens, _ = kernel_sum(fr.xyz, fr.rgb, s2*np.eye(3), LC, float(V(cfg, 'reference.a_eff.kappa')), device=dev)
            w = 1/dens.cpu().numpy(); lw = np.log(w/w.sum())
        kde = FrameKDE(fr.xyz, fr.rgb, fr.cov_xyz, s2, LC, log_w=lw, device=dev); k = choose_kappa(kde, bg, rho, tau, m2)
        kde.build_grid(k, tau, V(cfg, 'observation.first_pass_kappa')); kap.append(k)
        frames.append((kde, bg, rho))
    s2_0, LC_0 = scales[0]
    t_ = lambda x: torch.as_tensor(np.asarray(x, float), dtype=dt, device=dev)
    mu = torch.cat([t_(P0.xyz), t_(P0.rgb)], 1)
    Sig = torch.zeros(N, 6, 6, dtype=dt, device=dev); Sig[:, :3, :3] = t_(P0.cov_xyz); Sig[:, 3:, 3:] = t_(P0.cov_rgb)
    try: Lchol = torch.linalg.cholesky(Sig)
    except RuntimeError as e: raise FloatingPointError(f'NUMERICAL_FAILURE: reference covariance not SPD ({e})') from None
    ae = V(cfg, 'reference.a_eff')
    if ae['method'] == 'kernel_correlation':
        rho0, omit = kernel_sum(P0.xyz, P0.rgb, s2_0*np.eye(3), LC_0, float(ae['kappa']), device=dev)
        neff = N*N/float(rho0.sum()); inv = 1/rho0; a_eff = neff*inv/inv.sum()
    elif ae['method'] == 'unit': a_eff = torch.ones(N, dtype=dt, device=dev); neff = float(N); omit = 0.
    else: raise SystemExit(f"unknown a_eff method {ae['method']}")
    mp = V(cfg, 'motion_prior')
    dts = np.diff(np.asarray(seq.times, float))
    prior = MotionPrior(dts, np.diag(mp['P_nu0_diag']), np.diag(mp['Q_nu_diag']), device=dev)
    beta = tuple(V(cfg, 'support_prior.beta'))
    post = JointPosterior(frames, mu, Lchol, a_eff, prior, beta, tile=int(V(cfg, 'compute.tile')), qmc_nodes=int(V(cfg, 'reference.qmc_nodes')),
                          qmc_seed=int(V(cfg, 'reference.qmc_seed')), force_qmc=V(cfg, 'reference.integration') == 'crn_qmc')
    if V(cfg, 'reference.integration') not in ('laplace_at_prior_mean_with_crn_qmc_fallback', 'crn_qmc'): raise SystemExit('unknown reference.integration')
    calib = {'representation_spacing_m': [math.sqrt(s) for s, _ in scales], 'Lambda_C': [lc.tolist() for _, lc in scales], 'rho': rho_rep, 'kappa': kap,
             'grid_radius_m': [f[0].R for f in frames], 'background_mean': bg.mean.tolist(), 'background_cov': bg.cov.tolist(),
             'a_eff': {'method': ae['method'], 'n_eff': neff, 'min': float(a_eff.min()), 'median': float(a_eff.median()), 'max': float(a_eff.max()), 'omitted_bound': omit},
             'sensor': noise, 'certificate_tolerance_nats': tau, 'domain_mahalanobis2': m2, 'motion_prior': mp, 'support_prior_beta': beta}
    if ok_dir: jdump(ok_dir/'calibration_provenance.json', calib)
    if verbose: print(f"[k1] CALIBRATE rho median {np.median([r['rho'] for r in rho_rep]):.3g}, grid R {np.median([f[0].R for f in frames])*1e3:.1f} mm, n_eff {neff:.0f}", flush=True)
    # ------------------------------------------------ STATIC evidence (once)
    stage('STATIC') if stage else None; (prog or (lambda **k: None))(stage='STATIC'); lS = post.set_static()
    if verbose: print(f"[k1] STATIC log L^S cached ({post.static_diag})", flush=True)
    return {'post': post, 'prior': prior, 'seq': seq, 'a_eff': a_eff, 'beta': beta, 'T': T, 'N': N, 'n_eff': neff, 'calib': calib}
