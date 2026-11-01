"""Reports and small measured-data figures; no estimation or policy selection."""
from pathlib import Path
import ast
import numpy as np
from tools.f1_definitive_evidence import read_json,write_json,csv_read,digest,REPO


def code_map(out):
    paths={
        'endpoint provenance':REPO/'scene/endpoint.py',
        'all-camera projection candidates':REPO/'estimator/point_kalman_cuda.py',
        'nearest-pixel ownership':REPO/'estimator/projective_visibility.py',
        'absolute RGB-D implicit measurement':REPO/'estimator/projective_terms.py',
        'RigEuler linearization cache':REPO.parent/'RigEuler3DGS/estimator/renderer_bridge.py',
        'RigEuler accepted-step re-render':REPO.parent/'RigEuler3DGS/estimator/sparse_iekf.py',
        'RigEuler fixed linear operator':REPO.parent/'RigEuler3DGS/submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf.h',
    }
    needles={'endpoint provenance':'def endpoint_samples','all-camera projection candidates':'cand = vreason.T',
             'nearest-pixel ownership':'def visibility','absolute RGB-D implicit measurement':'def build_terms',
             'RigEuler linearization cache':'def build_linearization','RigEuler accepted-step re-render':'lin = build_linearization',
             'RigEuler fixed linear operator':'sorting, tile membership'}
    lines=['# Verified local code map','',
           'RigEuler freezes contributors, sorting and Jacobian coefficients for one linear solve. Its nonlinear acceptance evaluates a new render, and accepted steps are re-rendered/re-sorted. This is not permanent frame-0 pixel ownership.',
           '', 'RGBDiekf preserves canonical source IDs in its catalogue, but PointMeasurement constructs candidate rows from geometric visibility and ownership, without a material-association test against those IDs. The implicit row then compares fixed canonical color and optical Z against the current projected image. This is a code fact, not proof that every cross-camera match is physically wrong.','']
    records={}
    for label,p in paths.items():
        if not p.exists():records[label]={'status':'UNAVAILABLE','path':str(p)};lines.append(f'- {label}: UNAVAILABLE `{p}`.');continue
        text=p.read_text();line=next((j+1 for j,s in enumerate(text.splitlines()) if needles[label] in s),None)
        records[label]={'path':str(p),'sha256':digest(p),'line':line};lines.append(f'- {label}: `{p}:{line}`, SHA256 `{records[label]["sha256"]}`.')
    (Path(out)/'rig_euler_vs_rgbdiekf_correspondence.md').write_text('\n'.join(lines)+'\n')
    write_json(Path(out)/'reference_code_hashes.json',records);return records


def owner_figure(sc,pid,k,probe,owner,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=int(sc.src[pid,1]);cam=sc.cams[c];u,v=probe['u'],probe['v'];xx,yy=int(np.floor(u)),int(np.floor(v));im=sc.images(k)[c]
    x0,y0=max(0,xx-7),max(0,yy-7);x1,y1=min(cam.W,xx+9),min(cam.H,yy+9)
    if x1<=x0 or y1<=y0:return
    fig=plt.figure(figsize=(12,4));a=fig.add_subplot(131);a.imshow(im.rgb[y0:y1,x0:x1]);a.plot(u-x0,v-y0,'rx',label=f'point {pid}')
    sf,scam,pix=map(int,sc.src[pid]);a.plot(pix%cam.W-x0,pix//cam.W-y0,'g+',label=f'source {sf}/{scam}')
    pu,pv=probe['nearest_pixel_u_v'];a.plot(pu-x0,pv-y0,'bo',fillstyle='none',label=f"winner {owner.get('owner_point_id')}; ownership pixel")
    a.plot([xx-x0,xx+1-x0,xx+1-x0,xx-x0,xx-x0],[yy-y0,yy-y0,yy+1-y0,yy+1-y0,yy-y0],'y-');a.legend(fontsize=7);a.set_title('measured RGB; yellow = 2×2 footprint')
    a=fig.add_subplot(132);a.imshow(np.where(im.mask[y0:y1,x0:x1],im.depth[y0:y1,x0:x1],np.nan));a.set_title('actual masked depth (m)')
    a=fig.add_subplot(133,projection='3d');x=np.asarray(sc.trace(k)['prediction']['points'][str(pid)]['s_minus'])[:3];a.scatter(*x,color='red',label=f'{pid}: predicted')
    winner=owner.get('owner_point_id')
    if winner is not None:
        a.scatter(*sc.x0[winner],color='blue',label=f'{winner}: canonical origin (not current)')
    a.scatter(*sc.x0[pid],color='green',label='point canonical origin');a.legend(fontsize=6);a.set(xlabel='world x (m)',ylabel='world y (m)',zlabel='world z (m)')
    fig.suptitle(f'{sc.name}/{pid}, camera {c}, frame {k}; source {sf}/{scam}/{pix}; A_ALL_RAW; frozen production ownership')
    fig.tight_layout();fig.savefig(Path(out)/'figures'/f'{sc.name}_{pid}_{k}_owner.png',dpi=125);plt.close(fig)


def trajectory_figure(scene,pid,records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4));selected=[r for r in records if r['scene']==scene and r['point_id']==pid and r['variant_parameters']['ownership']=='P' and r['variant_parameters']['variance_mode']=='ALGEBRA_ONLY' and r['variant_parameters']['missing_reference']=='retain' and r['variant_parameters']['ambiguous']=='admit']
    for variant in sorted({r['variant'] for r in selected}):
        rows=[r for r in selected if r['variant']==variant];t=[r['frame'] for r in rows];a=rows[0]['arm']
        axes[0].plot(t,[r['canonical_drift_m'] for r in rows],label=a);axes[1].plot(t,[np.linalg.norm(r['v_plus_m_s']) for r in rows]);axes[2].plot(t,[len(r['rows_accepted']) for r in rows])
    axes[0].set_ylabel('canonical displacement (m)');axes[1].set_ylabel('speed (m/s)');axes[2].set_ylabel('accepted camera count')
    for ax in axes:ax.set_xlabel('frame')
    axes[0].legend(fontsize=6);fig.suptitle(f'{scene}/{pid}; source/residual arm legend; canonical displacement alone is not tracking success')
    fig.tight_layout();fig.savefig(Path(out)/'figures'/f'{scene}_{pid}_arms.png',dpi=125);plt.close(fig)


def generate(out):
    out=Path(out);target=out/'SOURCE_RESPONSIBILITY_CAUSAL_REPORT.md'
    if target.exists():raise FileExistsError(f'Preserving report {target}')
    result=read_json(out/'causal_evidence.json');tests=read_json(out/'tests.json') if (out/'tests.json').exists() else {'successful':False}
    inventory=read_json(out/'evidence_inventory.json');lines=['# Source responsibility causal experiments','',
        f"Tests successful: {tests.get('successful')}. Investigation status: **{result['status']}**.",'',
        'All replays are CPU float64 conditional single-point counterfactuals with frozen competitors. No production inference, noise, configuration or existing output was changed. Source pixels are provenance, never fixed tracking coordinates. No material correspondence is certified from depth agreement.',
        '',f"Repository commit `{inventory['git_commit']}`. Full dirty diff, source/input hashes and actual baseline paths: `evidence_inventory.json`. Code and renderer comparison: `rig_euler_vs_rgbdiekf_correspondence.md`.",'',
        '| Scene / point | Baseline gate | Decision | Material identity |','|---|---|---|---|']
    for p in result['points']:lines.append(f"| {p['scene']}/{p['point_id']} | {p['parity']} | {p['decision']} | MATERIAL_ID_UNDECIDABLE |")
    for p in result['points']:
        lines+=['',f"## {p['scene']} / {p['point_id']}",'',f"**{p['decision']}**; baseline {p['parity']}. {p.get('reason','')}",
            '',f"Source and physical probe: {p.get('source')}",
            f"Actual frame-1 source-owner conflict: {p.get('source_conflict_at_frame1','Unavailable or baseline not qualified')}.",
            f"First accepted source/conflicting camera rows and force attribution: `first_update_responsibility.csv`, `source_owner_conflicts.csv` and `information_steps.npz` filtered by this scene/point.",
            f"Frame-1 same-prior factorial and independent continuous arms: `association_counterfactuals.csv` and `sequence_state.csv`. Arm outcomes: {p.get('arm_outcomes',[])}",
            '', 'Competing hypotheses:',
            '- Removing a cross-camera row isolates its influence, but cannot by itself distinguish wrong material correspondence from persistent appearance/depth discrepancy.',
            '- Reference cancellation at unchanged xref is algebraic; missing references are never filled with zero. R+R0 sensitivity omits shared cross-time noise.',
            '- Source-priority transfers a single ownership slot; its displaced competitor is named. It is not an independent valid coupled posterior for both points.',
            '- Fixed-support original costs, changed cells and nonlinear convergence are separate from modified-residual costs. Undefined support is never a zero residual.',
            f"Evidence: {p.get('orthogonal_tests',{})}"]
        image=f"{p['scene']}_{p['point_id']}_arms.png"
        if (out/'figures'/image).exists():lines+=['',f'![Independent factorial trajectories](figures/{image})']
    lines+=['','## Required case-specific questions','',
        'Storage 8444: inspect its camera-0 source winner and displaced PID in source_owner_conflicts.csv; compare B, D, combined E and SRC_PRIORITY at identical priors. Camera 1 alone is the archived frame-1 admitted cause; camera-0 suppression is a distinct ownership mechanism. A static result with no rows is not evidence of a corrected likelihood.',
        'USB 67827: frame-1 accepted cameras are 5 and 6. Camera 4 is a late-admission/missing-reference test, not a frame-1 cause. The camera-6 and late camera-4 raw samples, NIS and reference status remain explicit in sequence_rows.csv and responsibility_frame0.csv.',
        'USB 87290/85696: moving status and GT errors exist only in the disconnected post-hoc evaluation. Source-only loss of all observations is a tracking failure even with zero velocity. Image samples at predicted trajectories cannot establish that a different material location is the true one without independent correspondence.',
        '', '## Population and architecture','',
        'population_source_vs_cross.csv contains conditional observational strata and denominators, not causal rates. GT labels are applied after image/ownership features. All-frame point trajectories are never re-filtered. Saved float32 population velocities/variances limit exact accepted-state force claims outside traced IDs.',
        'PointMeasurement.__call__ first creates camera-point candidates from geometric visibility rather than material identity; build_terms then forms the absolute RGB-D measurement. This code fact does not prove physical association failure. Position/velocity cross-covariance transfers spatial residual responses into velocity; later support loss is a separate mechanism.',
        '', '## Remaining qualifications','']
    lines += ['- '+s for s in result.get('limitations',[])];target.write_text('\n'.join(lines)+'\n')
    return target
