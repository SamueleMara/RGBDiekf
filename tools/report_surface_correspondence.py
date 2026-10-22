#!/usr/bin/env python3
"""Finalize measured audit, selected-pair residuals and exact attribution sets."""
import argparse,hashlib,json,sys,tempfile,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.frozen import FrozenCache
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.observations import samples
from pointwise_rgbd.diagnostics import distribution
from pointwise_rgbd.store import json_write


def tracking_pair_report(root,cache,cfg,data):
    """Re-evaluate saved exclusive-owner pairs on the SAME frozen XYZ.

    These are posterior-cache residuals, not original pre-update innovations.
    No association is rerun and no point is fitted.
    """
    import time
    started=time.monotonic();before=cache.frozen_signature();rows=[];count=0
    names=('distance_m','rgb_cosine','color_d2_original_2R')
    precision=np.linalg.inv(2*cfg.color_covariance)
    with tempfile.TemporaryDirectory(prefix='tracking_pair_metrics_',dir=root) as temporary:
        scratch=Path(temporary);streams={name:(scratch/name).open('wb') for name in names}
        for k in tqdm(range(len(data.times)),desc='Report actual saved exclusive tracking pairs (NO fitting)',unit='frame'):
            state,p,rgb,ids,_=cache.frame(k);cache.verify_frame(k,state,p);order=np.argsort(ids)
            for ci in range(len(data.cameras)):
                frame=data.observation(k,ci);raw=samples(frame,cfg)
                with np.load(root/'raw_classification'/f'{k:06d}_{ci:03d}.npz') as base:
                    if not np.array_equal(raw.pixel,base['pixel']):raise ValueError('Raw observations changed')
                    use=(base['baseline_owner']>=0)&np.isin(base['baseline_kind'],[1,3])
                    owner=base['baseline_owner'][use]
                selected=order[np.searchsorted(ids[order],owner)]
                if not np.array_equal(ids[selected],owner):raise ValueError('Unknown saved owner')
                distance=np.linalg.norm(raw.xyz[use]-state[selected,:3],axis=1)
                a=raw.rgb[use];b=rgb[selected];product=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
                cosine=np.full(len(a),np.nan);np.divide((a*b).sum(1),product,out=cosine,where=product>0);cosine=np.clip(cosine,-1,1)
                delta=a-b;color=np.einsum('ni,ij,nj->n',delta,precision,delta)
                values=dict(zip(names,(distance,cosine,color)))
                rows.append({'frame':k,'camera':ci,'actual_exclusive_owned_samples':len(owner),**{name:distribution(value) for name,value in values.items()}})
                for name,value in values.items():streams[name].write(np.asarray(value,np.float64).tobytes())
                count+=len(owner)
        for stream in streams.values():stream.close()
        global_metrics={name:distribution(np.memmap(scratch/name,mode='r',dtype=np.float64,shape=(count,))) for name in names}
    cache.assert_old_frozen(before)
    report={'actual_exclusive_owned_samples':count,'metrics':global_metrics,'seconds':time.monotonic()-started,'optimizer_calls':0,'qualification':'Actual immutable measurement/anchor owners only (kinds 1/3), excluding old kind-2 coverage bookkeeping. Residuals evaluated at cached posterior XYZ, not original pre-update innovations; RGB distance uses the original 2*R_rgb ranking convention, which had no hard color rejection.'}
    json_write(root/'exclusive_tracking_pair_metrics.json',report)
    with (root/'exclusive_tracking_pair_metrics_by_view.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--tests',type=Path);parser.add_argument('--tracking-pairs-only',action='store_true',help='Add actual saved exclusive-owner pair metrics without recomputing the already completed residual report');args=parser.parse_args()
    root=args.audit;summary=json.loads((root/'summary.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    if summary['status']!='AUDIT_COMPLETE':raise ValueError('Correspondence audit is incomplete')
    cache=FrozenCache(manifest['cache']);before=cache.frozen_signature();cfg=cache.config();data=Dataset(cache.manifest['dataset']['input'],**cache.manifest['dataset']['test_limits'])
    if args.tracking_pairs_only:
        tracking_pair_report(root,cache,cfg,data);return
    source_manifest=cache.root/'run_manifest.json'
    if source_manifest.stat().st_mtime_ns>(root/'manifest.json').stat().st_mtime_ns:raise ValueError('Cached statistical model was edited after this audit began; no reuse')
    fingerprints=json.loads((cache.root/'raw_observations/input_fingerprints.json').read_text())
    counts={'ownership_witness_and_new_appearance_recovers':0,'ownership_witness_but_new_appearance_rejects':0,'new_appearance_recovers_without_old_family_witness':0,'unmatched_by_both_without_ownership_witness':0,'frozen_measurement_owned_geometry_unexplained':0}
    nearest_first=None;metric_names=('geometry_d2','appearance_geometry_d2','appearance_color_d2','appearance_angular_d2','appearance_brightness_d2');sizes={name:0 for name in metric_names}
    residual_path=root/'selected_pair_residuals.jsonl'
    residual_path.write_text('')
    with tempfile.TemporaryDirectory(prefix='residual_statistics_',dir=root) as temporary:
        directory=Path(temporary);streams={name:(directory/name).open('wb') for name in metric_names}
        whiten=np.linalg.inv(np.linalg.cholesky(4*cfg.color_covariance))
        for k in tqdm(manifest['frames'],desc='Verify stored correspondence residuals (NO fitting)',unit='frame'):
            state,p,rgb,ids,_=cache.frame(k);cache.verify_frame(k,state,p)
            for ci in range(len(data.cameras)):
                frame=data.observation(k,ci)
                if fingerprints[f'{k:06d}_{ci:03d}']!=data.base.observation_fingerprints[f'{k:06d}_{ci:03d}']:raise ValueError('Audit inputs changed')
                raw=samples(frame,cfg)
                with np.load(root/'raw_classification'/f'{k:06d}_{ci:03d}.npz') as record:
                    old=record['baseline_owner']>=0;w=record['assigned_point_witness'];g=record['geometry_explained'];a=record['appearance_explained'];owned=(record['baseline_owner']>=0)&np.isin(record['baseline_kind'],[1,3])
                    counts['ownership_witness_and_new_appearance_recovers']+=int((~old&w&a).sum());counts['ownership_witness_but_new_appearance_rejects']+=int((~old&w&~a).sum())
                    counts['new_appearance_recovers_without_old_family_witness']+=int((~old&~w&a).sum());counts['unmatched_by_both_without_ownership_witness']+=int((~old&~w&~a).sum())
                    counts['frozen_measurement_owned_geometry_unexplained']+=int((owned&~g).sum())
                    first=np.flatnonzero(~a)
                    if len(first):
                        j=int(first[0]);key=(-k,ci,int(raw.pixel[j]))
                        if nearest_first is None or key<nearest_first['sort_key']:
                            nearest_first={'sort_key':key,'frame':k,'original_frame':data.frame_ids[k],'camera':ci,'camera_name':frame.camera.name,'pixel':int(raw.pixel[j]),'frozen_measurement_owner':int(record['baseline_owner'][j]) if owned[j] else None}
                    values={}
                    for prefix,mask,name in [('geometry',g,'geometry_point_id'),('appearance',a,'appearance_point_id')]:
                        selected=record[name][mask];rows=np.searchsorted(ids,selected)
                        if not np.array_equal(ids[rows],selected):raise AssertionError('Unknown SOLVED point identity in audit')
                        error=raw.xyz[mask]-state[rows,:3];bound=2*(p[rows,:3,:3]+raw.covariance[mask])
                        d2=np.einsum('ni,ni->n',error,np.linalg.solve(bound,error[...,None])[...,0])
                        if len(d2) and d2.max()>cfg.gate(3)+1e-8:raise AssertionError('Stored selected geometry fails its recorded gate')
                        values['geometry_d2' if prefix=='geometry' else 'appearance_geometry_d2']=d2
                        if prefix=='appearance':
                            u=raw.rgb[mask]@whiten.T;v=rgb[rows]@whiten.T;nu=np.linalg.norm(u,axis=1);nv=np.linalg.norm(v,axis=1)
                            color=((u-v)**2).sum(1);angular=2*np.maximum(nu*nv-(u*v).sum(1),0);brightness=(nu-nv)**2
                            if len(color) and color.max()>cfg.gate(3)+1e-8:raise AssertionError('Stored selected color fails its recorded gate')
                            np.testing.assert_allclose(color,angular+brightness,rtol=1e-11,atol=1e-10)
                            values.update(appearance_color_d2=color,appearance_angular_d2=angular,appearance_brightness_d2=brightness)
                    for name,value in values.items():streams[name].write(np.asarray(value,np.float64).tobytes());sizes[name]+=len(value)
                    with residual_path.open('a') as stream:stream.write(json.dumps({'frame':k,'camera':ci,**{name:distribution(value) for name,value in values.items()}})+'\n')
        for stream in streams.values():stream.close()
        residuals={name:distribution(np.memmap(directory/name,mode='r',dtype=np.float64,shape=(size,))) if size else distribution([]) for name,size in sizes.items()}
        json_write(root/'selected_pair_global_distributions.json',residuals)
    cache.assert_old_frozen(before)
    totals=summary['totals'];old_missing=totals['current_unmatched']
    ownership_fraction=f"{totals['current_unmatched_legacy_same_ties']/old_missing:.3%}" if old_missing else 'not applicable (no original misses)'
    if sum(counts[name] for name in counts if name!='frozen_measurement_owned_geometry_unexplained')!=old_missing:raise AssertionError('Attribution partition does not reproduce original unmatched set')
    json_write(root/'attribution.json',{'original_unmatched':old_missing,'controlled_ownership_only_reclassified':totals['current_unmatched_legacy_same_ties'],'strict_already_assigned_point_witness':totals['strict_owned_point_witness'],'partition_of_original_unmatched':{key:value for key,value in counts.items() if key!='frozen_measurement_owned_geometry_unexplained'},'frozen_measurement_owned_geometry_unexplained':counts['frozen_measurement_owned_geometry_unexplained'],'first_reverse_unexplained':nearest_first,'qualification':'Only the controlled old-family/gate ablation isolates ownership restrictions. Additional new matches also change query direction, world-space covariance handling and visibility/color gates; they are not all attributed to ownership.'})
    model={'source_run_manifest_sha256':hashlib.sha256(source_manifest.read_bytes()).hexdigest(),'config':cache.manifest['config'],'calibration_sha256':cache.manifest['calibration_sha256'],'source_manifest_stable_since_audit_started':True}
    json_write(root/'audited_model.json',model)
    tests=None if args.tests is None else json.loads(args.tests.read_text())
    dists=json.loads((root/'global_distributions.json').read_text())
    lines=['# Real-data correspondence audit results','',f"Frozen cache: `{cache.root}`.",f"Checked {summary['frames_checked']} frames × {summary['cameras_checked']} cameras; {totals['raw_valid']:,} valid foreground observations; {summary['cached_SOLVED']:,} cached SOLVED points.",'','No initial forward optimization, backward point fit, old-point update or trajectory interpolation was performed by the audit. All original frame digests and appended point hashes were verified.','', '| Coverage policy | Explained | Unexplained | Explained fraction |','|---|---:|---:|---:|']
    for prefix,label in [('current','Original exclusive-owner result'),('geometry','Observed → SOLVED geometric kNN'),('appearance','Geometric kNN + RGB angular/brightness')]:
        explained=totals[prefix+'_explained'];lines.append(f"| {label} | {explained:,} | {totals['raw_valid']-explained:,} | {explained/totals['raw_valid']:.6%} |")
    lines+=['','## Attributing the original unmatched observations','',f"The controlled same-family, same-gate, same-tie ablation reclassifies **{totals['current_unmatched_legacy_same_ties']:,} / {old_missing:,} ({ownership_fraction})** original unmatched pixels when exclusive ownership restrictions are removed. {totals['strict_owned_point_witness']:,} have a compatible old-family edge from an already-assigned point.",'',f"The new geometric matcher covers {totals['current_unmatched_geometry_explained']:,} of the old unmatched set; the appearance variant covers {totals['current_unmatched_appearance_explained']:,}. These broader changes also modify search direction, covariance and visibility/color handling, so they are not all an ownership-only effect.",'',f"The new appearance rule rejects {totals['current_explained_appearance_rejected']:,} samples that the old owner table called explained. New unmatched = old unmatched − newly covered old misses + newly rejected old matches. RGB compatibility rejects {totals['geometry_supported_color_rejected']:,} geometrically supported observations under the declared static-color model.",'','## Frozen-measurement consistency','',f"{totals['unexplained_with_frozen_measurement_owner']:,} appearance-unexplained samples already belong to frozen measurement histories; {counts['frozen_measurement_owned_geometry_unexplained']:,} also fail geometry-only coverage. They cannot be silently reinitialized as independent points. The controller retains these observations as unexplained consistency conflicts and excludes them from new independent assimilation; it does not refit old tracks or relabel the samples as explained.",'',f"First reverse unexplained observation: `{nearest_first}`.",'','## Distances, appearance and the 1% tail','', 'The 1% chi-square tail is unchanged: 6.634897 (1D), 9.210340 (2D), 11.344867 (3D). These are uncalibrated, dependent component tests; no joint 99% confidence is claimed. Geometry/color covariance bounds handle unknown source/target error correlation. The score is the sum of marginal Gaussian negative log likelihoods, with no arbitrary geometry/appearance weight. RGB angular and brightness terms decompose ONE whitened RGB distance. Absolute XYZ cosine is never used.','']
    for name,dist in dists.items():lines.append(f"- `{name}`: finite count {dist['count']:,}; quantiles `{dist['quantiles']}`.")
    lines+=['','Exact selected-pair Mahalanobis, angular and brightness distributions are in `selected_pair_global_distributions.json`; per-frame/camera distributions are in `selected_pair_residuals.jsonl`. Global/per-frame distance and cosine histograms are in `global_distributions.json` and `distributions/`. All quantities come from actual stored pairs. Accepted-pair residuals are selection-conditioned, not independent NIS consistency samples.','',f"Candidate edges: {totals['candidate_edges']:,}; visible {totals['visible_edges']:,}; 3D gate {totals['geometry_edges']:,}; ray gate {totals['ray_edges']:,}; depth gate {totals['depth_edges']:,}; appearance gate {totals['appearance_edges']:,}. These are successive passing edge counts, not disjoint observation counts.",'',f"Permitted many-to-one extra samples: geometry {totals['geometry_many_to_one_extra_samples']:,}; appearance {totals['appearance_many_to_one_extra_samples']:,}. They are surface multiplicity, not duplicate measurement assimilation.",'','## Frame zero and visuals','', 'Every frame-zero valid foreground observation is required to be explained by both new variants; the audit asserts this for each camera. Actual before/after RGB overlays, world-coordinate HTML views and static previews are under `visuals/`. `coverage_comparison.html` adds matching-policy, camera and representative-time controls. Display sampling is labeled and never affects inference.','',f"Audit runtime: {summary['runtime_seconds']:.3f} s. Peak RSS: {summary['peak_rss_bytes']:,} bytes. GPU: `{summary['gpu']}`.",'']
    if tests:lines +=[f"Tests: {tests['tests_run']} run, {tests['failures']} failures, {tests['errors']} errors, {tests['skips']} skips; CUDA available={tests['cuda_available']}.",'']
    lines +=['The eight-neighbor candidate family is finite; unexplained under these gates is not proof of genuinely absent physical geometry. Frozen trajectories may drift, calibration is conditional/unvalidated, and illumination may violate the approved RGB model. Matching correction does not repair those states or retune their noise.','', 'Detailed equations and source responsibilities: `AUDIT_CORRESPONDENCE_2026-10-22.md`.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(line) for line in (root/'by_frame_camera.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(2,1,figsize=(11,8))
    for prefix,label in [('current','Original owner-based'),('geometry','Geometric kNN'),('appearance','Geometry + appearance')]:
        ys=[]
        for k in manifest['frames']:
            selected=[r for r in rows if r['frame']==k];ys.append(sum(r[prefix+'_explained'] for r in selected)/sum(r['raw_valid'] for r in selected))
        axes[0].plot(manifest['frames'],ys,label=label)
    axes[0].set(ylabel='Explained fraction (all cameras)',ylim=(0,1));axes[0].legend()
    axes[1].bar(['Ownership-only recovery','Extra appearance recovery','Ownership witness / appearance rejects'],[counts['ownership_witness_and_new_appearance_recovers'],counts['new_appearance_recovers_without_old_family_witness'],counts['ownership_witness_but_new_appearance_rejects']])
    axes[1].set(ylabel='Original unmatched raw observations');fig.tight_layout();fig.savefig(root/'numerical_comparison.png');plt.close(fig)
    print(json.dumps({'ownership_only_reclassified':totals['current_unmatched_legacy_same_ties'],'old_unmatched':old_missing,'new_geometry_unmatched':totals['raw_valid']-totals['geometry_explained'],'new_appearance_unmatched':totals['raw_valid']-totals['appearance_explained'],'owned_but_unexplained':totals['unexplained_with_frozen_measurement_owner'],'first_reverse_unexplained':nearest_first},indent=2))


if __name__=='__main__':main()
