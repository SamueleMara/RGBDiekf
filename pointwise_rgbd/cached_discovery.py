"""Reuse frozen histories and the completed correspondence audit, without refits."""
import hashlib,json
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from .store import json_write


def prepare_frozen_history(cache,ownership,dataset,debug):
    cache.dataset=dataset # Used only for immutable source IDs of NEW tracks.
    debug.cached_history_source=str(cache.root)
    if not ownership.meta('initial_complete',False):
        cache.owner_db.backup(ownership.db)
        with ownership.db:ownership.set_meta('frozen_cache_source',str(cache.root))
    elif ownership.meta('frozen_cache_source')!=str(cache.root):
        raise ValueError('Cached discovery output belongs to a different frozen history')
    original=json.loads((cache.root/'initial_forward/summary.json').read_text())
    debug.frame_records=original['frame_records']
    debug.cached_initial_stage_seconds={key:sum(r.get(key,0.) for r in debug.frame_records) for key in ('frame_seconds','filter_seconds','knn_seconds','projection_seconds','index_build_seconds','association_seconds')}
    json_write(debug.root/'initial_forward/summary.json',{**original,'frozen_cache_source':str(cache.root),'initial_optimizer_invocations_this_run':0,'history_storage':str(cache.directory)})
    return cache


def import_coverage_audit(cache,ownership,dataset,directory,debug):
    from .incremental_coverage import cache_observations
    import time
    started=time.monotonic()
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text());summary=json.loads((directory/'summary.json').read_text())
    if summary['status']!='AUDIT_COMPLETE' or not summary['all_cached_frames'] or summary['optimizer_calls']!=0:raise ValueError('A complete, successful correspondence-only audit is required')
    if Path(manifest['cache']).resolve()!=cache.root or summary['cached_SOLVED']!=cache.count:raise ValueError('Coverage audit describes a different frozen SOLVED map')
    if summary['frames_checked']!=len(dataset.times) or summary['cameras_checked']!=len(dataset.cameras):raise ValueError('Coverage audit does not include every selected observation')
    model_path=directory/'audited_model.json'
    if not model_path.exists():raise ValueError('Run tools/report_surface_correspondence.py to finalize the audit/model provenance before discovery')
    model=json.loads(model_path.read_text())
    if hashlib.sha256((cache.root/'run_manifest.json').read_bytes()).hexdigest()!=model['source_run_manifest_sha256']:raise ValueError('Cached statistical model changed since correspondence audit')
    kernel=Path(__file__).with_name('coverage.py')
    if hashlib.sha256(kernel.read_bytes()).hexdigest()!=manifest['source_hashes']['pointwise_rgbd/coverage.py']:raise ValueError('Coverage kernel changed after audit; rerun correspondence audit, not optimization')
    version=ownership.meta('model_version',0)
    if ownership.coverage.completed(version):return
    row=cache.owner_db.execute("SELECT value FROM meta WHERE key='model_version'").fetchone()
    base_version=0 if row is None else json.loads(row[0])
    if version!=base_version:return # New appended points require fresh coverage.
    if ownership.meta('frozen_cache_source')!=str(cache.root):raise ValueError('Not a frozen-cache discovery output')
    fingerprints=json.loads((cache.root/'raw_observations/input_fingerprints.json').read_text())
    for k in tqdm(range(len(dataset.times)),desc='Verify frozen cache / import audited coverage (NO filtering)',unit='frame'):
        state,p,_,_,_=cache.frame(k);cache.verify_frame(k,state,p)
        for ci in range(len(dataset.cameras)):
            frame=dataset.observation(k,ci)
            if fingerprints[f'{k:06d}_{ci:03d}']!=dataset.base.observation_fingerprints[f'{k:06d}_{ci:03d}']:raise ValueError('Original audit RGB-D/mask content changed')
            with np.load(directory/'raw_classification'/f'{k:06d}_{ci:03d}.npz') as data:
                if not np.array_equal(data['pixel'],np.flatnonzero(frame.valid.ravel())):raise ValueError('Audited raw pixel identities changed')
                ownership.coverage.store.write(f'{k}:{ci}',model_version=np.array(version),pixel=data['pixel'],explained=data['appearance_explained'],geometry_explained=data['geometry_explained'],point_id=data['appearance_point_id'],geometry_point_id=data['geometry_point_id'])
            cache_observations(ownership.coverage,k,ci,frame,debug.config,ownership.coverage.read(k,ci,version))
    last=cache.owner_db.execute('SELECT MAX(id) FROM points').fetchone()[0]
    ownership.coverage.complete(version,-1 if last is None else last)
    json_write(debug.root/'discovery/imported_coverage_audit.json',{'source':str(directory),'seconds':time.monotonic()-started,'summary':summary,'no_measurement_ownership_or_old_state_changed':True,'diagnostics_reference':str(directory/'before_after_by_frame_camera.csv')})
