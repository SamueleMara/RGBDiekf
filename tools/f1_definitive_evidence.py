"""Evidence discovery, hashing and preregistered selection. Investigation only."""
from pathlib import Path
import csv,hashlib,json,subprocess
from datetime import datetime,timezone
import numpy as np

REPO=Path(__file__).resolve().parents[1]
DEFAULT_ROOT=Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030')
MANDATORY={'storage_45135':{8444:'static_primary',49199:'static',262894:'static',67801:'static',102354:'static',8416:'boundary',8417:'boundary',8418:'boundary',60019:'static_control',100564:'static_control'},
           'USB_100109':{67827:'static_primary',30502:'static',51975:'static_primary',51974:'static_control',30503:'static_control',87290:'moving_negative',85696:'moving_negative'},
           'blade_103706':{10251:'static_depth',14104:'static_depth'}}
PROTECTED=sorted([*REPO.glob('estimator/*.py'),*REPO.glob('scene/*.py'),*REPO.glob('configs/*'),REPO/'train.py'])
REQUIRED={
 'second_pass_20261030':['SECOND_PASS_ROOT_CAUSE.md','first_update_iterations.jsonl','first_update_decomposition.csv','frame0_frame1_raw_residuals.csv','iteration_support_objectives.csv','local_counterfactuals.csv'],
 'third_pass_temporal_reference_20261030':['README.md','THIRD_PASS_CAUSAL_RESULT.md','csv/frame0_references.csv','csv/per_camera_depth_crease.csv','csv/arm_outcomes.csv'],
 'fourth_pass_sequential_replay_20261030':['FOURTH_PASS_CAUSAL_REPORT.md','baseline_parity.csv','sequence_state.csv','sequence_camera_rows.csv','sequence_accepted_steps.csv','sequence_objective.csv','sequence_failure_events.csv','reference_covariance_tests.csv','ownership_mode_C_parity.csv','pixel_shift_sequence_8416.csv','diagnostic_units_corrections.csv','tests.log','sequence_arrays.npz']}


def json_value(x):
    if isinstance(x,np.ndarray):return json_value(x.tolist())
    if isinstance(x,np.generic):return json_value(x.item())
    if isinstance(x,dict):return {str(k):json_value(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_value(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(json_value(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read_json(path):return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def csv_write(path,rows,columns=()):
    rows=list(rows);keys=list(dict.fromkeys([*columns,*[k for r in rows for k in r]]));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(json_value(v),separators=(',',':')) if isinstance(v,(dict,list,tuple,np.ndarray)) else json_value(v) for k,v in r.items()})


def csv_read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def paths_for(root,scene):
    root=Path(root);plan=read_json(root/scene/'forensics/trace_plan.json')
    return {'baseline':Path(plan['baseline_arguments']['output']),'traced':root/scene/'traced','forensics':root/scene/'forensics'}


def file_record(path,root=None):
    path=Path(path)
    if not path.is_file():return {'path':str(path),'status':'UNAVAILABLE'}
    s=path.stat();r={'path':str(path.resolve()),'status':'AVAILABLE','size':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':digest(path)}
    if root:
        try:r['relative_path']=str(path.relative_to(root))
        except ValueError:pass
    if path.suffix=='.csv':
        with path.open() as f:r['columns']=next(csv.reader(f),[])
    elif path.suffix=='.npy':
        a=np.load(path,mmap_mode='r',allow_pickle=False);r.update(shape=list(a.shape),dtype=str(a.dtype))
    elif path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as z:r['arrays']={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
    return r


def make_inventory(root,out):
    root,out=Path(root),Path(out)
    if out.exists():raise FileExistsError(f'Preserving existing output: {out}. Use a fresh fifth_pass_* directory.')
    if not out.name.startswith('fifth_pass_'):raise ValueError('Analysis directory name must start with fifth_pass_')
    out.mkdir(parents=True)
    git=lambda *a:subprocess.check_output(['git','-C',str(REPO),*a],text=True)
    inventory={'created_utc':datetime.now(timezone.utc).isoformat(),'forensics_root':str(root.resolve()),'repo':str(REPO),'git_commit':git('rev-parse','HEAD').strip(),'git_branch':git('branch','--show-current').strip(),
               'git_status':git('status','--short'),'dirty_diff':git('diff','--binary'),'required':{},'scenes':{},'source_files':{},'protected_sources':{},'raw_files':{},
               'execution_contract':{'cpu_only':True,'no_full_scene_refilter':True,'no_inference_changes':True,'gt':'selection and evaluation only','analysis_budget':{'random_holdouts_per_scene':12,'stratified_per_scene':12},'parity_resolution':{'x_m':2e-12,'v_m_s':1e-10,'cov_atol':1e-12,'cov_rtol':1e-9,'covariance_metric':'Original entrywise allclose gate retained; per-block Frobenius error, reference norm and bound reported separately with xx m^2, xv/vx m^2/s, vv m^2/s^2'}}}
    # Save the manifest before any replay; checkpoints below keep partial preflight auditable.
    write_json(out/'evidence_inventory.json',inventory)
    for p in PROTECTED:
        if p.is_file():inventory['protected_sources'][str(p.relative_to(REPO))]=digest(p)
    for p in [*REPO.glob('tools/*f1*.py'),REPO/'tools/root_cause_oracle.py',REPO/'tools/root_cause_trace.py',REPO/'RGBDiekf_Definitive_F1_Static_Drift_Causal_Forensics_Codex.md',*REPO.glob('tests/*.py')]:
        inventory['source_files'][str(p.relative_to(REPO))]=file_record(p,REPO)
    for folder,names in REQUIRED.items():
        print('Inventory: hashing prior evidence',root/folder,flush=True)
        files=sorted(set([root/folder/n for n in names]+[p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.jsonl','.npz','.md','.log')]))
        for p in files:inventory['required'][str(p.relative_to(root))]=file_record(p,root)
    from tools.root_cause_trace import Run
    for scene in MANDATORY:
        try:
            print('Inventory: hashing baseline/trace and raw files for',scene,flush=True)
            paths=paths_for(root,scene);entry={'paths':paths,'files':{},'point_mapping':{}}
            for p in paths['forensics'].iterdir():
                if p.is_file() and p.suffix in ('.csv','.json','.npz'):entry['files'][str(p)]=file_record(p)
            for role in ('baseline','traced'):
                run=Run(paths[role],'forward')
                for p in [paths[role]/'run_manifest.json',*run.free.glob('*.npy'),*run.free.glob('*.npz'),run.free/'run_manifest.json',*run.free.glob('numeric/trace_frame_*.json')]:
                    if p.is_file():entry['files'][str(p)]=file_record(p)
                entry[role+'_manifest']=run.man;entry[role+'_F1_policy']=run.fman
                if role=='traced':
                    archived=run.man['cameras']['cameras'];actual=run.data.cameras
                    entry['current_vs_archived_input']={'frame_times_equal':bool(np.array_equal(run.data.times,run.fman['timestamps_seconds'])),
                        'camera_count_equal':len(actual)==len(archived),'camera_checks':[{'name':cam.name,'name_equal':cam.name==old['name'],
                            'K_equal':bool(np.array_equal(cam.intrinsics,old['intrinsics_fx_fy_cx_cy'])),'T_cw_max_abs_difference':float(abs(np.asarray(cam.world_to_camera)-np.asarray(old['T_cw'])).max()),
                            'dimensions_equal':cam.width==old['width'] and cam.height==old['height']} for cam,old in zip(actual,archived)],
                        'rig_scope':'Dataset exposes one calibrated rig, shared by all archived frames; physical calibration accuracy is not established by parity.'}
                    for pid in MANDATORY[scene]:
                        if pid>=len(run.can['point_id']):entry['point_mapping'][str(pid)]={'status':'POINT_UNAVAILABLE'};continue
                        entry['point_mapping'][str(pid)]={'catalogue_point_id':int(run.can['point_id'][pid]),'full_index':int(run.can['subsampled_to_full_index'][pid]),'source':run.can['original_source_ids'][pid],
                            'x_ref':run.can['canonical_xyz'][pid],'source_covariance_m2':run.can['canonical_source_covariance'][pid],'canonical_rgb':run.can['canonical_rgb'][pid]}
                    data=run.data;base=data.base
                    for k in range(min(11,len(data.times))):
                        for c in range(len(data.cameras)):
                            validpath=base.validity_paths[k][c] if base.validity_paths is not None else None
                            for name in (*base.entries[k][c],base.masks[k][c],validpath):
                                if name is None:continue
                                source=Path(run.man['arguments']['scene'])/name
                                inventory['raw_files'][str(source)]=file_record(source)
                    for folder in (Path(run.man['arguments']['scene'])/'metadata',):
                        for p in folder.glob('*'):
                            if p.is_file() and p.suffix in ('.yaml','.json','.csv','.txt'):inventory['raw_files'][str(p)]=file_record(p)
            entry['population_evidence']=file_record(paths['forensics']/'population_events.npz')
            entry['selection_seed']=int(digest(paths['baseline']/'run_manifest.json')[:16],16)
            entry['status']='AVAILABLE';inventory['scenes'][scene]=entry
        except (OSError,KeyError,ValueError) as e:inventory['scenes'][scene]={'status':'UNAVAILABLE','reason':str(e)}
        write_json(out/'evidence_inventory.json',inventory)
    inventory['preflight_complete']=True;write_json(out/'evidence_inventory.json',inventory)
    return inventory


def assert_protected(inventory):
    changed=[name for name,h in inventory['protected_sources'].items() if not (REPO/name).exists() or digest(REPO/name)!=h]
    if changed:raise RuntimeError('Inference/config files changed since preflight: '+', '.join(changed))


def verify_inputs(inventory):
    assert_protected(inventory)
    for section in ('required','raw_files','source_files'):
        for record in inventory[section].values():
            if record['status']=='AVAILABLE' and digest(record['path'])!=record['sha256']:raise RuntimeError('Input hash changed: '+record['path'])
    for scene in inventory['scenes'].values():
        for record in scene.get('files',{}).values():
            if digest(record['path'])!=record['sha256']:raise RuntimeError('Baseline/trace hash changed: '+record['path'])


def select_population(root,scene,inventory,out):
    """Stream saved baseline arrays; select random IDs before reading drift."""
    from tools.root_cause_trace import Run
    paths=paths_for(root,scene);run=Run(paths['baseline'],'forward');N=len(run.can['canonical_xyz'])
    with np.load(paths['forensics']/'population_events.npz') as z:
        static=z['gt_static'].astype(bool);moving=z['gt_moving'].astype(bool);dstatic=z['d_static'];dmoving=z['d_moving']
        if len(static)!=N or not np.array_equal(z['x0'],run.can['canonical_xyz']):raise ValueError('GT evaluation catalogue mapping differs from baseline')
    seed=inventory['scenes'][scene]['selection_seed'];g=np.random.default_rng(seed);ids=np.flatnonzero(static)
    holdout=np.sort(g.choice(ids,min(12,len(ids)),replace=False))
    write_json(out/f'{scene}_preregistered_holdout.json',{'seed':seed,'seed_source':'SHA256 baseline/run_manifest.json first 64 bits','ids':holdout,'selection_before_residual_inspection':True,'grounding':'nearest labelled cloud, not exact material identity'})
    X,V,st,rows,reasons=[run.mm(n) for n in ('posterior_xyz','velocity','point_status','valid_cameras','camera_reason')];src=run.can['original_source_ids'];order=run.fman['processing_order'];xref=run.can['canonical_xyz']
    drift=np.linalg.norm(np.asarray(X[order[-1]])-xref,axis=1);first=np.full(N,-1);initial_count=np.zeros(N,int);non_source=np.zeros(N,bool);firstloss=np.full(N,-1);firstdrift=np.full(N,-1)
    for k in order[1:]:
        measured=np.asarray(st[k])==1;fresh=(first<0)&measured;first[fresh]=k;initial_count[fresh]=np.asarray(rows[k])[fresh]
        admitted=np.asarray(reasons[k])==0
        admitted[src[:,1],np.arange(N)]=False;non_source|=admitted.any(0)&measured
        loss=(first>=0)&(np.asarray(rows[k])==0)&(firstloss<0);firstloss[loss]=k
        # Reporting band only; not used by any filter or causal onset detection.
        dep=(np.linalg.norm(np.asarray(X[k])-xref,axis=1)>1e-3)&(firstdrift<0);firstdrift[dep]=k
    bands=np.digitize(drift,[1e-4,1e-3,1e-2,1e-1]);strata={}
    for pid in ids:
        key=(int(first[pid]),int(initial_count[pid]),int(src[pid,1]),int(bands[pid]));strata.setdefault(key,[]).append(int(pid))
    # Round-robin over drift bands before finer strata; do not select only failures.
    keys=sorted(strata,key=lambda k:(k[3],k[2],k[0],k[1]));byband={}
    for key in keys:byband.setdefault(key[3],[]).append(key)
    interleaved=[]
    for j in range(max(map(len,byband.values()),default=0)):
        for band in sorted(byband):
            if j<len(byband[band]):interleaved.append(byband[band][j])
    chosen=[]
    for key in interleaved:
        candidates=np.setdiff1d(strata[key],holdout)
        if len(candidates):chosen.append(int(g.choice(candidates)))
        if len(chosen)==12:break
    stable=ids[non_source[ids]&(bands[ids]<=1)];stable=stable[np.lexsort((stable,src[stable,1],drift[stable]))]
    stable_denominator=len(stable)
    stable=np.setdiff1d(stable,holdout)[:min(4,len(stable))]
    chosen=list(dict.fromkeys([*map(int,stable),*chosen]))[:12]
    catalogue=[]
    selected=set(map(int,holdout))|set(chosen)|set(map(int,stable))|set(MANDATORY[scene])
    for pid in sorted(selected):
        if pid>=N:continue
        catalogue.append({'scene':scene,'point_id':pid,'role':MANDATORY[scene].get(pid,'population_control'),'fixed':pid in MANDATORY[scene],'random_holdout':pid in holdout,'stratified':pid in chosen,'stable_measured_control':pid in stable,
                          'source_frame_camera_pixel':src[pid],'full_index':int(run.can['subsampled_to_full_index'][pid]),'GT_static_evaluation_only':bool(static[pid]),'GT_moving_evaluation_only':bool(moving[pid]),'GT_distance_static_m':dstatic[pid],'GT_distance_moving_m':dmoving[pid],
                          'GT_identity_status':'COARSE_NEAREST_LABELLED_SAMPLE_NOT_EXACT_MATERIAL_ID','first_accepted_frame':int(first[pid]),'initial_admitted_cameras':int(initial_count[pid]),'final_drift_m':drift[pid],'drift_band':int(bands[pid]),'has_non_source_measurement':bool(non_source[pid])})
    summary={'scene':scene,'total_points':N,'GT_grounded_static':len(ids),'GT_attribution_incomplete':int((~static&~moving).sum()),'unmeasured_static':int((first[ids]<0).sum()),'stable_measured_non_source_population':stable_denominator,'stable_measured_non_source_selected':len(stable),
             'source_camera_static_counts':np.bincount(src[ids,1]).tolist(),'strata':[{'key':key,'denominator':len(v)} for key,v in strata.items()],
             'drift_before_row_loss_count':int(((firstdrift[ids]>=0)&((firstloss[ids]<0)|(firstdrift[ids]<firstloss[ids]))).sum()),'drift_event_count':int((firstdrift[ids]>=0).sum()),
             'reporting_drift_m':1e-3,'interpretation':'descriptive clustered samples; not IID causal prevalence','holdout_ids':holdout,'stratified_ids':chosen,'stable_control_ids':stable}
    return catalogue,summary
