"""Disconnected post-hoc evaluation. Never imported by responsibility policies."""
from pathlib import Path
import numpy as np
from tools.f1_definitive_evidence import csv_write


def moving_evaluation(root,records,out):
    from tools.f1_second_pass import Scene
    from tools.forensic_f1_drift import GroundTruth
    rows=[];scenes={};truth={}
    for r in records:
        if r['point_id'] not in (87290,85696,8416):continue
        name=r['scene']
        if name not in scenes:
            sc=Scene(root,name);scenes[name]=sc;truth[name]=GroundTruth(sc.run,Path(sc.run.man['arguments']['scene']))
        sc=scenes[name];x=truth[name].moved(sc.x0[r['point_id']],r['frame'])
        rows.append({'scene':name,'point_id':r['point_id'],'frame':r['frame'],'variant':r['variant'],'GT_position_m_EVALUATION_ONLY':x,
            'error_m_EVALUATION_ONLY':float(np.linalg.norm(np.asarray(r['x_plus_m'])-x)),'accepted_cameras':r['rows_accepted'],
            'inference_uses_GT':False,'material_identity':'coarse nearest labelled cloud, not exact material correspondence',
            'assessment':'NO_TRACKING_SUCCESS_WITHOUT_ROWS' if not r['rows_accepted'] else 'ERROR_REQUIRES_COMPARISON_TO_REAL_MOTION'})
    csv_write(Path(out)/'moving_GT_evaluation_only.csv',rows,('scene','point_id','frame'))
    return rows
