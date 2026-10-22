#!/usr/bin/env python3
"""Archive the recorded estimator version, verifying its original run hashes."""
import argparse,hashlib,json,shutil
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]


def original_diagnostic_schema(name,text):
    # Only reverse the final reporting additions. Each result must equal the
    # SHA256 recorded BEFORE the experiment; otherwise archival fails.
    if name=='pointwise_rgbd/association.py':
        text=''.join(line for line in text.splitlines(keepends=True) if 'stats.update(raw_pixels=' not in line)
    elif name=='pointwise_rgbd/discovery.py':
        text=text.replace("'matching_stats':[],'filter_reports':[]", "'matching_stats':[]")
        text=''.join(line for line in text.splitlines(keepends=True) if "track['filter_reports'].append" not in line)
    elif name=='pointwise_rgbd/store.py':text=text.replace("'matching_stats','filter_reports','backward_frames'","'matching_stats','backward_frames'")
    elif name=='pointwise_rgbd/diagnostics.py':
        text=''.join(line for line in text.splitlines(keepends=True) if "record['invalid_foreground_pixels']=" not in line and "json_write(directory/'optimizer_by_frame.json'" not in line)
        for key in ('raw_pixels','foreground_pixels','invalid_foreground_pixels'):
            text=text.replace("'"+key+"':stats['"+key+"'],",'')
    return text


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserving existing source archive')
    manifest=json.loads((args.run/'run_manifest.json').read_text());blobs={};restored=[]
    for name,expected in manifest['source_hashes'].items():
        blob=(REPO/name).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=expected:
            blob=original_diagnostic_schema(name,blob.decode()).encode();restored.append(name)
        if hashlib.sha256(blob).hexdigest()!=expected:raise ValueError(f'Cannot recover byte-exact recorded source: {name}')
        blobs[name]=blob
    pending=args.output.with_name(args.output.name+'.pending');pending.mkdir(parents=True)
    for name,blob in blobs.items():
        path=pending/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    for source in (REPO/'rgbd_common').rglob('*.py'):
        target=pending/source.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(REPO/'configs/example_uncalibrated.json',pending/'configs/example_uncalibrated.json')
    report={'run':str(args.run),'all_original_recorded_source_hashes_verified':True,'files_reconstructed_to_original_hash_after_logging_only_additions':restored,'dependency_qualification':'Shared rgbd_common modules and example noise configuration copied from the audited workspace; these were not changed during implementation. Their archive hashes are included separately.','dependency_hashes':{str(p.relative_to(pending)):hashlib.sha256(p.read_bytes()).hexdigest() for p in pending.rglob('*') if p.is_file()}}
    (pending/'ARCHIVE_PROVENANCE.json').write_text(json.dumps(report,indent=2)+'\n');pending.rename(args.output)
    print(json.dumps({k:v for k,v in report.items() if k!='dependency_hashes'},indent=2))


if __name__=='__main__':main()
