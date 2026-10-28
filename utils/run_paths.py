"""Stage folders of a four-pass run. A stage reused with --resume-from is read in place from the source run (recorded in
run_manifest.json['reused_stages']); nothing is linked or copied, so this also works on filesystems without symlinks (exFAT)."""
import json
from pathlib import Path


def stage_dir(root, name):
    """Folder of `name` (e.g. 'free_forward', 'joint/backward') for run `root`: the source run's when that folder was
    reused (exact folder name, e.g. joint/<d> or joint/<d>/support), otherwise the run's own."""
    root = Path(root); manifest = root/'run_manifest.json'
    if manifest.exists():
        reused = json.loads(manifest.read_text()).get('reused_stages') or {}
        if reused.get('from') and name in reused.get('folders', []):
            return stage_dir(reused['from'], name)
    return root/name
