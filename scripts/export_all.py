"""
Harvest all experiment artifacts from both cluster result trees into one
git-friendly folder.

Run on the Neptun cluster:

    cd ~/VIT/SIM2REAL_CLUSTER
    python3 scripts/export_all.py

Writes everything into ./cluster_export/{sim2real,cluster}/<run>/.

Why the conversions: the FrequencyAnalysis .gitignore excludes *.pt, logs/
and mask_epoch_*.png, so anything left in those formats would be silently
skipped by git. Masks are therefore saved as .npy, training histories as
.json, and log metrics as .txt. Nothing is emitted as .pt.

Checkpoints (checkpoint.pt ~141 MB, best_model.pt ~47 MB) are never copied.
The mask tensor is extracted out of each one and saved as a small .npy, so
the Phase 3 and lambda=0.05 masks survive without moving the checkpoints.

Masks are stored RAW (pre-activation). Apply the activation for the phase
when analysing:
    Phase 1 / Phase 3   normalize   -> w / w.mean()
    Phase 2             sigmoid     -> 2 * sigmoid(w)
"""

import torch
import numpy as np
import os
import shutil
import glob
import json
import re

HOME = os.path.expanduser('~')

SRC = {
    'sim2real': HOME + '/VIT/SIM2REAL',
    'cluster': HOME + '/VIT/SIM2REAL_CLUSTER',
}

OUT = HOME + '/VIT/SIM2REAL_CLUSTER/cluster_export'

MAX_COPY_BYTES = 8_000_000
METRIC_PREFIXES = ('Train:', 'Val:', 'Mask:', 'Reg:', 'Gap:')
EPOCH_RE = re.compile(r'^Epoch \d+/\d+ \(')


def find_mask(obj):
    """Locate a mask_weights tensor anywhere in a loaded checkpoint."""
    if torch.is_tensor(obj):
        return obj
    if not isinstance(obj, dict):
        return None
    if 'mask_weights' in obj:
        return obj['mask_weights']
    for key, val in obj.items():
        if torch.is_tensor(val) and 'mask' in str(key):
            return val
        if isinstance(val, dict):
            for subkey, subval in val.items():
                if 'mask_weights' in str(subkey):
                    return subval
    return None


def jsonable(obj):
    """Convert a training-history object into something json can write."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def export_run(run_dir, dest):
    """Export one result folder. Returns (masks, files) counts."""
    masks = 0
    files = 0
    os.makedirs(dest, exist_ok=True)

    # Mask evolution frames are numerous and near-duplicate; keep a sample.
    frames = sorted(glob.glob(os.path.join(run_dir, 'mask_epoch_*.png')))
    keep_frames = set(frames[::10] + frames[-1:])

    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue

        if name.endswith('.pt'):
            try:
                obj = torch.load(path, map_location='cpu',
                                 weights_only=False)
            except Exception as exc:
                print('  could not load %s: %r' % (path, exc))
                continue
            weights = find_mask(obj)
            if weights is not None:
                arr = weights.float().squeeze().numpy()
                np.save(os.path.join(dest, 'RAWMASK_' + name[:-3]), arr)
                masks += 1
            elif 'histor' in name:
                target = os.path.join(dest, name[:-3] + '.json')
                with open(target, 'w') as handle:
                    json.dump(jsonable(obj), handle)
                files += 1
            continue

        if name.startswith('mask_epoch_') and path not in keep_frames:
            continue

        if os.path.getsize(path) < MAX_COPY_BYTES:
            shutil.copy2(path, dest)
            files += 1

    return masks, files


def export_logs(log_dir, dest):
    """Strip tqdm spam out of the logs, keeping only metric lines."""
    if not os.path.isdir(log_dir):
        return 0
    os.makedirs(dest, exist_ok=True)
    written = 0
    for path in sorted(glob.glob(os.path.join(log_dir, '*.log'))):
        target = os.path.join(dest, os.path.basename(path) + '.metrics.txt')
        with open(path, errors='ignore') as src, open(target, 'w') as out:
            for line in src:
                stripped = line.strip()
                keep = stripped.startswith(METRIC_PREFIXES)
                if not keep and 'New best' in stripped:
                    keep = True
                if not keep and EPOCH_RE.match(stripped):
                    keep = True
                if keep:
                    out.write(line)
        written += 1
    return written


def main():
    total_masks = 0
    total_files = 0
    total_logs = 0

    for label, base in SRC.items():
        root = os.path.join(base, 'experiments', 'results')
        if not os.path.isdir(root):
            print('MISSING %s' % root)
            continue
        print('== %s (%s)' % (label, root))
        for run in sorted(os.listdir(root)):
            run_dir = os.path.join(root, run)
            if not os.path.isdir(run_dir) or run.startswith('.'):
                continue
            dest = os.path.join(OUT, label, run)
            masks, files = export_run(run_dir, dest)
            total_masks += masks
            total_files += files
            print('   %-42s masks=%d files=%d' % (run, masks, files))
        total_logs += export_logs(os.path.join(base, 'logs'),
                                  os.path.join(OUT, label, '_logs'))

    print('')
    print('masks extracted: %d' % total_masks)
    print('files copied:    %d' % total_files)
    print('logs distilled:  %d' % total_logs)
    print('output:          %s' % OUT)


if __name__ == '__main__':
    main()
