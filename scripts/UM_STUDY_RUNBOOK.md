# U/M study runbook

Implements `Orchestration/PLAN_FROM_HERE_2026-09-22.md` work items B and C, approved by
both reviewers on 23 September 2026. Everything lives in `scripts/um_study.py`.

**The two counted training runs are held** until the implementation has been reviewed.
Steps 1–4 below are all safe to run: they read data and checkpoints, and write only into
`~/VIT/um_study/`.

## Setup on Neptun

```bash
cd ~/VIT/SIM2REAL_CLUSTER && git pull
export TMPDIR=/mnt/local_learning/data/$USER/tmp && mkdir -p $TMPDIR
```

Paths are resolved from `$HOME` inside the script; nothing needs editing. It expects:

| What | Path |
|---|---|
| training pool | `~/VIT/SIM2REAL/data/imagenet_100k_cache` |
| evaluation set | `~/VIT/SIM2REAL/data/imagenet_25k_cache` |
| v5 checkpoint | `~/VIT/SIM2REAL_CLUSTER/experiments/results/resnet18_phase2/best_model.pt` |
| Phase 3.2 checkpoint | `~/VIT/SIM2REAL/experiments/results/resnet18_phase3.2/best_model.pt` |
| Phase 1 mask | `~/VIT/SIM2REAL/experiments/results/resnet18/learned_mask.pt` |
| outputs | `~/VIT/um_study/` |

## Step 1 — preflight, checks C1–C8

```bash
python3 scripts/um_study.py preflight --workers 8
```

Runs on one GPU in a couple of minutes, except C7, which hashes the encoded bytes of all
125,000 images to prove the partitions are disjoint. That reads ~9.5 GB and takes a while;
`--skip-hash` skips it for a quick pass, but **C7 must pass before any training**.

All checks must print PASS. Results land in `~/VIT/um_study/preflight/preflight.json`, and
the split manifest with cache fingerprints in `.../preflight/manifest.json`.

## Step 2 — verify the historical states

```bash
python3 scripts/um_study.py verify-historical \
    --export-dir ~/VIT/SIM2REAL_CLUSTER/cluster_export --workers 8
```

Loads both recovered checkpoints with `strict=True`, confirms BatchNorm buffers are
present, runs a finite forward pass, prints effective-mask statistics, and cross-references
the recovered mask against the exported array when one matches.

It then **reproduces Phase 3.2's recorded 75.385%** by evaluating that checkpoint on its own
20k development split — the seed-42 split of the 100k pool, which still exists. This is the
real end-to-end validation that the reconstruction is correct; matching key counts and a
stored scalar are not. v5's recorded score cannot be reproduced, because it was measured on
full-ImageNet validation, which was deleted on 2026-07-20.

## Step 3 — work item B, historical common-set evaluation

```bash
python3 scripts/um_study.py eval-historical --workers 8
```

Six unique inference conditions on the 25,000-image common set, all in evaluation mode with
historical BatchNorm buffers loaded and fixed:

| Condition | Mask family |
|---|---|
| `B1_pristine_identity` | none — this is also B4's identity counterpart |
| `B4_pristine_phase1mask` | Phase 1 **raw** gains |
| `B2_phase3_2_mask` / `_identity` | Phase 3.2 **mean-normalised** gains |
| `B3_v5_mask` / `_identity` | v5 **bounded sigmoid** gains |

Each historical family is its original one. The post-hoc unit-mean plotting convention is
never applied inside a forward pass. Writes per-image CSVs plus `summary.json`, including
the paired mask-on-minus-identity difference within each fitted classifier.

## Step 4 — work item C9, the throughput pilot

```bash
python3 scripts/um_study.py pilot --steps 60 --workers 8          # fp32
python3 scripts/um_study.py pilot --steps 60 --workers 8 --amp    # if fp32 is too slow
```

Measures arm M — the slower arm — and reports images/sec, minutes per training epoch, peak
GPU memory, and how many epochs fit in 2/4/8 hours per arm. Pilot weights are discarded and
nothing is saved beyond `pilot.json`. **Both arms reset before the counted comparison.**

Pick the shared epoch budget from this number and the real deadline, and record it before
looking at any comparative result.

## Step 5 — HELD: the counted pair

Do not run until the implementation has been reviewed and the budget fixed.

```bash
# two arms, one GPU each, concurrent
CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/um_study.py train --arm U --epochs <N> \
    > ~/VIT/um_study/U.log 2>&1 & disown
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/um_study.py train --arm M --epochs <N> \
    > ~/VIT/um_study/M.log 2>&1 & disown
```

Same `<N>` for both — that is the point of the comparison. No early stopping; `final.pt` is
the primary endpoint, fixed in advance, and `best_dev.pt` is secondary.

```bash
python3 scripts/um_study.py eval-pair --endpoint final --workers 8
```

Reports M − U on top-1 and cross-entropy with paired image-level uncertainty, plus M with
its mask removed. That uncertainty is conditional on these two fitted models and does not
estimate training-seed variability.

## If a check fails

Repair the protocol, record the change, and run a **fresh matched pair**. Never replace one
arm selectively, and never tune against a comparative result.

## What this design fixes relative to the historical runs

- single process, so no rank-local validation and no aggregation ambiguity
- BatchNorm buffers fixed in both arms, verified by C5 rather than assumed
- both arms traverse the identical FFT→IFFT path; U just omits the multiply
- mask starts at exact identity (`w=0` ⇒ `m=1`) with zero weight decay, so the
  shrink-toward-identity confound cannot arise
- augmentation is a pure function of (seed, epoch, row), verified by C6
- fixed budget, no early stopping, endpoint declared before results are seen
