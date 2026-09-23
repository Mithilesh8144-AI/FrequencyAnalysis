#!/usr/bin/env python3
"""U/M matched pair + historical common-set evaluation.

Implements Orchestration/PLAN_FROM_HERE_2026-09-22.md work items B and C.

Single file by design: it has to be moved to the cluster in one piece.

Subcommands
-----------
  preflight          C1-C8 technical checks. Run first; all must PASS.
  pilot              C9 throughput measurement on the slower arm (M).
  verify-historical  Reconstruct historical pipelines and validate them end to end.
  eval-historical    Work item B: six inference conditions on the 25k common set.
  train              Work item C: one arm of the matched pair. HELD until reviewed.
  eval-pair          Work item C: final paired evaluation of the two arms.

Design constraints carried from the plan and the reviewer sign-off:
  * Both arms traverse the identical FFT -> IFFT path; U simply omits the multiply.
  * New mask family: symmetric bounded 2*sigmoid(w_sym), w=0 so m == 1 exactly.
  * Historical pipelines keep their ORIGINAL mask families -- Phase 1 raw gains,
    Phase 2 v5 bounded sigmoid, Phase 3.2 mean-normalised gains. The post-hoc
    unit-mean plotting convention is never applied in a forward pass.
  * BatchNorm buffers stay fixed in every arm and every evaluation: the classifier
    is held in eval() mode throughout, with requires_grad on parameters only.
  * Augmentation randomness is derived from (seed, epoch, row index), so the two
    arms see byte-identical transformed inputs regardless of worker scheduling.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# --------------------------------------------------------------------------
# Fixed configuration. Declared here, not tuned against comparative results.
# --------------------------------------------------------------------------

CFG = dict(
    image_size=224,
    batch_size=64,
    classifier_lr=1e-5,
    mask_lr=1e-3,
    classifier_wd=1e-4,
    mask_wd=0.0,
    grad_clip=1.0,
    optimizer="adam",
    schedule="constant",
    split_seed=42,          # matches the historical 80k/20k split
    run_seed=0,
    train_frac=0.8,
    amp=False,              # fp32 unless the pilot says otherwise
)

HOME = Path(os.path.expanduser("~"))
PATHS = dict(
    pool_100k=HOME / "VIT/SIM2REAL/data/imagenet_100k_cache",
    eval_25k=HOME / "VIT/SIM2REAL/data/imagenet_25k_cache",
    ckpt_v5=HOME / "VIT/SIM2REAL_CLUSTER/experiments/results/resnet18_phase2/best_model.pt",
    ckpt_p32=HOME / "VIT/SIM2REAL/experiments/results/resnet18_phase3.2/best_model.pt",
    mask_p1=HOME / "VIT/SIM2REAL/experiments/results/resnet18/learned_mask.pt",
    out=HOME / "VIT/um_study",
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------

class SymmetricBoundedMask(nn.Module):
    """New family for arm M: m = 2*sigmoid(w_sym), w_sym conjugate-symmetric.

    The pairing i -> (N-i) mod N maps each frequency bin onto its conjugate
    partner on the fftshifted grid, with index 0 (Nyquist) self-paired.
    Averaging w with its pairing makes w_sym -- and therefore m -- exactly
    symmetric, so m(k) == m(-k) holds by construction rather than by penalty.
    w initialised at zero gives m == 1.0 exactly, i.e. exact identity.
    """

    def __init__(self, n=224):
        super().__init__()
        self.n = n
        self.mask_weights = nn.Parameter(torch.zeros(1, 1, n, n))
        self.register_buffer("pair_idx", (n - torch.arange(n)) % n, persistent=False)

    def effective(self):
        w = self.mask_weights
        w_flip = w[:, :, self.pair_idx, :][:, :, :, self.pair_idx]
        return 2.0 * torch.sigmoid(0.5 * (w + w_flip))


class HistoricalMask(nn.Module):
    """Replicates frequency/mask.py::Learnable2DFrequencyMask._apply_activation.

    family: 'raw'       -> m = w                  (Phase 1)
            'sigmoid'   -> m = 2*sigmoid(w)       (Phase 2 v4/v5, lambda run)
            'normalize' -> m = w / (mean(w)+1e-8) (Phase 3.x)
    """

    def __init__(self, family, n=224):
        super().__init__()
        assert family in ("raw", "sigmoid", "normalize")
        self.family = family
        self.mask_weights = nn.Parameter(torch.ones(1, 1, n, n))

    def effective(self):
        w = self.mask_weights
        if self.family == "sigmoid":
            return 2.0 * torch.sigmoid(w)
        if self.family == "normalize":
            return w / (w.mean() + 1e-8)
        return w


class FreqPipeline(nn.Module):
    """Image -> FFT -> (optional mask) -> IFFT -> classifier.

    Module names are `classifier` and `freq_mask` so historical
    `pipeline_state_dict` payloads load directly.
    """

    def __init__(self, classifier, mask_module=None):
        super().__init__()
        self.classifier = classifier
        self.freq_mask = mask_module

    def forward(self, x, use_mask=True):
        fr = torch.fft.fft2(x.float(), dim=(-2, -1))
        fr = torch.fft.fftshift(fr, dim=(-2, -1))
        if self.freq_mask is not None and use_mask:
            fr = fr * self.freq_mask.effective()
        fr = torch.fft.ifftshift(fr, dim=(-2, -1))
        rec = torch.fft.ifft2(fr, dim=(-2, -1)).real
        return self.classifier(rec.to(x.dtype))


def freeze_bn_eval(pipeline):
    """Classifier held in eval() so BatchNorm buffers never update.

    Parameters still receive gradients; only the running statistics are frozen.
    ResNet-18 has no dropout, so nothing else changes between modes.
    """
    pipeline.classifier.eval()
    return pipeline


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def build_transforms(size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


class SeededSubset(Dataset):
    """Subset with augmentation randomness pinned to (seed, epoch, row index).

    Reseeding the global torch RNG inside __getitem__ is deliberate: torchvision
    v1 transforms draw from it, so this makes the transformed tensor a pure
    function of the row and the epoch. Both arms therefore see byte-identical
    inputs regardless of worker count or scheduling order (check C6).
    """

    def __init__(self, hf_dataset, indices, transform, augment, seed=0):
        self.ds = hf_dataset
        self.indices = list(indices)
        self.transform = transform
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.indices[i]
        item = self.ds[row]
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.augment:
            s = (self.seed * 1_000_003 + self.epoch * 10_007 + row) % (2**31 - 1)
            torch.manual_seed(s)
        x = self.transform(img)
        return x, int(item["label"]), int(row)


def manifest_path():
    return PATHS["out"] / "preflight" / "manifest.json"


REQUIRED_CHECKS = (
    "C1_identity_fft", "C2_mask_symmetry", "C3_identical_start", "C4_mask_gradients",
    "C5_U_bn_buffers_fixed", "C5_U_classifier_updates",
    "C5_M_bn_buffers_fixed", "C5_M_classifier_updates",
    "C6_streams_match", "C6_loader_reuse_epoch_varies", "C6_ids_stable",
    "C6_reuse_matches_fresh",
    "C7_disjoint_after_resolution",
    "C8_U_state_roundtrip", "C8_U_prediction_roundtrip",
    "C8_M_state_roundtrip", "C8_M_prediction_roundtrip",
)

MANIFEST_REQUIRED_FIELDS = ("hash_basis", "raw_overlap_encoded", "raw_overlap_decoded",
                            "manifest_id", "train", "dev", "eval")


def load_manifest(required=True):
    p = manifest_path()
    if not p.exists():
        if required:
            raise SystemExit(
                f"No manifest at {p}. Run `preflight` (without --skip-hash) first: "
                "the cleaned, duplicate-resolved partitions are defined there.")
        return None
    m = json.load(open(p))
    missing = [k for k in MANIFEST_REQUIRED_FIELDS if k not in m]
    if missing:
        raise SystemExit(
            f"Manifest at {p} is missing {missing}. It predates the decoded-content "
            "check; re-run `preflight` to regenerate it. Old byte-only manifests are "
            "not accepted.")
    if m["hash_basis"] != "decoded_rgb_pixels":
        raise SystemExit(f"Manifest hash_basis is {m['hash_basis']!r}, expected "
                         "'decoded_rgb_pixels'. Re-run `preflight`.")
    print(f"  manifest {m['manifest_id'][:16]}… train={m['n_train']} dev={m['n_dev']} "
          f"eval={m['n_eval']}")
    print(f"  overlaps before resolution — decoded: {m['raw_overlap_decoded']}")
    print(f"                               encoded: {m['raw_overlap_encoded']}")
    return m


def load_pool_split(pool_path, train_frac, split_seed):
    """Reproduces the historical split exactly: random_split over range(N) at seed 42."""
    from datasets import load_from_disk
    ds = load_from_disk(str(pool_path))
    n = len(ds)
    n_train = int(train_frac * n)
    gen = torch.Generator().manual_seed(split_seed)
    tr, va = random_split(range(n), [n_train, n - n_train], generator=gen)
    return ds, list(tr.indices), list(va.indices)


def make_loader(dataset, batch_size, shuffle, seed, workers=8):
    """persistent_workers is deliberately OFF.

    Worker processes hold their own copy of the dataset object. With persistent
    workers, a parent-side `set_epoch()` never reaches them, so every epoch would
    silently reuse the first epoch's augmentation. Recreating workers each epoch
    costs a little startup time and keeps the epoch-dependent augmentation real.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    def _winit(wid):
        base = (seed * 7919 + wid) % (2**31 - 1)
        np.random.seed(base)
        torch.manual_seed(base)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=gen if shuffle else None,
        num_workers=workers, pin_memory=True, worker_init_fn=_winit,
        persistent_workers=False, drop_last=False,
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(pipeline, loader, device, use_mask=True, amp=False):
    """Full-set evaluation with per-image records. eval() mode throughout."""
    pipeline.eval()
    crit = nn.CrossEntropyLoss(reduction="none")
    rows, n, c1, c5, loss_sum = [], 0, 0, 0, 0.0
    for x, y, idx in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = pipeline(x, use_mask=use_mask)
        out = out.float()
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite logits during evaluation")
        losses = crit(out, y)
        if not torch.isfinite(losses).all():
            raise RuntimeError("non-finite per-example loss during evaluation")
        top5 = out.topk(5, dim=1).indices
        hit1 = (top5[:, 0] == y)
        hit5 = (top5 == y.unsqueeze(1)).any(dim=1)
        n += y.numel()
        c1 += int(hit1.sum())
        c5 += int(hit5.sum())
        loss_sum += float(losses.sum())
        for j in range(y.numel()):
            rows.append((int(idx[j]), int(y[j]), int(top5[j, 0]),
                         int(hit1[j]), int(hit5[j]), float(losses[j])))
    return dict(n=n, top1=100.0 * c1 / n, top5=100.0 * c5 / n,
                loss=loss_sum / n, correct1=c1, correct5=c5), rows


def save_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("row_index,label,pred_top1,correct1,correct5,loss\n")
        for r in rows:
            f.write("%d,%d,%d,%d,%d,%.6f\n" % r)


def load_rows(path):
    """Read a per-image CSV into {row_index: (label, correct1, correct5, loss)}.

    Rejects duplicate row indices rather than silently overwriting them.
    """
    out = {}
    with open(path) as f:
        f.readline()
        for line in f:
            q = line.strip().split(",")
            if not q or not q[0]:
                continue
            i = int(q[0])
            if i in out:
                raise ValueError(f"duplicate row index {i} in {path}")
            out[i] = (int(q[1]), int(q[3]), int(q[4]), float(q[5]))
    return out


def paired_diff(path_a, path_b, field="correct1", expect_n=None,
                n_boot=10000, seed=0, chunk=500):
    """Paired image-level uncertainty on (b - a).

    Validates that both files cover exactly the same row IDs with the same
    labels before pairing -- a truncated or mismatched CSV must fail loudly
    rather than yield a plausible interval over fewer examples.

    Conditional on the two fitted models. Says nothing about seed variability.
    """
    A, B = load_rows(path_a), load_rows(path_b)
    if set(A) != set(B):
        d = set(A) ^ set(B)
        raise ValueError(f"row-ID sets differ between {path_a} and {path_b}: "
                         f"{len(d)} in symmetric difference")
    ids = sorted(A)
    bad = [i for i in ids if A[i][0] != B[i][0]]
    if bad:
        raise ValueError(f"label mismatch on {len(bad)} rows, e.g. {bad[:5]}")
    if expect_n is not None and len(ids) != expect_n:
        raise ValueError(f"expected {expect_n} paired rows, found {len(ids)}")

    col = {"correct1": 1, "correct5": 2, "loss": 3}[field]
    scale = 100.0 if field.startswith("correct") else 1.0
    d = np.array([B[i][col] - A[i][col] for i in ids], dtype=np.float64)

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    done, n = 0, len(d)
    while done < n_boot:                      # bounded memory, not (n_boot x n)
        k = min(chunk, n_boot - done)
        means[done:done + k] = d[rng.integers(0, n, size=(k, n))].mean(axis=1)
        done += k
    means *= scale
    unit = "pp" if field.startswith("correct") else "nats"
    return dict(n_paired=n, field=field, unit=unit,
                mean_diff=float(d.mean() * scale),
                ci_lo=float(np.percentile(means, 2.5)),
                ci_hi=float(np.percentile(means, 97.5)))


# --------------------------------------------------------------------------
# Historical reconstruction
# --------------------------------------------------------------------------

HISTORICAL = {
    "p32": dict(ckpt="ckpt_p32", family="normalize", score_key="val_acc",
                recorded=75.385, label="Phase 3.2 R18 (epoch-12 peak)"),
    "v5": dict(ckpt="ckpt_v5", family="sigmoid", score_key="val_acc1",
               recorded=61.27077458450831, label="Phase 2 v5 R18"),
}


def build_historical(name, device):
    spec = HISTORICAL[name]
    payload = torch.load(PATHS[spec["ckpt"]], map_location="cpu", weights_only=False)
    sd = payload["pipeline_state_dict"]
    clf = tv_models.resnet18(weights=None, num_classes=1000)
    pipe = FreqPipeline(clf, HistoricalMask(spec["family"]))
    pipe.load_state_dict(sd, strict=True)   # raises on any key or shape mismatch
    pipe.to(device).eval()
    return pipe, payload, spec


def build_pristine(device, mask_module=None):
    clf = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    pipe = FreqPipeline(clf, mask_module).to(device).eval()
    return pipe


def load_phase1_mask():
    """Phase 1 saved only freq_mask.state_dict(); gains are RAW (no activation)."""
    sd = torch.load(PATHS["mask_p1"], map_location="cpu", weights_only=False)
    w = sd["mask_weights"] if isinstance(sd, dict) and "mask_weights" in sd else sd
    m = HistoricalMask("raw")
    with torch.no_grad():
        m.mask_weights.copy_(w.view(1, 1, 224, 224))
    return m


# --------------------------------------------------------------------------
# Subcommand: preflight (C1-C8)
# --------------------------------------------------------------------------

def hash_images(ds_path):
    """Return (encoded_hashes, decoded_hashes) in a single pass.

    The encoded hash covers the stored bytes. The decoded hash covers canonical
    RGB dimensions plus raw pixel bytes, so it also catches the same image stored
    under a different encoding -- the check the plan actually specified. Decoded
    hashing strictly subsumes encoded hashing, so resolution uses the decoded one
    and both counts are reported.
    """
    import io
    import datasets
    from PIL import Image
    ds = datasets.load_from_disk(str(ds_path))
    ds = ds.cast_column("image", datasets.Image(decode=False))
    enc, dec = [], []
    n = len(ds)
    for i in range(n):
        b = ds[i]["image"]["bytes"]
        enc.append(hashlib.sha256(b).hexdigest())
        im = Image.open(io.BytesIO(b))
        if im.mode != "RGB":
            im = im.convert("RGB")
        h = hashlib.sha256()
        h.update(f"{im.size[0]}x{im.size[1]}|".encode())
        h.update(im.tobytes())
        dec.append(h.hexdigest())
        if (i + 1) % 20000 == 0:
            print(f"      hashed {i+1}/{n}", flush=True)
    return enc, dec


def cmd_preflight(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    outdir = PATHS["out"] / "preflight"
    outdir.mkdir(parents=True, exist_ok=True)

    def report(tag, ok, detail, gate=True):
        results[tag] = dict(pass_=bool(ok), detail=detail, gating=bool(gate))
        mark = ("PASS" if ok else "FAIL") if gate else "INFO"
        print(f"  [{mark}] {tag}: {detail}", flush=True)

    print("== C1 identity FFT/IFFT equivalence ==")
    x = torch.randn(4, 3, 224, 224, device=device)
    fr = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    rec = torch.fft.ifft2(torch.fft.ifftshift(fr, dim=(-2, -1)), dim=(-2, -1)).real
    err = float((rec - x).abs().max())
    report("C1_identity_fft", err < 1e-4, f"max|IFFT(FFT(x))-x| = {err:.3e}")

    print("== C2 symmetric mask construction ==")
    m = SymmetricBoundedMask().to(device)
    eff0 = m.effective()
    ident_err = float((eff0 - 1.0).abs().max().detach())
    with torch.no_grad():
        m.mask_weights.normal_(0, 0.5)
    eff = m.effective()
    idx = m.pair_idx
    sym_err = float((eff - eff[:, :, idx, :][:, :, :, idx]).abs().max().detach())
    report("C2_mask_symmetry", ident_err == 0.0 and sym_err < 1e-6,
           f"|m(w=0)-1|max = {ident_err:.3e}; |m(k)-m(-k)|max = {sym_err:.3e}")

    print("== C3/C4/C5 gradient, state and BatchNorm behaviour ==")
    torch.manual_seed(0)
    clf_u = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    clf_m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    sd_u = {k: v.clone() for k, v in clf_u.state_dict().items()}
    same_start = all(torch.equal(sd_u[k], clf_m.state_dict()[k]) for k in sd_u)
    report("C3_identical_start", same_start,
           f"{len(sd_u)} tensors compared across both arms")

    for arm_tag, clf_x in (("U", clf_u), ("M", clf_m)):
        pipe_x = build_arm(arm_tag, device)
        bn0 = {k: v.clone() for k, v in pipe_x.classifier.state_dict().items()
               if "running_" in k or "num_batches_tracked" in k}
        cl0 = {k: v.clone() for k, v in pipe_x.classifier.state_dict().items()
               if "running_" not in k and "num_batches_tracked" not in k}
        ox = build_optimizer(pipe_x, arm_tag)
        xq = torch.randn(8, 3, 224, 224, device=device)
        yq = torch.randint(0, 1000, (8,), device=device)
        ox.zero_grad()
        nn.CrossEntropyLoss()(pipe_x(xq), yq).backward()
        ox.step()
        sdx = pipe_x.classifier.state_dict()
        bn_ok = all(torch.equal(bn0[k].cpu(), sdx[k].cpu()) for k in bn0)
        cl_ok = any(not torch.equal(cl0[k].cpu(), sdx[k].cpu()) for k in cl0)
        report(f"C5_{arm_tag}_bn_buffers_fixed", bn_ok, f"{len(bn0)} buffers unchanged after step")
        report(f"C5_{arm_tag}_classifier_updates", cl_ok, "classifier parameters changed after step")

    pipe_m = build_arm("M", device)
    xb = torch.randn(8, 3, 224, 224, device=device)
    yb = torch.randint(0, 1000, (8,), device=device)
    opt = build_optimizer(pipe_m, "M")
    opt.zero_grad()
    nn.CrossEntropyLoss()(pipe_m(xb), yb).backward()
    g = pipe_m.freq_mask.mask_weights.grad
    g_ok = g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
    report("C4_mask_gradients", g_ok,
           f"finite={bool(torch.isfinite(g).all())}, sum|g|={float(g.abs().sum()):.3e}")

    print("== C6 matched augmentation streams ==")
    ds, tr_idx, va_idx = load_pool_split(PATHS["pool_100k"], CFG["train_frac"], CFG["split_seed"])
    train_tf, eval_tf = build_transforms()
    probe = sorted(tr_idx)[:64]

    def loader_batch_hash(epoch, workers):
        """Hash the first batch as it actually arrives through the DataLoader.

        Going through the real loader with real workers is the point: a
        parent-side set_epoch() that never reaches worker processes would pass
        a direct-indexing check and fail here.
        """
        sub = SeededSubset(ds, probe, train_tf, augment=True, seed=CFG["run_seed"])
        sub.set_epoch(epoch)
        dl = make_loader(sub, 16, False, CFG["run_seed"], workers)
        xb, yb, ib = next(iter(dl))
        return (hashlib.sha256(xb.numpy().tobytes()).hexdigest(), ib.tolist())

    w = max(2, args.workers)
    h3a, id3a = loader_batch_hash(3, w)
    h3b, id3b = loader_batch_hash(3, w)
    report("C6_streams_match", h3a == h3b and id3a == id3b,
           f"epoch-3 batch identical across two independent loaders, {w} workers ({h3a[:16]}…)")

    # The discriminating test: ONE loader, epoch changed on its parent dataset
    # between iterations. Under the original persistent-worker bug the workers
    # kept a stale copy and this would return the same batch twice.
    sub_r = SeededSubset(ds, probe, train_tf, augment=True, seed=CFG["run_seed"])
    dl_r = make_loader(sub_r, 16, False, CFG["run_seed"], w)
    sub_r.set_epoch(3)
    xb3, _, ib3 = next(iter(dl_r))
    sub_r.set_epoch(4)
    xb4, _, ib4 = next(iter(dl_r))
    r3 = hashlib.sha256(xb3.numpy().tobytes()).hexdigest()
    r4 = hashlib.sha256(xb4.numpy().tobytes()).hexdigest()
    report("C6_loader_reuse_epoch_varies", r3 != r4,
           f"same loader object, epoch 3 -> 4 changes the batch ({r3[:12]}… -> {r4[:12]}…)")
    report("C6_ids_stable", ib3.tolist() == ib4.tolist(),
           "sample IDs unchanged across epochs; only augmentation differs")
    report("C6_reuse_matches_fresh", r3 == h3a,
           "reused loader at epoch 3 matches a freshly built one")

    print("== C7 content disjointness and duplicate resolution ==")
    if args.skip_hash:
        report("C7_disjoint_after_resolution", False,
               "SKIPPED via --skip-hash (required before training)")
    else:
        t0 = time.time()
        print("   hashing 100k pool (encoded + decoded)…", flush=True)
        e100, d100 = hash_images(PATHS["pool_100k"])
        print("   hashing 25k eval set (encoded + decoded)…", flush=True)
        e25, d25 = hash_images(PATHS["eval_25k"])

        def counts(h100, h25):
            tr_h = {h100[i] for i in tr_idx}
            va_h = {h100[i] for i in va_idx}
            ev_h = set(h25)
            return dict(train_eval=len(tr_h & ev_h), dev_eval=len(va_h & ev_h),
                        train_dev=len(tr_h & va_h),
                        dups_100k=len(h100) - len(set(h100)),
                        dups_25k=len(h25) - len(set(h25)))

        raw_enc, raw_dec = counts(e100, e25), counts(d100, d25)
        report("C7_raw_overlap_encoded", True,
               f"stored-bytes hash — train∩eval={raw_enc['train_eval']}, "
               f"dev∩eval={raw_enc['dev_eval']}, train∩dev={raw_enc['train_dev']}, "
               f"dups 100k={raw_enc['dups_100k']}, 25k={raw_enc['dups_25k']}", gate=False)
        report("C7_raw_overlap_decoded", True,
               f"decoded RGB pixel hash — train∩eval={raw_dec['train_eval']}, "
               f"dev∩eval={raw_dec['dev_eval']}, train∩dev={raw_dec['train_dev']}, "
               f"dups 100k={raw_dec['dups_100k']}, 25k={raw_dec['dups_25k']}", gate=False)

        # Resolution uses the decoded hash, which subsumes the encoded one.
        h100, h25 = d100, d25
        tr_h = {h100[i] for i in tr_idx}
        train_clean = list(tr_idx)
        dev_clean = [i for i in va_idx if h100[i] not in tr_h]
        dev_h = {h100[i] for i in dev_clean}
        banned = tr_h | dev_h
        eval_clean, seen = [], set()
        for i, h in enumerate(h25):
            if h in banned or h in seen:
                continue
            seen.add(h)
            eval_clean.append(i)

        tr_h2 = {h100[i] for i in train_clean}
        dv_h2 = {h100[i] for i in dev_clean}
        ev_h2 = {h25[i] for i in eval_clean}
        ok7 = (not (tr_h2 & ev_h2) and not (dv_h2 & ev_h2) and not (tr_h2 & dv_h2)
               and len(ev_h2) == len(eval_clean))
        report("C7_disjoint_after_resolution", ok7,
               f"train={len(train_clean)} dev={len(dev_clean)} eval={len(eval_clean)} "
               f"(dropped {len(va_idx)-len(dev_clean)} dev, {len(h25)-len(eval_clean)} eval); "
               f"all pairwise intersections 0 on decoded content  [{time.time()-t0:.0f}s]")
        # Training-set internal duplicates are left in place deliberately: they are
        # not leakage, and removing them would change the historical 80,000 pool.
        report("C7_train_internal_dups", True,
               f"{len(train_clean)-len(tr_h2)} duplicate images remain inside train "
               f"(not leakage; pool kept at the historical size)", gate=False)

        manifest = dict(
            created="preflight", hash_basis="decoded_rgb_pixels",
            pool_100k=str(PATHS["pool_100k"]), eval_25k=str(PATHS["eval_25k"]),
            split_seed=CFG["split_seed"], train_frac=CFG["train_frac"],
            raw_overlap_encoded=raw_enc, raw_overlap_decoded=raw_dec,
            train=train_clean, dev=dev_clean, eval=eval_clean,
            n_train=len(train_clean), n_dev=len(dev_clean), n_eval=len(eval_clean),
            fingerprint_100k=hashlib.sha256("".join(d100).encode()).hexdigest(),
            fingerprint_25k=hashlib.sha256("".join(d25).encode()).hexdigest(),
            torch=torch.__version__,
            transforms="RandomResizedCrop(224,(0.8,1.0))+RandomHorizontalFlip / Resize(224)",
        )
        manifest["manifest_id"] = hashlib.sha256(
            json.dumps({k: manifest[k] for k in sorted(manifest)}, sort_keys=True,
                       default=str).encode()).hexdigest()
        json.dump(manifest, open(outdir / "manifest.json", "w"))
        print(f"   manifest written to {outdir/'manifest.json'}  id={manifest['manifest_id'][:16]}…")

    print("== C8 checkpoint round-trip for the new writer, both arms ==")
    for arm_tag in ("U", "M"):
        pc = build_arm(arm_tag, device)
        with torch.no_grad():
            for prm in pc.classifier.parameters():
                prm.add_(torch.randn_like(prm) * 1e-3)   # move off the pristine state
            if pc.freq_mask is not None:
                pc.freq_mask.mask_weights.normal_(0, 0.3)
        xq = torch.randn(4, 3, 224, 224, device=device)
        with torch.no_grad():
            y1 = pc(xq)
        tmp = outdir / f"_c8_{arm_tag}.pt"
        torch.save({"pipeline_state_dict": pc.state_dict()}, tmp)
        pr = build_arm(arm_tag, device)
        pr.load_state_dict(torch.load(tmp, map_location=device,
                                      weights_only=False)["pipeline_state_dict"])
        pr.to(device)
        freeze_bn_eval(pr)
        with torch.no_grad():
            y2 = pr(xq)
        sd1, sd2 = pc.state_dict(), pr.state_dict()
        tmp.unlink()
        report(f"C8_{arm_tag}_state_roundtrip",
               all(torch.equal(sd1[k].cpu(), sd2[k].cpu()) for k in sd1),
               f"{len(sd1)} tensors identical after save/reload (buffers and mask included)")
        report(f"C8_{arm_tag}_prediction_roundtrip",
               bool(torch.equal(y1, y2)),
               f"logits bit-identical after reload; max|Δ|={float((y1-y2).abs().max()):.3e}")

    missing = [k for k in REQUIRED_CHECKS if k not in results]
    if missing:
        print(f"  [FAIL] coverage: required checks not run: {missing}")
    ok = all(v["pass_"] for v in results.values() if v.get("gating", True)) and not missing
    payload = dict(checks=results, required=list(REQUIRED_CHECKS), missing=missing,
                   all_passed=ok, torch=torch.__version__,
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    mp = manifest_path()
    if mp.exists():
        payload["manifest_id"] = json.load(open(mp))["manifest_id"]
    json.dump(payload, open(outdir / "preflight.json", "w"), indent=2, default=str)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'ONE OR MORE CHECKS FAILED'} "
          f"-> {outdir/'preflight.json'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Subcommand: verify-historical (C8 for the recovered states)
# --------------------------------------------------------------------------

def cmd_verify_historical(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_tf = build_transforms()
    out = {}

    for name in ("p32", "v5"):
        print(f"== {HISTORICAL[name]['label']} ==")
        pipe, payload, spec = build_historical(name, device)
        sd = payload["pipeline_state_dict"]
        n_bn = sum(1 for k in sd if "running_" in k)
        print(f"   state keys: {len(sd)} | BN buffers: {n_bn} | family: {spec['family']}")
        print(f"   recorded {spec['score_key']} = {payload[spec['score_key']]}")

        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            y = pipe(x)
        finite = bool(torch.isfinite(y).all())
        print(f"   forward pass finite: {finite}, logits {tuple(y.shape)}")

        eff = pipe.freq_mask.effective().detach().cpu().numpy().squeeze()
        print(f"   effective mask: mean={eff.mean():.6f} std={eff.std():.6f} "
              f"min={eff.min():.4f} max={eff.max():.4f}")

        entry = dict(state_keys=len(sd), bn_buffers=n_bn, forward_finite=finite,
                     recorded_score=float(payload[spec["score_key"]]),
                     mask_mean=float(eff.mean()), mask_std=float(eff.std()))

        # Cross-reference the recovered mask against the exported array, if present.
        if args.export_dir:
            for cand in Path(args.export_dir).rglob("RAWMASK_best_model.npy"):
                arr = np.load(cand)
                raw = pipe.freq_mask.mask_weights.detach().cpu().numpy().squeeze()
                if arr.shape == raw.shape:
                    d = float(np.abs(arr - raw).max())
                    if d < 1e-5:
                        print(f"   mask matches export: {cand} (max diff {d:.2e})")
                        entry["export_match"] = str(cand)
                        break

        # Phase 3.2's recorded score was measured on the 100k pool's own 20k
        # development split, which still exists -- so it can be reproduced exactly.
        # v5's was measured on full-ImageNet validation, which was deleted.
        if name == "p32" and not args.skip_reproduce:
            ds, tr_idx, va_idx = load_pool_split(PATHS["pool_100k"], CFG["train_frac"], CFG["split_seed"])
            dev = SeededSubset(ds, va_idx, eval_tf, augment=False)
            loader = make_loader(dev, CFG["batch_size"], False, CFG["run_seed"], args.workers)
            t0 = time.time()
            stats, _ = evaluate(pipe, loader, device, use_mask=True, amp=False)
            dt = time.time() - t0
            delta = stats["top1"] - spec["recorded"]
            print(f"   REPRODUCTION on its own 20k dev split: top1={stats['top1']:.3f}% "
                  f"(recorded {spec['recorded']}), delta={delta:+.3f} pp, n={stats['n']}, {dt:.0f}s")
            entry.update(reproduced_top1=stats["top1"], reproduction_delta=delta,
                         reproduction_n=stats["n"], reproduction_seconds=dt)
        out[name] = entry

    PATHS["out"].mkdir(parents=True, exist_ok=True)
    json.dump(out, open(PATHS["out"] / "verify_historical.json", "w"), indent=2)
    print(f"\nwritten -> {PATHS['out']/'verify_historical.json'}")
    return 0


# --------------------------------------------------------------------------
# Subcommand: eval-historical (work item B)
# --------------------------------------------------------------------------

def cmd_eval_historical(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from datasets import load_from_disk
    _, eval_tf = build_transforms()
    man = load_manifest()
    eval_ds = load_from_disk(str(PATHS["eval_25k"]))
    common = SeededSubset(eval_ds, man["eval"], eval_tf, augment=False)
    loader = make_loader(common, CFG["batch_size"], False, CFG["run_seed"], args.workers)
    outdir = PATHS["out"] / "historical"
    outdir.mkdir(parents=True, exist_ok=True)

    # Six unique inference conditions. B4's identity counterpart IS B1.
    conditions = []
    conditions.append(("B1_pristine_identity", build_pristine(device, None), True,
                       "reference | validation-derived; prior project exposure disclosed"))
    conditions.append(("B4_pristine_phase1mask", build_pristine(device, load_phase1_mask().to(device)), True,
                       "reconstructed combination | fitting-exposed"))
    for name in ("p32", "v5"):
        pipe, payload, spec = build_historical(name, device)
        tag = "B2_phase3_2" if name == "p32" else "B3_v5"
        conditions.append((f"{tag}_mask", pipe, True, spec["label"] + " | learned mask"))
        conditions.append((f"{tag}_identity", pipe, False, spec["label"] + " | mask replaced by identity"))

    results = {}
    for tag, pipe, use_mask, note in conditions:
        print(f"  running {tag} …", flush=True)
        t0 = time.time()
        stats, rows = evaluate(pipe, loader, device, use_mask=use_mask, amp=False)
        dt = time.time() - t0
        save_rows(rows, outdir / f"{tag}.csv")
        stats["seconds"] = dt
        stats["note"] = note
        results[tag] = stats
        print(f"  {tag:28s} top1={stats['top1']:6.3f}% top5={stats['top5']:6.3f}% "
              f"loss={stats['loss']:.4f} n={stats['n']} [{dt:.0f}s]", flush=True)

    # Within-classifier mask-on vs identity, paired over images, plus the
    # Phase 1 reconstruction against the same pristine classifier (B4 vs B1),
    # and the Phase 3.2 pipeline against the pristine reference (B2 vs B1).
    n_exp = len(man["eval"])
    pairs = {}
    comparisons = [
        ("B2_phase3_2_mask_minus_identity", "B2_phase3_2_identity", "B2_phase3_2_mask"),
        ("B3_v5_mask_minus_identity", "B3_v5_identity", "B3_v5_mask"),
        ("B4_phase1mask_minus_identity", "B1_pristine_identity", "B4_pristine_phase1mask"),
        ("B2_phase3_2_minus_pristine", "B1_pristine_identity", "B2_phase3_2_mask"),
        ("B3_v5_minus_pristine", "B1_pristine_identity", "B3_v5_mask"),
    ]
    for tag, a, b in comparisons:
        pairs[tag] = {}
        for field in ("correct1", "correct5", "loss"):
            pairs[tag][field] = paired_diff(outdir / f"{a}.csv", outdir / f"{b}.csv",
                                            field=field, expect_n=n_exp,
                                            seed=CFG["run_seed"])
        t = pairs[tag]["correct1"]
        print(f"  {tag:38s} top-1 {t['mean_diff']:+7.3f} pp "
              f"[{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}]  n={t['n_paired']}", flush=True)

    json.dump(dict(conditions=results, paired=pairs), open(outdir / "summary.json", "w"), indent=2)
    print(f"\nwritten -> {outdir/'summary.json'}")
    return 0


# --------------------------------------------------------------------------
# Subcommand: pilot (C9) and train (work item C)
# --------------------------------------------------------------------------

def build_arm(arm, device):
    clf = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    mask = SymmetricBoundedMask() if arm == "M" else None
    pipe = FreqPipeline(clf, mask).to(device)
    freeze_bn_eval(pipe)
    for p in pipe.classifier.parameters():
        p.requires_grad_(True)
    return pipe


def build_optimizer(pipe, arm):
    groups = [{"params": pipe.classifier.parameters(),
               "lr": CFG["classifier_lr"], "weight_decay": CFG["classifier_wd"]}]
    if arm == "M":
        groups.insert(0, {"params": pipe.freq_mask.parameters(),
                          "lr": CFG["mask_lr"], "weight_decay": CFG["mask_wd"]})
    return optim.Adam(groups)


class NumericalFailure(RuntimeError):
    """Raised at the point a non-finite loss or gradient occurs."""


def train_one_epoch(pipe, loader, opt, scaler, device, epoch, amp, arm, outdir,
                    log_every=50):
    crit = nn.CrossEntropyLoss()
    pipe.classifier.eval()              # BN buffers stay frozen
    if pipe.freq_mask is not None:
        pipe.freq_mask.train()
    skipped, tot, c1, loss_sum, t0 = 0, 0, 0, 0.0, time.time()
    for i, (x, y, _) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = pipe(x)
            loss = crit(out, y)
        if not torch.isfinite(loss):
            _fail(outdir, arm, epoch, i, f"non-finite loss: {float(loss)}")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for grp in opt.param_groups:
            for prm in grp["params"]:
                if prm.grad is not None and not torch.isfinite(prm.grad).all():
                    if not amp:
                        _fail(outdir, arm, epoch, i, "non-finite gradient (fp32)")
            torch.nn.utils.clip_grad_norm_(grp["params"], CFG["grad_clip"])
        before = scaler.get_scale() if amp else None
        scaler.step(opt)
        scaler.update()
        if amp and scaler.get_scale() < before:
            skipped += 1               # AMP skipped this update; counted, not hidden
        tot += y.numel()
        c1 += int((out.float().argmax(1) == y).sum())
        loss_sum += float(loss) * y.numel()
        if log_every and i % log_every == 0:
            print(f"    ep{epoch} step {i}/{len(loader)} loss={float(loss):.4f} "
                  f"top1={100.0*c1/tot:.2f}% {tot/(time.time()-t0):.1f} img/s", flush=True)
    return dict(train_loss=loss_sum / tot, train_top1=100.0 * c1 / tot,
                n=tot, seconds=time.time() - t0, skipped_updates=skipped,
                steps=len(loader))


def _fail(outdir, arm, epoch, batch, msg):
    """Durable failure record, then stop. Never produce a normal endpoint."""
    rec = dict(arm=arm, epoch=epoch, batch=batch, message=msg,
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    Path(outdir).mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(Path(outdir) / "FAILURE.json", "w"), indent=2)
    raise NumericalFailure(f"[{arm}] epoch {epoch} batch {batch}: {msg}")


def cmd_pilot(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pilot on arm M (the slower arm), device={device}, amp={args.amp}")
    man = load_manifest(required=False)
    if man:
        from datasets import load_from_disk
        ds = load_from_disk(str(PATHS["pool_100k"]))
        tr_idx, dv_idx = man["train"], man["dev"]
    else:
        print("  (no manifest yet — timing only, using the raw seed-42 split)")
        ds, tr_idx, dv_idx = load_pool_split(PATHS["pool_100k"], CFG["train_frac"], CFG["split_seed"])
    train_tf, _ = build_transforms()
    sub = SeededSubset(ds, tr_idx, train_tf, augment=True, seed=CFG["run_seed"])
    sub.set_epoch(0)
    loader = make_loader(sub, CFG["batch_size"], True, CFG["run_seed"], args.workers)

    pipe = build_arm("M", device)
    opt = build_optimizer(pipe, "M")
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    n_seen, t0, warmup, measured = 0, None, 5, 0
    for i, (x, y, _) in enumerate(loader):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()      # drain warmup before the clock starts
            t0 = time.time(); n_seen = 0
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            loss = crit(pipe(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"pilot: non-finite loss at step {i}")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for grp in opt.param_groups:
            torch.nn.utils.clip_grad_norm_(grp["params"], CFG["grad_clip"])
        scaler.step(opt); scaler.update()
        if i >= warmup:
            n_seen += y.numel(); measured += 1
        if measured >= args.steps:            # exactly args.steps, not one more
            break
    if device.type == "cuda":
        torch.cuda.synchronize()              # all queued work done before reading

    dt = time.time() - t0
    ips = n_seen / dt
    epoch_s = len(tr_idx) / ips
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    est = {
        "images_per_second": ips,
        "seconds_per_train_epoch": epoch_s,
        "minutes_per_train_epoch": epoch_s / 60.0,
        "train_images": len(tr_idx),
        "peak_gpu_gib": mem,
        "amp": args.amp,
        "batch_size": CFG["batch_size"],
        "workers": args.workers,
        "measured_steps": measured,
        "measured_images": n_seen,
        "measured_seconds": dt,
    }
    for budget_h in (2, 4, 8):
        est[f"epochs_in_{budget_h}h_one_arm"] = int(budget_h * 3600 / epoch_s)

    # Per-epoch cost is not training alone: every epoch also evaluates the development
    # split and writes checkpoints. Measured here, not assumed negligible.
    _, eval_tf = build_transforms()
    dv = SeededSubset(ds, dv_idx, eval_tf, augment=False, seed=CFG["run_seed"])
    dv_loader = make_loader(dv, CFG["batch_size"], False, CFG["run_seed"], args.workers)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_dev = time.time()
    evaluate(pipe, dv_loader, device, use_mask=True, amp=args.amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dev_s = time.time() - t_dev

    PATHS["out"].mkdir(parents=True, exist_ok=True)
    tmp_ckpt = PATHS["out"] / f"_pilot_ckpt_{args.tag or 'x'}.pt"
    t_ck = time.time()
    torch.save({"pipeline_state_dict": pipe.state_dict()}, tmp_ckpt)
    ckpt_s = time.time() - t_ck
    tmp_ckpt.unlink(missing_ok=True)

    per_epoch_s = epoch_s + dev_s + 2 * ckpt_s      # final.pt and best_dev.pt
    est.update({
        "dev_images": len(dv_idx),
        "dev_eval_seconds": dev_s,
        "checkpoint_write_seconds": ckpt_s,
        "seconds_per_epoch_total": per_epoch_s,
        "minutes_per_epoch_total": per_epoch_s / 60.0,
        "declared_epochs": args.epochs,
        "estimated_run_minutes_one_arm": args.epochs * per_epoch_s / 60.0,
    })
    print(json.dumps(est, indent=2))
    name = f"pilot_{args.tag}.json" if args.tag else "pilot.json"
    json.dump(est, open(PATHS["out"] / name, "w"), indent=2)
    print(f"written -> {PATHS['out']/name}")
    print(f"\n{args.epochs} epochs at {per_epoch_s:.1f}s/epoch (train {epoch_s:.1f} + "
          f"dev {dev_s:.1f} + checkpoints {2*ckpt_s:.1f}) "
          f"= {args.epochs*per_epoch_s/60.0:.1f} min per arm.")
    print("\nPilot weights are discarded; nothing was saved. Reset before the counted pair.")
    return 0


def cmd_train(args):
    """HELD: do not launch until the implementation has been reviewed."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm = args.arm
    outdir = PATHS["out"] / f"arm_{arm}"
    existing = [f.name for f in (outdir.glob("*") if outdir.exists() else [])
                if f.name in ("final.pt", "initial.pt", "history.json", "best_dev.pt",
                              "FAILURE.json", "config.json")]
    if existing and not args.restart:
        raise SystemExit(
            f"{outdir} already holds run evidence {existing} — possibly a completed, "
            "interrupted or failed run. Counted runs are never overwritten. Pass "
            "--restart only for a documented, intentional restart.")
    pf = PATHS["out"] / "preflight" / "preflight.json"
    if not pf.exists():
        raise SystemExit(f"No preflight record at {pf}. Run `preflight` first.")
    pfj = json.load(open(pf))
    if "checks" not in pfj:
        raise SystemExit(f"{pf} predates the coverage gate. Re-run `preflight`.")
    miss = [k for k in REQUIRED_CHECKS if k not in pfj["checks"]]
    failed = [k for k, v in pfj["checks"].items()
              if v.get("gating", True) and not v.get("pass_")]
    if miss or failed:
        raise SystemExit(f"Preflight incomplete or failing. missing={miss} failed={failed}. "
                         "Not launching.")
    outdir.mkdir(parents=True, exist_ok=True)
    CFG["amp"] = args.amp
    CFG["epochs"] = args.epochs
    CFG["arm"] = arm

    man = load_manifest()
    from datasets import load_from_disk
    ds = load_from_disk(str(PATHS["pool_100k"]))
    tr_idx, va_idx = man["train"], man["dev"]
    CFG["n_train"], CFG["n_dev"] = len(tr_idx), len(va_idx)
    CFG["manifest_fingerprint_100k"] = man["fingerprint_100k"]
    CFG["manifest_fingerprint_25k"] = man["fingerprint_25k"]
    if pfj.get("manifest_id") != man["manifest_id"]:
        raise SystemExit(
            f"Preflight was recorded against manifest {pfj.get('manifest_id')}, but the "
            f"current manifest is {man['manifest_id']}. Re-run `preflight`.")
    CFG["manifest_id"] = man["manifest_id"]
    CFG["preflight_time"] = pfj.get("time")
    CFG["raw_overlap_decoded"] = man["raw_overlap_decoded"]
    CFG["raw_overlap_encoded"] = man["raw_overlap_encoded"]
    json.dump(CFG, open(outdir / "config.json", "w"), indent=2)
    train_tf, eval_tf = build_transforms()
    tr = SeededSubset(ds, tr_idx, train_tf, augment=True, seed=CFG["run_seed"])
    dv = SeededSubset(ds, va_idx, eval_tf, augment=False)
    tr_loader = make_loader(tr, CFG["batch_size"], True, CFG["run_seed"], args.workers)
    dv_loader = make_loader(dv, CFG["batch_size"], False, CFG["run_seed"], args.workers)

    torch.manual_seed(CFG["run_seed"])
    pipe = build_arm(arm, device)
    opt = build_optimizer(pipe, arm)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)   # one per run, not per epoch
    torch.save({"pipeline_state_dict": pipe.state_dict()}, outdir / "initial.pt")

    history, best = [], (-1.0, -1)
    for ep in range(1, args.epochs + 1):
        tr.set_epoch(ep)
        stats = train_one_epoch(pipe, tr_loader, opt, scaler, device, ep,
                                args.amp, arm, outdir)
        dev_stats, _ = evaluate(pipe, dv_loader, device, use_mask=True, amp=args.amp)
        rec = dict(epoch=ep, **stats, dev_top1=dev_stats["top1"], dev_top5=dev_stats["top5"],
                   dev_loss=dev_stats["loss"], dev_n=dev_stats["n"])
        if arm == "M":
            eff = pipe.freq_mask.effective().detach().cpu().numpy().squeeze()
            np.save(outdir / f"mask_epoch_{ep:03d}.npy", eff)
            rec.update(mask_mean=float(eff.mean()), mask_std=float(eff.std()),
                       mask_D=float(((eff - 1.0) ** 2).mean()))
        history.append(rec)
        print(f"  [{arm}] epoch {ep}/{args.epochs} train_top1={stats['train_top1']:.2f}% "
              f"dev_top1={dev_stats['top1']:.2f}% ({dev_stats['n']} imgs) "
              f"{stats['seconds']:.0f}s", flush=True)
        json.dump(history, open(outdir / "history.json", "w"), indent=2)
        if dev_stats["top1"] > best[0]:
            best = (dev_stats["top1"], ep)
            torch.save({"pipeline_state_dict": pipe.state_dict(), "epoch": ep,
                        "dev_top1": dev_stats["top1"], "config": CFG},
                       outdir / "best_dev.pt")
    CFG["skipped_updates_total"] = sum(h.get("skipped_updates", 0) for h in history)
    json.dump(CFG, open(outdir / "config.json", "w"), indent=2)
    torch.save({"pipeline_state_dict": pipe.state_dict(), "epoch": args.epochs,
                "dev_top1": history[-1]["dev_top1"], "config": CFG}, outdir / "final.pt")
    print(f"[{arm}] done. final epoch {args.epochs}; best dev {best[0]:.2f}% at epoch {best[1]}")
    print("Primary endpoint is final.pt, fixed in advance. best_dev.pt is secondary.")
    return 0


def cmd_eval_pair(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from datasets import load_from_disk
    _, eval_tf = build_transforms()
    man = load_manifest()
    eval_ds = load_from_disk(str(PATHS["eval_25k"]))
    common = SeededSubset(eval_ds, man["eval"], eval_tf, augment=False)
    loader = make_loader(common, CFG["batch_size"], False, CFG["run_seed"], args.workers)
    outdir = PATHS["out"] / "pair_eval"
    outdir.mkdir(parents=True, exist_ok=True)

    # The two arms must be comparable before anything is compared.
    payloads, cfgs = {}, {}
    for arm in ("U", "M"):
        ck = PATHS["out"] / f"arm_{arm}" / f"{args.endpoint}.pt"
        if not ck.exists():
            print(f"missing {ck}"); return 1
        payloads[arm] = torch.load(ck, map_location="cpu", weights_only=False)
        cfgs[arm] = payloads[arm].get("config", {})
    problems = []
    if payloads["U"].get("epoch") != payloads["M"].get("epoch"):
        problems.append(f"epoch mismatch: U={payloads['U'].get('epoch')} "
                        f"M={payloads['M'].get('epoch')}")
    for key in ("manifest_id", "classifier_lr", "classifier_wd", "batch_size",
                "split_seed", "run_seed", "grad_clip", "amp", "epochs"):
        if cfgs["U"].get(key) != cfgs["M"].get(key):
            problems.append(f"{key} mismatch: U={cfgs['U'].get(key)} M={cfgs['M'].get(key)}")
    if cfgs["U"].get("manifest_id") != man["manifest_id"]:
        problems.append("arms were trained against a different manifest than this evaluation")
    su, sm = cfgs["U"].get("skipped_updates_total"), cfgs["M"].get("skipped_updates_total")
    if (su or 0) != (sm or 0):
        problems.append(f"unequal AMP-skipped updates: U={su} M={sm}")
    if problems:
        raise SystemExit("Arms are not comparable:\n  - " + "\n  - ".join(problems))
    print(f"  arms comparable: epoch={payloads['U'].get('epoch')}, "
          f"manifest={man['manifest_id'][:16]}…")

    store, results = {}, {}
    for arm in ("U", "M"):
        pipe = build_arm(arm, device)
        pipe.load_state_dict(payloads[arm]["pipeline_state_dict"])
        pipe.to(device).eval()
        stats, rows = evaluate(pipe, loader, device, use_mask=True, amp=False)
        save_rows(rows, outdir / f"{arm}_{args.endpoint}.csv")
        store[arm] = rows
        results[arm] = stats
        print(f"  {arm}: top1={stats['top1']:.3f}% top5={stats['top5']:.3f}% "
              f"loss={stats['loss']:.4f} n={stats['n']}", flush=True)
        if arm == "M":
            s_id, r_id = evaluate(pipe, loader, device, use_mask=False, amp=False)
            save_rows(r_id, outdir / f"M_identity_{args.endpoint}.csv")
            results["M_identity"] = s_id
            store["M_identity"] = r_id
            print(f"  M with mask removed: top1={s_id['top1']:.3f}%")

    n_exp = len(man["eval"])
    out = dict(conditions=results, paired={})
    for tag, a, b in (("M_minus_U", f"U_{args.endpoint}", f"M_{args.endpoint}"),
                      ("M_minus_M_identity", f"M_identity_{args.endpoint}", f"M_{args.endpoint}")):
        out["paired"][tag] = {f: paired_diff(outdir / f"{a}.csv", outdir / f"{b}.csv",
                                             field=f, expect_n=n_exp, seed=CFG["run_seed"])
                              for f in ("correct1", "correct5", "loss")}
    out["scope"] = ("Single seed. Paired image-level uncertainty is conditional on these two "
                    "fitted models and does not estimate training-seed variability.")
    json.dump(out, open(outdir / "summary.json", "w"), indent=2)
    print(json.dumps(out["paired"]["M_minus_U"], indent=2))
    print(f"written -> {outdir/'summary.json'}")
    return 0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight"); p.add_argument("--skip-hash", action="store_true")
    p.add_argument("--workers", type=int, default=8); p.set_defaults(fn=cmd_preflight)

    p = sub.add_parser("verify-historical")
    p.add_argument("--export-dir", type=str, default=None)
    p.add_argument("--skip-reproduce", action="store_true")
    p.add_argument("--workers", type=int, default=8); p.set_defaults(fn=cmd_verify_historical)

    p = sub.add_parser("eval-historical")
    p.add_argument("--workers", type=int, default=8); p.set_defaults(fn=cmd_eval_historical)

    p = sub.add_parser("pilot")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--tag", default=None,
                   help="suffix for the output file, so concurrent pilots do not overwrite "
                        "each other: pilot_<tag>.json")
    p.add_argument("--epochs", type=int, default=15,
                   help="budget to cost out in the total-runtime estimate")
    p.add_argument("--workers", type=int, default=8); p.set_defaults(fn=cmd_pilot)

    p = sub.add_parser("train")
    p.add_argument("--arm", choices=["U", "M"], required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--restart", action="store_true",
                   help="allow overwriting a completed counted run (documented restarts only)")
    p.add_argument("--workers", type=int, default=8); p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval-pair")
    p.add_argument("--endpoint", choices=["final", "best_dev"], default="final")
    p.add_argument("--workers", type=int, default=8); p.set_defaults(fn=cmd_eval_pair)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
