# Cluster Setup & Run Guide

## 1. SSH into Neptun

```bash
ssh bab61wot@neptun.cs.uni-kl.de
```

---

## 2. Copy dataset to fast local storage

```bash
mkdir -p /mnt/local_learning/data/bab61wot
cp -r ~/VIT/SIM2REAL/data/imagenet_100k_cache/ /mnt/local_learning/data/bab61wot/
```

---

## 3. Install dependencies

```bash
cd ~/SIM2REAL_CLUSTER
pip3 install -r requirements.txt
```

---

## 4. Run experiments

**Single architecture (for testing):**
```bash
python3 scripts/train_phase3.2.py --arch resnet18 --data-dir /mnt/local_learning/data/bab61wot/imagenet_100k_cache
```

**All 4 architectures in parallel (one per GPU), runs in background:**
```bash
nohup bash run_experiments.sh &
```

---

## 5. Monitor progress

```bash
tail -f logs/resnet18.log
tail -f logs/alexnet.log
tail -f logs/vgg16.log
tail -f logs/resnet50.log
```

Press `Ctrl+C` to stop tailing. Training continues in background.

---

## 6. After training — clean up local storage

```bash
rm -rf /mnt/local_learning/data/bab61wot/
```

Results are saved to `~/SIM2REAL_CLUSTER/experiments/results/` which is on your persistent home directory.
