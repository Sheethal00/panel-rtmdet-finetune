# Panel RTMDet-tiny fine-tuning flow

A reusable local pipeline for fine-tuning RTMDet-tiny on annotated distribution
panel images (COCO JSON format), producing a checkpoint and an ONNX export
matching the shape/provider assumptions already validated in `panel-bench`
(640x640 input, CPU execution provider).

This is meant to be reused as your annotated dataset grows — from today's
5–8 samples up through the full fine-tune. Nothing here is one-shot.

---

## Directory layout

```
panel-rtmdet-finetune/
├── configs/
│   ├── rtmdet_tiny_panel.py      # MMDetection config, edit num_classes/classes here
│   └── classes.yaml              # single source of truth for class names
├── data/
│   └── raw_coco/                 # put your COCO JSON + images here
├── tools/
│   ├── setup_env.sh              # one-time environment setup
│   ├── split_coco.py             # train/val split, or leave-one-out for tiny sets
│   ├── train.sh                  # fine-tuning entry point
│   ├── export_onnx.py            # checkpoint -> ONNX, static 640x640
│   └── verify_onnx.py            # sanity-check exported model, CPU provider
├── checkpoints/                  # training outputs land here (gitignored-style, empty for now)
├── exports/                      # ONNX exports land here
└── logs/                         # training + export logs
```

---

## One-time setup

```bash
cd panel-rtmdet-finetune
bash tools/setup_env.sh
```

This creates a conda/venv environment, installs PyTorch (CUDA build — edit the
script if your local GPU needs a different CUDA version), MMDetection, MMCV,
and ONNX export dependencies. It also downloads the official COCO-pretrained
RTMDet-tiny checkpoint as the fine-tuning starting point — this is the same
architecture panel-bench already measured (222.8 ms p50 / 36.3 MB ΔPSS, CPU
provider, on a Snapdragon 778G), so cost numbers from that table are a
reasonable prior for what this produces, not a guarantee.

---

## Step 1 — put data in place

```
data/raw_coco/
├── images/
│   ├── panel_001.jpg
│   └── ...
└── annotations.json      # single COCO JSON covering all images
```

Edit `configs/classes.yaml` with your actual class names (the annotation
handoff doc mentions a two-class scheme — put those two names here, not
placeholders, before training).

---

## Step 2 — split the data

```bash
python tools/split_coco.py --mode auto
```

`split_coco.py` picks the right strategy for you:

- **≥ ~20 images**: standard 80/20 train/val split.
- **< ~20 images** (your current 5–8 sample case): defaults to **leave-one-out**
  mode instead of a train/val split — it will *not* silently let you train and
  test on the same images. If you explicitly want a train=test plumbing run
  (verifying the export/inference pipeline only, not accuracy), pass
  `--mode plumbing-only`, which requires you to also pass
  `--i-understand-this-is-not-an-accuracy-test`. The script prints that label
  into every downstream log and into the exported model's metadata, so this
  can't quietly get mistaken for a real evaluation three steps downstream.

---

## Step 3 — fine-tune

```bash
bash tools/train.sh --config configs/rtmdet_tiny_panel.py
```

Logs go to `logs/`, checkpoints to `checkpoints/`. The config starts from the
COCO-pretrained weights and fine-tunes the detection head for your classes at
640x640, matching the input size already benchmarked on-device.

---

## Step 4 — export to ONNX

```bash
python tools/export_onnx.py --checkpoint checkpoints/latest.pth --out exports/
```

Produces a static-shape (640x640) multi-box ONNX export. The output filename
is deliberately checked against the known `freeDimsFor` matching bug in
`MainActivity.kt` (any filename containing "rec" — including inside words like
`direct` or `corrected` — silently gets assigned the `[48,320]` recognizer
shape instead of `[640,640]`). The script will refuse to write a filename
containing "rec" and tell you why.

---

## Step 5 — verify before it goes anywhere near a device

```bash
python tools/verify_onnx.py --model exports/rtmdet_tiny_panel.onnx
```

Runs the export with `onnxruntime`, **CPU provider only** — per panel-bench,
CPU beat XNNPACK and NNAPI on every model tested, so there's no reason to
evaluate other providers here. Confirms:

- output tensor shape is multi-box (not the single image-sized COCO box)
- a dummy 640x640 input runs without shape errors
- reports desktop latency as a sanity check only — **this is not a device
  number**; re-measure on-device (778G bench, then the 845 floor) before
  drawing any latency conclusion.

---

## Carrying provenance forward

Every run directory under `logs/` is stamped with: which config, which
checkpoint/hash, which dataset split mode, git commit if available, and
timestamp. Given how many times a number in this project has turned out to be
right-shaped but wrong-caused, don't let a checkpoint or export leave this
folder without that stamp attached — `tools/train.sh` and
`tools/export_onnx.py` write it automatically to a `run_manifest.json` next to
their output; don't delete it.

## Known open items this pipeline does not resolve

- The ~100 px/module threshold is still a 4-panel estimate, unvalidated at
  scale — this pipeline can produce more annotated data to fix that, it
  doesn't fix it by existing.
- Accuracy on 5–8 images (whether split LOO or plumbing-only) says nothing
  about real-world detection quality. Wait for a larger annotated set before
  trusting any accuracy number this produces.
- On-device timing/memory still needs the 845-class floor device, not just the
  778G panel-bench numbers this config's export is designed to be comparable
  against.
