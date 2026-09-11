# Exporting a fine-tuned RTMDet-tiny checkpoint

This is the process for turning a trained `.pth` checkpoint into an ONNX
file the Android app (and `harness/onnx_backends.py`) can actually consume.
Read this before touching the export step again — most of what's here was
found the hard way, and skipping it means re-discovering the same bugs.

---

## Why this isn't a standard MMDetection export

MMDetection 2.x shipped `tools/deployment/pytorch2onnx.py`. **MMDetection
3.x removed it.** The replacement, MMDeploy, bakes NMS and box-decoding
*into* the ONNX graph.

That's the wrong format for this project. `harness/onnx_backends.py`'s
`RTMDetOnnx` class expects a **raw head export** — six outputs (3
classification maps + 3 box-regression maps, one pair per FPN level, no
sigmoid applied, no NMS in the graph) — and decodes them in Python, so the
exact same decode logic can be mirrored on Android
(`RtmdetDecode.kt`). Using MMDeploy's default export here would produce a
file `RTMDetOnnx` cannot correctly read.

`tools/export_onnx.py` does the raw export directly: it loads the model via
`mmdet.apis.init_detector`, wraps `extract_feat` + `bbox_head` in a small
`torch.nn.Module` (bypassing the full `predict()` pipeline, which is where
MMDeploy's decode/NMS would otherwise get baked in), and exports that with
plain `torch.onnx.export`.

---

## Running the export

```bash
conda activate panel-rtmdet
cd panel-rtmdet-finetune

python tools/export_onnx.py \
    --checkpoint checkpoints/best_coco_bbox_mAP_epoch_40.pth \
    --config configs/rtmdet_tiny_panel.py \
    --out exports/ \
    --plumbing-only        # only if the checkpoint came from a small-holdout
                            # or plumbing-only split (see split_coco.py) --
                            # stamps the manifest so this can't later be
                            # mistaken for an accuracy-validated model
```

**Output:**
- `exports/rtmdet_tiny_panel.onnx` — the model
- `exports/rtmdet_tiny_panel.manifest.json` — checkpoint used, class list,
  the class-index mapping, output names/order, and the `plumbing_only` flag
  if set

**Filename safety**: the script refuses to write a filename containing
`"rec"` anywhere (case-insensitive) — `MainActivity.kt`'s `freeDimsFor()`
matches that substring and silently assigns the `[48,320]` OCR-recognizer
input shape instead of `[640,640]`. This has bitten the project once
already; don't rename the export to something like `direct_export.onnx` or
`corrected_v2.onnx`.

---

## Output format, exactly

```
outs[0] cls_score, level 0 (stride 8),  shape (1, num_classes, 80, 80)
outs[1] cls_score, level 1 (stride 16), shape (1, num_classes, 40, 40)
outs[2] cls_score, level 2 (stride 32), shape (1, num_classes, 20, 20)
outs[3] bbox_pred, level 0 (stride 8),  shape (1, 4, 80, 80)
outs[4] bbox_pred, level 1 (stride 16), shape (1, 4, 40, 40)
outs[5] bbox_pred, level 2 (stride 32), shape (1, 4, 20, 20)
```

`RTMDetOnnx.detect()` (Python) and `RtmdetDecode.detect()` (Kotlin) both
assume this exact order via `n = len(outs) // 2; cls_maps, box_maps =
outs[:n], outs[n:]`. If a future mmdetection version reorders its own
output tuple, this will fail loudly (shape mismatch) rather than silently
pairing the wrong maps — don't "fix" that by reordering blindly; check
`splitOutputs()`'s shape-based pairing logic in `RtmdetDecode.kt` first,
which derives pairing from tensor shape rather than trusting position.

---

## The two decode bugs already found and fixed here — do not reintroduce

**1. Double-stride multiplication.** RTMDet's `forward()`
(`mmdet/models/dense_heads/rtmdet_head.py`) already multiplies `bbox_pred`
by stride internally, before returning it. An earlier version of
`harness/onnx_backends.py`'s decode step multiplied by stride a *second*
time, inflating distances up to 32x and saturating every box to the image
edge. **Fixed** — the decode step only multiplies the grid-cell centers
(`cx`, `cy`) by stride, not the box distances (`dl/dt/dr/db`). Kotlin's
`RtmdetDecode.kt` was already correct on this (independently verified
against the source).

**2. Per-class vs. global NMS.** Boxes of different classes at the same
physical location (e.g. a device the model is genuinely uncertain is RCBO
vs. RCCB) must be suppressed against each other, not kept separately.
**Fixed** in both Python (`RTMDetOnnx._nms`) and Kotlin (`RtmdetDecode`'s
`nms()`, previously `nmsPerClass()`) — global NMS, threshold 0.55.

If you ever rewrite either decode path, re-verify against a known image
(see below) before trusting the output — both of these bugs produced
plausible-*looking* results, not obvious crashes.

---

## Verification — do this every time, not just once

Shape correctness does not mean decode correctness. Two checks, in order:

**1. Confirm mean/std match the checkpoint.**
```bash
python -c "
from mmdet.apis import init_detector
m = init_detector('configs/rtmdet_tiny_panel.py', 'checkpoints/best_coco_bbox_mAP_epoch_40.pth', device='cpu')
print(m.data_preprocessor.mean)
print(m.data_preprocessor.std)
"
```
Compare against `RTMDetOnnx.DEFAULT_MEAN` / `DEFAULT_STD` in
`harness/onnx_backends.py`. If this config never touched
`data_preprocessor`, they'll match the RTMDet COCO defaults — but confirm
it, don't assume it. If they don't match, pass `mean=`/`std=` explicitly
when constructing `RTMDetOnnx`.

**2. Run it against a real image with known ground truth.**
```bash
# from the panel-harness conda env (needs onnxruntime + cv2, NOT mmdet)
python3 -c "
from harness.onnx_backends import RTMDetOnnx
import cv2

det = RTMDetOnnx(
    'exports/rtmdet_tiny_panel.onnx',
    class_map={0: 'MCB', 1: 'RCBO', 2: 'RCCB'},
    bgr=False,   # cv2.imread() gives BGR; bgr=False tells the class not to
                 # flip it. bgr=True (the default) expects RGB input and
                 # converts internally -- feeding it cv2's native BGR output
                 # with the default would double-flip the channels.
    conf=0.70,   # validated threshold -- see below
)
img = cv2.imread('panel_images/11_1.jpeg')   # ground truth: 7 devices
results = det.detect(img)
print(f'{len(results)} devices found')
for d in results:
    print(f'  {d.label:6s} score={d.score:.3f}  box={d.box}')
"
```
`11_1.jpeg` has 7 known devices (6 MCB + 1 RCBO). Expect a result close to
that count with sensible box positions. If you get wildly wrong box
sizes, near-zero or hundreds of detections, or every box saturated to the
image edge — stop and re-check the two bugs above before assuming the
model itself is bad.

**Confidence threshold: use `conf=0.70`, not the class default of 0.30.**
Confirmed via IoU-matched debugging on `11_1`: real detections scored
0.758–0.931, spurious ones (no ground-truth match) scored 0.333–0.677 —
clean separation, no overlap. The class's own default (`conf=0.30`) lets
spurious detections through. This has not yet been changed in
`harness/onnx_backends.py`'s own default — pass `conf=0.70` explicitly at
every call site until that default is updated upstream.

---

## Known limitation: class imbalance, not a decode bug

If you see a device confidently labeled the wrong class (RCBO called RCCB
or vice versa) with a *correct* box position, that's a real, separate,
already-documented finding — RCCB/RCBO are the minority classes in the
current 15-image training set, and the model genuinely confuses them on a
handful of real panels (`11_1`, `3_3`). More training examples of those two
classes fixes this; re-checking the decode math will not, since box
geometry is already confirmed correct in these cases.

---

## Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| `expected 3 class outputs (C=N), got X and Y` | Re-export — the model's class count doesn't match what `RTMDetOnnx`/`RtmdetDecode` expected. Check `classes.yaml` matches what the checkpoint was actually trained with. |
| Every box saturated near the image edge | Double-stride bug reintroduced — check the decode step isn't multiplying `dl/dt/dr/db` by stride a second time. |
| Device count roughly right but scores look compressed (0.5–0.7 range only) | Sigmoid applied twice, or not at all — check the raw head export isn't already applying sigmoid before this decode step does it again. |
| Reasonable boxes, wrong classes throughout (not just RCBO/RCCB) | Channel indexing likely transposed — check `cls[c * plane + idx]` indexing, not `idx * C + c`. |
| `ONNX file not found` at inference time but filename looks right | Check the export didn't get silently refused for containing "rec" in the name — check the script's stderr output at export time. |
