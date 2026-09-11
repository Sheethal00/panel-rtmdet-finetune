#!/usr/bin/env python3
"""
Export a fine-tuned RTMDet-tiny checkpoint to a static-shape ONNX model,
producing the RAW HEAD output format harness/backends.py's RTMDetOnnx class
already expects and decodes -- not a decoded/NMS'd export.

WHY RAW, NOT DECODED: MMDetection 3.x removed its bundled ONNX export script
(tools/deployment/pytorch2onnx.py, which an earlier version of this file
called). The replacement, MMDeploy, bakes decode+NMS INTO the ONNX graph --
which is the OPPOSITE of what this project's own harness/backends.py
RTMDetOnnx class expects: six raw outputs (3 classification maps + 3 box
regression maps, at strides 8/16/32, no sigmoid, no NMS), decoded in Python
so the same decode logic can later be mirrored on Android. Using MMDeploy's
default detection export here would produce a file harness/backends.py
cannot correctly consume. This script exports the raw head instead, via a
plain torch.onnx.export() on the model's extract_feat + bbox_head forward,
bypassing the full predict() post-processing pipeline entirely.

Output order matches harness/backends.py's expectation EXACTLY:
    outs[0:3] = cls_score at strides 8, 16, 32
    outs[3:6] = bbox_pred at strides 8, 16, 32
(RTMDetOnnx does `n = len(outs) // 2; cls_maps, box_maps = outs[:n], outs[n:]`)

Refuses to write any output filename containing the substring "rec"
(case-insensitive), because MainActivity.kt's freeDimsFor() matches any
filename containing "rec" ANYWHERE -- including inside words like "direct"
or "corrected" -- and silently assigns it the [48,320] recognizer input
shape instead of [640,640].

Usage:
    python tools/export_onnx.py --checkpoint checkpoints/best.pth \
        --config configs/rtmdet_tiny_panel.py --out exports/
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import yaml

FORBIDDEN_SUBSTRING = "rec"


def check_filename_safety(filename: str):
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    if FORBIDDEN_SUBSTRING in stem:
        print(
            f"REFUSING to write '{filename}': filename contains 'rec', which "
            f"MainActivity.kt's freeDimsFor() matches anywhere in the string "
            f"to assign the [48,320] recognizer shape. This export is "
            f"[640,640]. Rename to avoid 'rec' as a substring (also avoid "
            f"words like 'direct', 'corrected' for the same reason) and "
            f"re-run.",
            file=sys.stderr,
        )
        sys.exit(1)


class RawHeadWrapper:
    """Built lazily inside main(), after torch/mmdet are confirmed importable
    -- keeps this script's --help usable even before the training env is set
    up, and gives a clear error if run outside the panel-rtmdet environment."""
    pass


def build_wrapper(model):
    import torch

    class _Wrapper(torch.nn.Module):
        """Wraps extract_feat + bbox_head, bypassing predict()'s decode+NMS.

        Returns a FLAT tuple (not nested lists) because ONNX export needs a
        flat output structure: cls_scores as three separate tensors, then
        bbox_preds as three separate tensors, in stride order (8, 16, 32) --
        matching harness/backends.py's RTMDetOnnx.detect() exactly, which
        slices outs[:3] as cls maps and outs[3:] as box maps.
        """

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            feats = self.model.extract_feat(x)
            cls_scores, bbox_preds = self.model.bbox_head(feats)
            # cls_scores / bbox_preds are each a list/tuple of 3 tensors,
            # one per FPN level, already in stride order per RTMDet's
            # standard head implementation. Flatten to one tuple.
            return (*cls_scores, *bbox_preds)

    return _Wrapper(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/rtmdet_tiny_panel.py")
    parser.add_argument("--out", default="exports/")
    parser.add_argument(
        "--name",
        default="rtmdet_tiny_panel",
        help="Output filename stem (without .onnx). Checked against the "
             "freeDimsFor 'rec' substring bug before writing.",
    )
    parser.add_argument(
        "--opset", type=int, default=11,
        help="ONNX opset version. 11 is a safe, broadly-supported default "
             "for onnxruntime CPU inference.",
    )
    parser.add_argument(
        "--plumbing-only",
        action="store_true",
        help="Set if the source checkpoint was trained with "
             "split_coco.py --mode plumbing-only or --mode small-holdout "
             "(a val set too small to trust as a real accuracy measure). "
             "Stamps the export metadata so this can't be mistaken for an "
             "accuracy-validated model downstream.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    check_filename_safety(args.name)
    out_path = os.path.join(args.out, f"{args.name}.onnx")

    with open(
        os.path.join(os.path.dirname(args.config), "classes.yaml")
    ) as f:
        class_cfg = yaml.safe_load(f)
    input_size = class_cfg["input_size"]
    classes = class_cfg["classes"]

    print(f"== Exporting {args.checkpoint} -> {out_path} at static {input_size} ==")
    print(f"== Classes (index order matters for harness/backends.py's "
          f"class_map): {list(enumerate(classes))} ==")

    try:
        import torch
        from mmdet.apis import init_detector
    except ImportError as e:
        print(f"ERROR: {e}. Run this from the panel-rtmdet conda "
              f"environment (conda activate panel-rtmdet).", file=sys.stderr)
        sys.exit(1)

    model = init_detector(args.config, args.checkpoint, device="cpu")
    model.eval()
    wrapper = build_wrapper(model)

    dummy = torch.randn(1, 3, input_size[0], input_size[1])

    with torch.no_grad():
        sample_outs = wrapper(dummy)
    n_levels = len(sample_outs) // 2
    print(f"== Traced output shapes ({len(sample_outs)} total, "
          f"{n_levels} cls + {n_levels} bbox) ==")
    for i, o in enumerate(sample_outs):
        kind = "cls_score" if i < n_levels else "bbox_pred"
        stride_idx = i if i < n_levels else i - n_levels
        print(f"   [{i}] {kind} (level {stride_idx}): shape {tuple(o.shape)}")

    output_names = (
        [f"cls_score_l{i}" for i in range(n_levels)]
        + [f"bbox_pred_l{i}" for i in range(n_levels)]
    )

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["input"],
        output_names=output_names,
        opset_version=args.opset,
        dynamic_axes=None,  # static shape, matching panel-bench's protocol
    )

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "input_size": input_size,
        "classes": classes,
        "class_map_for_harness_backends": {i: c for i, c in enumerate(classes)},
        "output_format": "RAW HEAD (no sigmoid, no NMS) -- 3 cls_score maps "
                          "then 3 bbox_pred maps, stride order (8, 16, 32). "
                          "Decode with harness.backends.RTMDetOnnx, NOT a "
                          "generic ONNX detection loader.",
        "output_names": output_names,
        "output_file": out_path,
        "plumbing_only": args.plumbing_only,
    }
    if args.plumbing_only:
        manifest["warning"] = (
            "Source checkpoint trained with a val set too small to trust "
            "(plumbing-only or small-holdout with very few val images). "
            "This export's detection output has NOT been accuracy-validated "
            "at any meaningful scale. Use only for pipeline/shape/latency/"
            "memory verification."
        )

    manifest_path = os.path.join(args.out, f"{args.name}.manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"== Export complete. Manifest written to {manifest_path} ==")
    print(f"\nIMPORTANT next step: this export has NOT been verified against "
          f"real geometry yet -- shape correctness alone doesn't confirm the "
          f"decode math (box scale, cls/bbox ordering) is right, and a wrong "
          f"scale here is SILENT (harness/backends.py's own docstring warns "
          f"of exactly this). Run harness.backends.RTMDetOnnx.detect() "
          f"against one of the already-annotated images (e.g. 11_1 or 12_1, "
          f"both with known ground truth: 7 devices each) and check the "
          f"returned box count and rough positions before trusting this "
          f"export for anything else. Also verify data_preprocessor.mean/"
          f"std in the checkpoint match RTMDetOnnx's DEFAULT_MEAN/STD -- "
          f"getting this wrong is silent too.")
    if args.plumbing_only:
        print("REMINDER: this is a plumbing-only / small-holdout export. "
              "No accuracy claim.")


if __name__ == "__main__":
    main()