#!/usr/bin/env python3
"""
Export a fine-tuned RTMDet-tiny checkpoint to a static-shape ONNX model.

Refuses to write any output filename containing the substring "rec"
(case-insensitive), because MainActivity.kt's freeDimsFor() matches any
filename containing "rec" ANYWHERE -- including inside words like "direct"
or "corrected" -- and silently assigns it the [48,320] recognizer input
shape instead of [640,640]. This has already bitten this project once with
RapidOCR silently ignoring an argument; don't repeat the pattern with a
filename.

Usage:
    python tools/export_onnx.py --checkpoint checkpoints/best.pth \
        --config configs/rtmdet_tiny_panel.py --out exports/
"""
import argparse
import json
import os
import subprocess
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
        "--plumbing-only",
        action="store_true",
        help="Set if the source checkpoint was trained with "
             "split_coco.py --mode plumbing-only. Stamps the export "
             "metadata so this can't be mistaken for an accuracy-validated "
             "model downstream.",
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

    print(f"== Exporting {args.checkpoint} -> {out_path} at static {input_size} ==")

    # Uses MMDetection's deployment export path (mmdeploy), assumed installed
    # by setup_env.sh. Adjust the mmdeploy config reference if your local
    # setup uses a different deploy config name.
    cmd = [
        "python", "mmdetection/tools/deployment/pytorch2onnx.py",
        args.config,
        args.checkpoint,
        "--output-file", out_path,
        "--shape", str(input_size[0]), str(input_size[1]),
        "--dynamic-export", "False",  # static shape, matches panel-bench protocol
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Export failed.", file=sys.stderr)
        sys.exit(result.returncode)

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "input_size": input_size,
        "classes": class_cfg["classes"],
        "output_file": out_path,
        "plumbing_only": args.plumbing_only,
    }
    if args.plumbing_only:
        manifest["warning"] = (
            "Source checkpoint trained with train==test data. This export's "
            "detection output has NOT been accuracy-validated. Use only for "
            "pipeline/shape/latency/memory verification."
        )

    manifest_path = os.path.join(args.out, f"{args.name}.manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"== Export complete. Manifest written to {manifest_path} ==")
    if args.plumbing_only:
        print("REMINDER: this is a plumbing-only export. No accuracy claim.")


if __name__ == "__main__":
    main()
