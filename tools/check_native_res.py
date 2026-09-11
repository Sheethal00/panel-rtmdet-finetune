#!/usr/bin/env python3
"""
Compare each image file's actual native resolution against what's recorded
in the COCO annotation JSON. Catches a silent Roboflow-export resize before
it corrupts a native-resolution measurement (exactly the failure mode this
project just spent a whole thread untangling for a different reason).

Usage:
    python check_native_res.py --coco annotations_coco.json --images-dir ./images
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from PIL import Image


KNOWN_EXTS = ["jpeg", "jpg", "png", "bmp", "webp"]


def roboflow_original_name(roboflow_name: str) -> Optional[str]:
    """
    Reverse Roboflow's export renaming convention.

    Roboflow takes an original file like '1_2.jpeg', replaces the dot before
    the extension with an underscore, then appends '.rf.<hash>.<ext>':

        1_2.jpeg  ->  1_2_jpeg.rf.mvvM3g0sUmnOC7mj49Ss.jpeg

    Returns the reconstructed original name, or None if the pattern isn't
    recognized (e.g. the file wasn't renamed by Roboflow at all).
    """
    if ".rf." not in roboflow_name:
        return None
    prefix = roboflow_name.split(".rf.")[0]
    for ext in KNOWN_EXTS:
        suffix = f"_{ext}"
        if prefix.lower().endswith(suffix):
            stem = prefix[: -len(suffix)]
            return f"{stem}.{ext}"
    return None


def find_matching_file(images_dir: Path, roboflow_name: str) -> Optional[Path]:
    """Try exact match first, then the reversed Roboflow name, then a fuzzy
    glob on the leading token as a last resort."""
    exact = images_dir / roboflow_name
    if exact.exists():
        return exact

    reconstructed = roboflow_original_name(roboflow_name)
    if reconstructed:
        candidate = images_dir / reconstructed
        if candidate.exists():
            return candidate

    # Fuzzy fallback: match by the leading token before "_jpeg"/"_png"/etc.
    # or before ".rf." -- covers cases the exact reversal doesn't catch.
    prefix = roboflow_name.split(".rf.")[0]
    for ext in KNOWN_EXTS:
        prefix = prefix.replace(f"_{ext}", "")
    matches = list(images_dir.glob(f"{prefix}.*")) + list(images_dir.glob(f"{prefix}_*"))
    matches = [m for m in matches if m.suffix.lower().lstrip(".") in KNOWN_EXTS]
    if len(matches) == 1:
        return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True)
    parser.add_argument(
        "--images-dir", required=True,
        help="folder containing the actual image files (native camera "
             "resolution, not necessarily the Roboflow export)",
    )
    args = parser.parse_args()

    with open(args.coco) as f:
        coco = json.load(f)

    images_dir = Path(args.images_dir)
    print(f"{'roboflow file_name':42s} {'matched to':30s} {'json w,h':>12s} {'actual w,h':>12s}  match")
    print("-" * 110)

    n_mismatch = 0
    n_missing = 0
    for img in sorted(coco["images"], key=lambda x: x["file_name"]):
        json_w, json_h = img["width"], img["height"]
        path = find_matching_file(images_dir, img["file_name"])

        if path is None:
            print(f"{img['file_name']:42s} {'NOT FOUND':30s} {json_w:>5d}x{json_h:<6d}")
            n_missing += 1
            continue

        with Image.open(path) as im:
            actual_w, actual_h = im.size

        match = (actual_w, actual_h) == (json_w, json_h)
        status = "OK" if match else "MISMATCH"
        if not match:
            n_mismatch += 1

        print(f"{img['file_name']:42s} {path.name:30s} {json_w:>5d}x{json_h:<6d}  "
              f"{actual_w:>5d}x{actual_h:<6d}  {status}")

    print()
    if n_missing:
        print(f"{n_missing} file(s) not found in {images_dir} -- check the path "
              f"and that file_name in the JSON matches the actual filenames.")
    if n_mismatch:
        print(f"{n_mismatch} mismatch(es) found. Any mismatched file has been "
              f"resized somewhere between annotation and this folder -- do NOT "
              f"use it for a native-resolution measurement (e.g. mlkit_yield.py) "
              f"until you find the source of the resize, or you'll reintroduce "
              f"the exact resolution-mismatch problem already resolved in this "
              f"project for the RapidOCR/1600px case.")
    if not n_mismatch and not n_missing:
        print("All images match the JSON's recorded resolution exactly. "
              "Safe to proceed with native-resolution measurements.")


if __name__ == "__main__":
    main()