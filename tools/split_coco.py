#!/usr/bin/env python3
"""
Split a COCO JSON annotation file into train/val sets.

Deliberately does NOT default to train=test on small datasets. With very few
annotated images (this project currently has 5-8), a standard split leaves
almost nothing to validate on, and training==testing measures memorization,
not accuracy. This script makes you choose explicitly:

    --mode auto           80/20 split if enough images, otherwise refuses
                           and tells you to pick --mode loo or --mode plumbing-only
    --mode loo            leave-one-out: writes N train/val pairs, one per
                           held-out image. Use this for any accuracy signal
                           on a tiny dataset.
    --mode plumbing-only  train==test, on purpose, for pipeline verification
                           ONLY. Requires the extra confirmation flag below
                           and stamps every output file with a warning label.

Usage:
    python tools/split_coco.py --mode auto
    python tools/split_coco.py --mode loo
    python tools/split_coco.py --mode plumbing-only --i-understand-this-is-not-an-accuracy-test
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

SMALL_DATASET_THRESHOLD = 20  # below this, auto mode refuses a normal split


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def images_by_id(coco):
    return {img["id"]: img for img in coco["images"]}


def anns_by_image(coco):
    out = {}
    for ann in coco["annotations"]:
        out.setdefault(ann["image_id"], []).append(ann)
    return out


def subset_coco(coco, image_ids):
    img_ids = set(image_ids)
    images = [img for img in coco["images"] if img["id"] in img_ids]
    annotations = [ann for ann in coco["annotations"] if ann["image_id"] in img_ids]
    return {
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def write_manifest(out_dir, mode, extra):
    manifest = {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco",
        default="data/raw_coco/annotations.json",
        help="Path to the source COCO JSON file",
    )
    parser.add_argument(
        "--out",
        default="data/raw_coco/splits",
        help="Output directory for split JSON files",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "loo", "plumbing-only"],
        default="auto",
    )
    parser.add_argument(
        "--i-understand-this-is-not-an-accuracy-test",
        action="store_true",
        dest="confirm_plumbing",
        help="Required alongside --mode plumbing-only",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coco = load_coco(args.coco)
    image_ids = [img["id"] for img in coco["images"]]
    n = len(image_ids)
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    if args.mode == "auto":
        if n < SMALL_DATASET_THRESHOLD:
            print(
                f"REFUSING: only {n} images found, below the "
                f"{SMALL_DATASET_THRESHOLD}-image threshold for a normal "
                f"80/20 split. A split this small either leaves almost "
                f"nothing to validate on, or invites training on everything.\n"
                f"Use --mode loo (leave-one-out, gives an accuracy signal on "
                f"unseen images) or --mode plumbing-only (train==test, "
                f"pipeline verification only, no accuracy claim).",
                file=sys.stderr,
            )
            sys.exit(1)
        shuffled = image_ids[:]
        random.shuffle(shuffled)
        split_at = int(0.8 * n)
        train_ids, val_ids = shuffled[:split_at], shuffled[split_at:]
        with open(os.path.join(args.out, "train.json"), "w") as f:
            json.dump(subset_coco(coco, train_ids), f)
        with open(os.path.join(args.out, "val.json"), "w") as f:
            json.dump(subset_coco(coco, val_ids), f)
        write_manifest(
            args.out, "auto-80-20",
            {"n_images": n, "n_train": len(train_ids), "n_val": len(val_ids)},
        )
        print(f"Wrote train ({len(train_ids)}) / val ({len(val_ids)}) split to {args.out}")

    elif args.mode == "loo":
        fold_dir = os.path.join(args.out, "loo_folds")
        os.makedirs(fold_dir, exist_ok=True)
        for i, held_out in enumerate(image_ids):
            train_ids = [iid for iid in image_ids if iid != held_out]
            fold_path = os.path.join(fold_dir, f"fold_{i:02d}")
            os.makedirs(fold_path, exist_ok=True)
            with open(os.path.join(fold_path, "train.json"), "w") as f:
                json.dump(subset_coco(coco, train_ids), f)
            with open(os.path.join(fold_path, "val.json"), "w") as f:
                json.dump(subset_coco(coco, [held_out]), f)
        write_manifest(
            args.out, "leave-one-out",
            {"n_images": n, "n_folds": n,
             "note": "each fold's val.json is a single held-out image never seen in that fold's training"},
        )
        print(f"Wrote {n} leave-one-out folds to {fold_dir}")
        print("Train and evaluate each fold separately; do not average silently "
              "without reporting n — with this few images per-fold variance will be large.")

    elif args.mode == "plumbing-only":
        if not args.confirm_plumbing:
            print(
                "REFUSING: --mode plumbing-only requires "
                "--i-understand-this-is-not-an-accuracy-test.\n"
                "This mode trains and evaluates on the SAME images. It verifies "
                "that the pipeline runs end-to-end (export shape, on-device "
                "inference, gate logic consumes real output). It proves NOTHING "
                "about detection accuracy. Any number this produces must be "
                "labeled 'plumbing check, no accuracy claim' wherever it is used.",
                file=sys.stderr,
            )
            sys.exit(1)
        with open(os.path.join(args.out, "train.json"), "w") as f:
            json.dump(subset_coco(coco, image_ids), f)
        with open(os.path.join(args.out, "val.json"), "w") as f:
            json.dump(subset_coco(coco, image_ids), f)
        write_manifest(
            args.out, "PLUMBING-ONLY-TRAIN-EQUALS-TEST",
            {
                "n_images": n,
                "warning": "train.json and val.json are IDENTICAL. This run "
                           "measures pipeline correctness only. No accuracy "
                           "claim can be made from any metric this produces.",
            },
        )
        print("Wrote train==val plumbing-only split.")
        print("REMINDER: label every downstream number from this run as a "
              "plumbing check, not an accuracy result.")


if __name__ == "__main__":
    main()
