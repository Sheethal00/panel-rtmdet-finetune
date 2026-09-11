#!/usr/bin/env python3
"""
Split a COCO JSON annotation file into train/val sets.

Three modes, each answering a different question -- pick based on what you
actually need right now, not by default:

    --mode auto            80/20 split if >= 20 images, otherwise refuses and
                            tells you to pick one of the modes below.
    --mode small-holdout    A single train/val split for datasets too small for
                            --mode auto's threshold (e.g. 15 images). Produces
                            ONE deployable checkpoint, unlike --mode loo. The
                            val metric from a split this small is NOT a
                            trustworthy accuracy number -- treat it as a sanity
                            check (is loss/mAP moving in the right direction at
                            all) rather than a real evaluation. Use --val-count
                            to control how many images are held out (default 3
                            of 15 -- enough to catch a completely broken
                            training run, not enough to trust the resulting
                            number as "the" accuracy).
    --mode loo              Leave-one-out: writes N train/val pairs, one per
                            held-out image. Gives the best available accuracy
                            SIGNAL on a tiny dataset, but produces N separate
                            models, not one checkpoint to deploy. Use this when
                            the question is "roughly how good is this
                            approach", not "give me a model to export."
    --mode plumbing-only    train==test, on purpose, for pipeline verification
                            ONLY. Requires the extra confirmation flag below
                            and stamps every output file with a warning label.

Usage:
    python tools/split_coco.py --mode auto
    python tools/split_coco.py --mode small-holdout --val-count 3
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
        choices=["auto", "small-holdout", "loo", "plumbing-only"],
        default="auto",
    )
    parser.add_argument(
        "--val-count", type=int, default=3,
        help="Number of images to hold out for validation in --mode small-holdout",
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
                f"80/20 split.\n"
                f"Pick one explicitly:\n"
                f"  --mode small-holdout   one real checkpoint, val metric is "
                f"a sanity check only (not a trustworthy accuracy number)\n"
                f"  --mode loo             best available accuracy SIGNAL, "
                f"but {n} separate models, no single deployable checkpoint\n"
                f"  --mode plumbing-only   train==test, pipeline verification "
                f"only, no accuracy claim",
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

    elif args.mode == "small-holdout":
        if args.val_count >= n:
            print(f"REFUSING: --val-count {args.val_count} >= {n} total images "
                  f"would leave nothing to train on.", file=sys.stderr)
            sys.exit(1)
        shuffled = image_ids[:]
        random.shuffle(shuffled)
        val_ids = shuffled[:args.val_count]
        train_ids = shuffled[args.val_count:]
        with open(os.path.join(args.out, "train.json"), "w") as f:
            json.dump(subset_coco(coco, train_ids), f)
        with open(os.path.join(args.out, "val.json"), "w") as f:
            json.dump(subset_coco(coco, val_ids), f)
        write_manifest(
            args.out, "small-holdout",
            {
                "n_images": n, "n_train": len(train_ids), "n_val": len(val_ids),
                "warning": f"val set has only {len(val_ids)} image(s). The "
                           f"resulting val mAP/loss is a directional sanity "
                           f"check -- confirms training is doing SOMETHING "
                           f"reasonable -- not a statistically meaningful "
                           f"accuracy estimate. Do not quote this number as "
                           f"'the' model accuracy.",
            },
        )
        print(f"Wrote train ({len(train_ids)}) / val ({len(val_ids)}) small-holdout "
              f"split to {args.out}")
        print(f"REMINDER: with only {len(val_ids)} val image(s), treat the "
              f"resulting metric as a sanity check, not an accuracy claim.")

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
             "note": "each fold's val.json is a single held-out image never seen in that fold's training. "
                     "This produces N models, not one -- use small-holdout instead if you need a single "
                     "checkpoint to export."},
        )
        print(f"Wrote {n} leave-one-out folds to {fold_dir}")
        print("Train and evaluate each fold separately; do not average silently "
              "without reporting n -- with this few images per-fold variance will be large.")

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