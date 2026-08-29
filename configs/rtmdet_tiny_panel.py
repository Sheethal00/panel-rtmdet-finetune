# RTMDet-tiny fine-tuning config for distribution panel devices.
#
# Extends the official COCO-pretrained rtmdet_tiny config. Only the head
# (num_classes), dataset paths, and a few schedule params are overridden —
# everything else inherits MMDetection's tuned RTMDet-tiny defaults.
#
# Before running: edit configs/classes.yaml, not this file, to set your
# actual class names. This file reads that yaml at build time.

import yaml
import os

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "classes.yaml")) as f:
    _cfg = yaml.safe_load(f)

CLASSES = tuple(_cfg["classes"])
NUM_CLASSES = len(CLASSES)
INPUT_SIZE = tuple(_cfg["input_size"])  # (H, W) e.g. (640, 640)

_base_ = [
    "mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
]

# --- data ---
data_root = "data/raw_coco/"
metainfo = dict(classes=CLASSES)

train_dataloader = dict(
    batch_size=4,  # small dataset — keep batch size modest, adjust once you
                   # have more than a few dozen images
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="splits/train.json",
        data_prefix=dict(img="images/"),
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="splits/val.json",
        data_prefix=dict(img="images/"),
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + "splits/val.json",
)
test_evaluator = val_evaluator

# --- model head ---
model = dict(
    bbox_head=dict(num_classes=NUM_CLASSES),
    # start from COCO-pretrained weights; only the head is reinitialized
    # for the new num_classes — tools/setup_env.sh downloads this checkpoint
    init_cfg=dict(
        type="Pretrained",
        checkpoint="checkpoints/rtmdet_tiny_8xb32-300e_coco_pretrained.pth",
    ),
)

# --- input size, matched to what panel-bench already measured on-device ---
train_pipeline_stage2 = dict(scale=INPUT_SIZE)

# --- schedule: short fine-tune, small dataset ---
max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

optim_wrapper = dict(
    optimizer=dict(lr=0.001),  # lower than the from-scratch COCO LR;
                                # this is a fine-tune, not a fresh train
)

# --- logging / checkpointing ---
default_hooks = dict(
    checkpoint=dict(interval=5, save_best="auto"),
    logger=dict(interval=1),
)

work_dir = "checkpoints/"
