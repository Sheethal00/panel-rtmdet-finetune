# RTMDet-tiny fine-tuning config for distribution panel devices.
#
# Extends the official COCO-pretrained rtmdet_tiny config. Only the head
# (num_classes), dataset paths, and a few schedule params are overridden --
# everything else inherits MMDetection's tuned RTMDet-tiny defaults.
#
# DELIBERATELY hardcoded, not read dynamically from classes.yaml.
#
# An earlier version read classes.yaml via `import yaml` + `open()` at the
# top of this file. That works fine on its own, but mmengine's config
# parser treats any config file containing non-trivial top-level Python
# (imports, function calls) as a "lazy-import" config, which then REJECTS
# the classic `_base_ = [...]` string-list syntax below with:
#   ConfigParsingError: Only `read_base` context manager can be used
# Fixing that the "correct" way means either wrapping _base_ in mmengine's
# `read_base()` context manager (fragile with mmdet's file-path-style base
# configs) or, simpler and more robust: keep this file boring. Plain
# literals only. If you change classes.yaml, update CLASSES below to match
# -- yes, this is a small DRY sacrifice, but it avoids an entire category
# of mmengine parsing errors that cost real debugging time to find once.
#
# Currently matches configs/classes.yaml exactly: MCB, RCBO, RCCB (3
# classes, confirmed against the real 15-image Roboflow annotation:
# 109 MCB, 16 RCCB, 3 RCBO). The original handoff doc specifies a two-class
# scheme that was never reconciled against this -- flagged, not resolved,
# proceeding with 3 as the safer/reversible default.
CLASSES = ("MCB", "RCBO", "RCCB")
NUM_CLASSES = 3
INPUT_SIZE = (640, 640)  # matches panel-bench's measured RTMDet-tiny shape

# _base_ paths are resolved relative to THIS FILE'S OWN DIRECTORY
# (configs/), not the project root and not the CWD train.sh runs from.
# The original version of this path ("mmdetection/configs/rtmdet/...")
# was wrong for exactly this reason -- it would have resolved to
# configs/mmdetection/configs/rtmdet/..., which doesn't exist. Corrected
# to go up one level first.
_base_ = [
    "../mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
]

# --- data ---
data_root = "data/raw_coco/"
metainfo = dict(classes=CLASSES)

train_dataloader = dict(
    batch_size=4,  # small dataset -- keep batch size modest, adjust once you
                   # have more than a few dozen images
    num_workers=4,  # base config inherits a worker count sized for
                    # COCO-scale training; capped here after the first real
                    # run warned it was spinning up 10 workers against a
                    # system-suggested max of 4 -- not fatal, but worth
                    # avoiding rather than ignoring on a resource-constrained
                    # machine, especially given this is already CPU-only.
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
    # for the new num_classes -- tools/setup_env.sh downloads this checkpoint.
    # This path IS resolved relative to CWD at runtime (train.sh runs from
    # the project root), unlike _base_ above which is parse-time and
    # relative to this file -- two different resolution rules for two
    # different mmengine mechanisms, worth remembering if either path
    # needs touching again.
    init_cfg=dict(
        type="Pretrained",
        checkpoint="checkpoints/rtmdet_tiny_8xb32-300e_coco_pretrained.pth",
    ),
)

# --- input resolution ---
# RTMDet-tiny's stock base config already trains at 640x640 by default --
# the same resolution panel-bench measured on-device -- so no override is
# needed here. An earlier version of this file tried to override
# train_pipeline_stage2 with dict(scale=INPUT_SIZE), but the base config
# defines train_pipeline_stage2 as a LIST of transform steps, not a dict,
# and mmengine's config merger refuses to merge a dict into a list without
# an explicit _delete_=True. Rather than guess at replicating the base
# pipeline's exact list structure just to set a value it already has,
# removing the override is the correct fix: INPUT_SIZE above exists purely
# as documentation now, confirming intent matches the inherited default,
# not as something that needs to be actively applied.

# --- schedule: short fine-tune, small dataset ---
max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

optim_wrapper = dict(
    optimizer=dict(lr=0.001),  # lower than the from-scratch COCO LR;
                                # this is a fine-tune, not a fresh train
)

# --- learning rate schedule, recalibrated for this dataset's iteration count ---
# CONFIRMED BUG (from the first real training run): loss stayed flat at
# ~2.45-2.47 for 11 straight epochs with val mAP = 0.000 at both checkpoints.
# Cause: the base RTMDet config's warmup schedule assumes COCO-scale training
# (thousands of iterations per epoch) and typically warms up over ~1000
# iterations. This dataset has 12 training images at batch_size=4, i.e. only
# 3 iterations per epoch -- so the inherited warmup would take 300+ epochs to
# finish ramping up to the real LR. optim_wrapper.optimizer.lr=0.001 sets the
# PEAK learning rate after warmup, but if warmup itself never finishes, the
# effective LR stays near zero indefinitely regardless of that setting.
# Overriding the full schedule here (same list type as the base config, so
# mmengine replaces it outright rather than attempting a merge) with a
# warmup short enough to actually complete within this run, followed by
# cosine decay over the remaining epochs.
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0e-3,
        by_epoch=False,
        begin=0,
        end=10,  # 10 iterations, not ~1000 -- roughly 3 epochs at 3 iters/epoch
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=0.001 * 0.05,
        begin=3,
        T_max=max_epochs - 3,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# --- logging / checkpointing ---
default_hooks = dict(
    checkpoint=dict(interval=5, save_best="auto"),
    logger=dict(interval=1),
)

work_dir = "checkpoints/"
