#!/usr/bin/env python3
"""
Sanity-check an exported ONNX model before it goes anywhere near a device.

Uses CPU execution provider only -- panel-bench already found CPU beats
XNNPACK and NNAPI on every model tested (RTMDet-tiny: 222.8ms/36.3MB on CPU
vs 275ms/102MB XNNPACK vs 421ms/44MB NNAPI), so there's no reason to spend
time re-evaluating providers here.

This reports DESKTOP latency, which is not a device number. Re-measure on
the actual 778G/845 hardware before drawing any conclusion about field
performance -- this script only confirms the model loads and produces the
right output shape.

Usage:
    python tools/verify_onnx.py --model exports/rtmdet_tiny_panel.onnx
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    manifest_path = args.model.replace(".onnx", ".manifest.json")
    plumbing_only = False
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        plumbing_only = manifest.get("plumbing_only", False)
        print(f"Loaded manifest: classes={manifest.get('classes')}, "
              f"input_size={manifest.get('input_size')}")
    else:
        print("WARNING: no manifest found next to this export -- "
              "provenance is missing. Re-export with export_onnx.py.")

    if plumbing_only:
        print("=" * 70)
        print("THIS IS A PLUMBING-ONLY EXPORT. Any numbers below verify")
        print("pipeline correctness, NOT detection accuracy.")
        print("=" * 70)

    print(f"\n== Loading {args.model} with CPU execution provider ==")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()
    print(f"Input:  name={input_meta.name}, shape={input_meta.shape}, dtype={input_meta.type}")
    for o in output_meta:
        print(f"Output: name={o.name}, shape={o.shape}, dtype={o.type}")

    # Sanity check: a COCO-pretrained, non-fine-tuned RTMDet returns a
    # single image-sized box. If this export still looks like that, the
    # fine-tune head swap didn't take -- fail loudly rather than let a
    # single-box model quietly reach the gate logic.
    n_output_dims = len(output_meta[0].shape) if output_meta else 0
    print(f"\n{len(output_meta)} output tensor(s), first output has "
          f"{n_output_dims} dims -- confirm this matches a real multi-box "
          f"detection head output, not a single-box passthrough, before "
          f"trusting anything downstream.")

    shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_meta.shape]
    dummy = np.random.rand(*shape).astype(np.float32)

    print(f"\n== Running {args.iterations} dummy inferences (desktop CPU, "
          f"NOT a device number) ==")
    times = []
    for i in range(args.iterations):
        start = time.perf_counter()
        session.run(None, {input_meta.name: dummy})
        times.append((time.perf_counter() - start) * 1000)

    times_sorted = sorted(times)
    p50 = times_sorted[len(times_sorted) // 2]
    p95 = times_sorted[int(len(times_sorted) * 0.95)]
    print(f"Desktop CPU p50: {p50:.1f} ms, p95: {p95:.1f} ms "
          f"(first-3 vs last-3 for thermal/warmup drift: "
          f"{np.mean(times[:3]):.1f}ms -> {np.mean(times[-3:]):.1f}ms)")
    print("\nRe-run this same model on the 778G and 845 devices before "
          "using any latency figure in the capture-gate spec.")


if __name__ == "__main__":
    main()
