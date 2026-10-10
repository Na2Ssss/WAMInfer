"""Time the complete image/state-to-action path after warmup."""

import argparse
import json
import importlib
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from WAMInfer import OpenWAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="NPZ containing image (HWC uint8) and proprio")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--reuse-tokens", action="store_true")
    parser.add_argument("--adaptive-2f4f", action="store_true")
    parser.add_argument(
        "--motion-metric", help="module:function accepting physical actions and raw proprio; returns metres"
    )
    args = parser.parse_args()
    if args.adaptive_2f4f and not args.motion_metric:
        parser.error("--adaptive-2f4f requires --motion-metric module:function")
    metric = None
    if args.motion_metric:
        module, name = args.motion_metric.split(":")
        metric = getattr(importlib.import_module(module), name)
    torch.set_num_threads(4)
    with np.load(args.input, allow_pickle=False) as request:
        pixels, proprio = request["image"], request["proprio"]
    image = Image.fromarray(pixels)
    prompt = Path(args.prompt_file).read_text().strip()
    model = OpenWAM(args.checkpoint, reuse_tokens=args.reuse_tokens, adaptive_2f4f=args.adaptive_2f4f)

    def run():
        return model.generate(
            prompt,
            image,
            proprio=proprio,
            height=pixels.shape[0],
            width=pixels.shape[1],
            motion_metric=None if metric is None else lambda actions: metric(actions, proprio),
        )

    try:
        for _ in range(args.warmup):
            run()
        times = []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = run()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        print(
            json.dumps(
                dict(
                    mode="parallel",
                    reuse_tokens=args.reuse_tokens,
                    adaptive_2f4f=args.adaptive_2f4f,
                    **model.architecture.last_stats,
                    gpu=torch.cuda.get_device_name(),
                    mean_ms=float(np.mean(times)),
                    median_ms=float(np.median(times)),
                    p95_ms=float(np.percentile(times, 95)),
                    runs=args.runs,
                    action_shape=list(result["actions"].shape),
                    protocol="10 Euler updates, repeated identical observation, CPU image/state to CPU actions, no video decode",
                ),
                indent=2,
            )
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
