"""Time the complete image/state-to-action path after warmup."""

import argparse
import json
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
    args = parser.parse_args()
    torch.set_num_threads(4)
    with np.load(args.input, allow_pickle=False) as request:
        pixels, proprio = request["image"], request["proprio"]
    image = Image.fromarray(pixels)
    prompt = Path(args.prompt_file).read_text().strip()
    model = OpenWAM(args.checkpoint)

    def run():
        return model.generate(prompt, image, proprio=proprio, height=pixels.shape[0], width=pixels.shape[1])

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
                    gpu=torch.cuda.get_device_name(),
                    mean_ms=float(np.mean(times)),
                    median_ms=float(np.median(times)),
                    p95_ms=float(np.percentile(times, 95)),
                    runs=args.runs,
                    action_shape=list(result["actions"].shape),
                    protocol="10 full steps, CPU image/state to CPU actions, no video decode",
                ),
                indent=2,
            )
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
