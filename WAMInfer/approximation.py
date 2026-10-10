"""Optional observation FFN reuse and motion-adaptive residual refresh."""

import math
import numpy as np
import torch


class TokenReuse:
    def __init__(self, buckets):
        self.buckets = tuple(sorted(set(buckets)))
        if not self.buckets or self.buckets[0] < 1:
            raise ValueError("Refresh capacities must be positive integers")
        self.reset()

    def reset(self):
        self.key = self.image = self.features = self.cache = None

    def capacities(self, n):
        return tuple(b for b in self.buckets if b < n) + (n,)

    def begin(self, image, inputs, prompt, patch):
        latents = inputs["latents"]
        h, w = latents.shape[-2] // patch[1], latents.shape[-1] // patch[2]
        n = h * w
        rgb = np.array(image, copy=True)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or latents.shape[2] < 2:
            raise ValueError("Token reuse requires one HWC RGB observation and future video frames")
        key = (prompt, tuple(latents.shape), rgb.shape)
        if key != self.key:
            self.reset()
        mandatory = np.ones(n, dtype=bool)
        if self.image is not None:
            changes = np.subtract(rgb, self.image, dtype=np.float32)
            np.abs(changes, out=changes)
            changes = changes.reshape(h, rgb.shape[0] // h, w, rgb.shape[1] // w, 3)
            mandatory = changes.mean(axis=(1, 3, 4)).reshape(-1) > 5.0
        required = int(mandatory.sum())
        required += math.ceil(0.25 * (n - required))
        count = next(b for b in self.capacities(n) if b >= required)
        reuse = dict(
            n=n,
            count=count,
            mandatory=torch.as_tensor(mandatory, device=latents.device),
            features=self.features if count < n else None,
            cache=self.cache if count < n else None,
        )
        return reuse, (key, rgb)

    def commit(self, observation, features, cache):
        self.key, self.image = observation
        # Own the committed history: a later failed request must not overwrite it.
        self.features, self.cache = features.clone(), cache.clone()


def select_tokens(state, features):
    """Pure graph operation: history is committed once, after the request."""
    reuse = state.extras["token_reuse"]
    n, count = reuse["n"], reuse["count"]
    current = features[:, :n].contiguous()
    state.extras["token_features"] = current
    if count < n:
        now, before = current.float(), reuse["features"].float()
        score = 2 * (now - before).abs().sum(-1) / (now.abs() + before.abs()).sum(-1).clamp_min(1e-6)
        score = score.squeeze(0).masked_fill(reuse["mandatory"], float("inf"))
        selected = torch.argsort(score, descending=True, stable=True)[:count].sort().values
        future = torch.arange(n, features.shape[1], device=features.device)
        state.extras["token_indices"] = torch.cat((selected, future))


class MotionRefinement:
    def __init__(self, metric, threshold, schedule):
        if metric is None or len(schedule) != 11:
            raise ValueError("adaptive_2f4f requires 10 Euler updates and a motion_metric returning metres")
        self.metric, self.threshold = metric, threshold
        self.nfe, self.amplitude, self.residual = 2, None, None

    def full(self, index):
        return index in (0, 3) or (self.nfe == 4 and index in (6, 9))

    def decide(self, actions):
        self.amplitude = float(self.metric(actions))
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("motion_metric must return a finite nonnegative displacement in metres")
        self.nfe = 2 if self.amplitude > self.threshold else 4
