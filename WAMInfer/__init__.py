"""Parallel-only external acceleration for upstream OpenWAM."""

from .runtime import OpenWAM, Runtime as accelerate

__all__ = ["OpenWAM", "accelerate"]
