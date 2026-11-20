"""Window-of-frames slicing for the window-emission DMM variant."""

from __future__ import annotations

import torch


def build_windows(x: torch.Tensor, d: int, stride: int = 1) -> torch.Tensor:
    """Slice a [B, T, D] sequence into [B, T', d+1, D] sliding windows.

    Window k ends at raw frame `d + k*stride` and covers frames
    `(d + k*stride - d) .. (d + k*stride)` inclusive, i.e. `d+1` frames.
    """
    _, T, _ = x.shape
    windows = [x[:, t - d : t + 1, :] for t in range(d, T, stride)]
    return torch.stack(windows, dim=1)
