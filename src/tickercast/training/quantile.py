"""Turning 3 raw model outputs into an ordered range, and the loss that trains it.

Ordered range:
    raw = (a, b, c) from the network
    mid  = a
    low  = mid - softplus(b)      softplus(x) = log(1 + e^x) is always > 0
    high = mid + softplus(c)
so low < mid < high always holds. Without this, the network could output a "low" above
its "high", which makes no sense.

Pinball (quantile) loss, for a guess p of the q-th percentile and a real value y:
    if y > p:  loss = q * (y - p)          (guessed too low)
    else:      loss = (1 - q) * (p - y)    (guessed too high)
For q = 0.9 ("high"), guessing too low costs 9x more than guessing too high, so the model
learns to put "high" where only ~10% of real moves land above it. For q = 0.5 it is the
plain absolute error, which gives the median.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

QUANTILES = (0.1, 0.5, 0.9)


def to_range(raw: torch.Tensor) -> torch.Tensor:
    """(batch, 3) raw outputs -> (batch, 3) ordered [low, mid, high]."""
    mid = raw[:, 0]
    low = mid - F.softplus(raw[:, 1])
    high = mid + F.softplus(raw[:, 2])
    return torch.stack([low, mid, high], dim=1)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles=QUANTILES) -> torch.Tensor:
    """pred: (batch, 3) ordered range; y: (batch,) real moves. Average over the 3 quantiles."""
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    diff = y[:, None] - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()
