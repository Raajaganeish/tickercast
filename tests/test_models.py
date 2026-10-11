from __future__ import annotations

import numpy as np
import torch

from tickercast.evaluation.metrics import score
from tickercast.models.patchtst import PatchTST, PatchTSTConfig, count_parameters


def test_patchtst_forward_shape_and_backward():
    cfg = PatchTSTConfig(n_features=10, n_tickers=5, n_sectors=3, window=32, patch_len=8,
                         d_model=16, n_layers=2, n_heads=2, d_ff=32)
    model = PatchTST(cfg)
    x = torch.randn(4, 32, 10)
    out = model(x, torch.tensor([0, 1, 2, 4]), torch.tensor([0, 1, 2, 0]))
    assert out.shape == (4,)
    out.sum().backward()
    assert count_parameters(model) > 0


def test_score_perfect_and_random():
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(50), 20)
    ret = rng.normal(0, 0.01, len(dates))
    label = (ret > 0).astype(float)
    perfect = score(1 / (1 + np.exp(-ret * 1000)), label, ret, dates)
    assert perfect["accuracy"] > 0.99 and perfect["auc"] > 0.99 and perfect["daily_ic"] > 0.99
    noise = score(rng.random(len(dates)), label, ret, dates)
    assert 0.4 < noise["auc"] < 0.6 and abs(noise["daily_ic"]) < 0.1
