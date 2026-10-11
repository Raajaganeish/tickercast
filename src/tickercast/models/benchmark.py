"""Measure real PatchTST training speed on this machine for several model sizes.

For each size: run a few warm-up steps, then time real training steps (forward + backward
+ AdamW) on real batches, and estimate how long one full pass over the training data takes.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from tickercast.config import Config
from tickercast.dataset.windows import WindowDataset
from tickercast.models.patchtst import PatchTST, PatchTSTConfig, best_device, count_parameters

SIZES = {
    "small":  dict(d_model=64,  n_layers=2, n_heads=4, d_ff=128),
    "medium": dict(d_model=128, n_layers=4, n_heads=4, d_ff=256),
    "large":  dict(d_model=192, n_layers=4, n_heads=6, d_ff=512),
    "xlarge": dict(d_model=256, n_layers=6, n_heads=8, d_ff=1024),
}


def run_benchmark(cfg: Config, batch_size: int = 512, steps: int = 30, warmup: int = 5) -> pd.DataFrame:
    device = best_device()
    ds = WindowDataset(cfg.path("processed"), split="train")
    meta = ds.meta
    rng = np.random.default_rng(0)
    print(f"benchmark: device={device}, train windows={len(ds):,}, batch={batch_size}")

    rows = []
    for name, size in SIZES.items():
        mcfg = PatchTSTConfig(n_features=len(ds.features), n_tickers=len(meta["tickers"]),
                              n_sectors=len(meta["sectors"]), window=ds.window, **size)
        model = PatchTST(mcfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def step():
            x, y, t, s = ds.batch(rng.integers(0, len(ds), batch_size))
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            t, s = torch.from_numpy(t).to(device), torch.from_numpy(s).to(device)
            loss = loss_fn(model(x, t, s), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return loss

        for _ in range(warmup):
            step()
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.time()
        for _ in range(steps):
            loss = step()
        loss.item()  # wait for the GPU to finish
        sec_per_step = (time.time() - start) / steps
        epoch_min = len(ds) / batch_size * sec_per_step / 60
        rows.append({"size": name, **size, "params": count_parameters(model),
                     "samples_per_sec": batch_size / sec_per_step, "minutes_per_epoch": epoch_min,
                     "epochs_in_50_min": 50 / epoch_min})
        print(f"  {name:7s} {rows[-1]['params']:>10,} params  {rows[-1]['samples_per_sec']:>8,.0f} samples/s  "
              f"{epoch_min:5.1f} min/epoch")
        del model, opt
        if device.type == "mps":
            torch.mps.empty_cache()
    return pd.DataFrame(rows).set_index("size")
