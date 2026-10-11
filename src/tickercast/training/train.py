"""Train PatchTST to forecast tomorrow's range: low / mid / high of the next-day change.

The recipe, for each model size:

    1. Every epoch, shuffle the 1.46M training windows and go through them in batches of 512.
    2. For each batch: the model predicts (low, mid, high) for each window, we measure the
       pinball loss against what really happened, and AdamW nudges the weights to reduce it.
    3. After each epoch, score the validation years (2020-2021), which the model never trains on.
    4. Keep the weights from the epoch with the best validation loss. Stop early when
       validation stops improving for `patience` epochs (the model has started memorizing).

Then pick the size with the best validation loss. The test years (2022+) are not used here;
see training/evaluate.py.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch

from tickercast.config import Config
from tickercast.dataset.windows import WindowDataset
from tickercast.evaluation.range_metrics import range_score
from tickercast.models.benchmark import SIZES
from tickercast.models.patchtst import PatchTST, PatchTSTConfig, best_device, count_parameters
from tickercast.training.quantile import pinball_loss, to_range

TARGET = "fwd_ret_1d"
TARGET_SCALE = 100.0   # train in percent: 0.012 -> 1.2, so numbers are a comfortable size for the network


@dataclass
class TrainConfig:
    sizes: list[str] = field(default_factory=lambda: ["small", "medium", "large", "xlarge"])
    features: str = "all"          # "all" (66) or "stock_only" (24)
    max_epochs: int = 10
    patience: int = 2              # epochs without validation improvement before stopping
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    dropout: float = 0.2
    grad_clip: float = 1.0
    seed: int = 0
    target_mode: str = "vol_scaled"  # "raw": predict % directly; "vol_scaled": predict in units of the
                                     # stock's 20-day volatility, then multiply back (adapts to calm/wild periods)


def run_name(tcfg: "TrainConfig") -> str:
    return f"{tcfg.features}_{tcfg.target_mode}"


def feature_list(cfg: Config, which: str) -> list[str] | None:
    if which == "all":
        return None
    path = cfg.path("raw").parent / "results" / "stock_specific_features.json"
    return json.loads(path.read_text())


def target_unit(ds: WindowDataset, idx: np.ndarray, mode: str) -> np.ndarray:
    """What one unit of model output means, per window: 1% ("raw") or 1 x the stock's 20-day volatility."""
    if mode == "raw":
        return np.full(len(idx), 1.0 / TARGET_SCALE, dtype=np.float32)
    return ds.vol[ds.ends[idx]]


def to_device(batch, unit, device):
    x, y, t, s = batch
    return (torch.from_numpy(x).to(device), torch.from_numpy(y / unit).to(device),
            torch.from_numpy(t).to(device), torch.from_numpy(s).to(device))


@torch.no_grad()
def predict(model, ds: WindowDataset, device, mode: str = "raw", batch_size: int = 4096) -> np.ndarray:
    """(n_windows, 3) array of [low, mid, high] in percent."""
    model.eval()
    out = []
    for i in range(0, len(ds), batch_size):
        idx = np.arange(i, min(i + batch_size, len(ds)))
        unit = target_unit(ds, idx, mode)
        x, _, t, s = to_device(ds.batch(idx), unit, device)
        pred = to_range(model(x, t, s)).float().cpu().numpy()
        out.append(pred * unit[:, None] * TARGET_SCALE)
    return np.concatenate(out)


def train_one_size(size: str, tcfg: TrainConfig, train_ds, val_ds, device, ckpt_dir) -> tuple[list[dict], dict]:
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    meta = train_ds.meta
    mcfg = PatchTSTConfig(n_features=len(train_ds.features), n_tickers=len(meta["tickers"]),
                          n_sectors=len(meta["sectors"]), window=train_ds.window, n_outputs=3,
                          dropout=tcfg.dropout, **SIZES[size])
    model = PatchTST(mcfg).to(device)
    n_params = count_parameters(model)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    steps_per_epoch = len(train_ds) // tcfg.batch_size
    total = steps_per_epoch * tcfg.max_epochs

    def lr_at(step):  # linear warm-up, then cosine decay to 5% of the peak
        if step < tcfg.warmup_steps:
            return (step + 1) / tcfg.warmup_steps
        progress = (step - tcfg.warmup_steps) / max(1, total - tcfg.warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    y_val = val_ds.y[val_ds.ends] * TARGET_SCALE
    val_dates = val_ds.dates[val_ds.ends]

    history, best, bad_epochs = [], None, 0
    print(f"\n[{size}] {n_params:,} parameters, {steps_per_epoch:,} steps per epoch, target {tcfg.target_mode}")
    for epoch in range(1, tcfg.max_epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        order = rng.permutation(len(train_ds))
        for step in range(steps_per_epoch):
            idx = order[step * tcfg.batch_size:(step + 1) * tcfg.batch_size]
            x, y, t, s = to_device(train_ds.batch(idx), target_unit(train_ds, idx, tcfg.target_mode), device)
            loss = pinball_loss(to_range(model(x, t, s)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            sched.step()
            running += loss.item() if step % 50 == 0 else 0.0   # sample the loss (syncing every step is slow)
        train_loss = running / len(range(0, steps_per_epoch, 50))

        pred = predict(model, val_ds, device, tcfg.target_mode)
        sc = range_score(pred[:, 0], pred[:, 1], pred[:, 2], y_val, val_dates)
        row = {"size": size, "params": n_params, "epoch": epoch, "train_loss": train_loss,
               "val_loss": sc["pinball_loss"], **{k: v for k, v in sc.items() if k not in ("rows", "pinball_loss")},
               "lr": sched.get_last_lr()[0], "minutes": (time.time() - start) / 60}
        history.append(row)
        improved = best is None or row["val_loss"] < best["val_loss"] - 1e-5
        print(f"  epoch {epoch:2d}  train {train_loss:.4f}  val {row['val_loss']:.4f}  coverage {row['coverage']:.3f}  "
              f"width~move {row['width_vs_move_corr']:.3f}  mid IC {row.get('mid_daily_ic', float('nan')):+.4f}  "
              f"{row['minutes']:.1f} min {'*' if improved else ''}")
        if improved:
            best, bad_epochs = row, 0
            torch.save({"model_config": mcfg.to_dict(), "train_config": asdict(tcfg), "features": train_ds.features,
                        "target": TARGET, "target_scale": TARGET_SCALE, "state_dict": model.state_dict(),
                        "best": row}, ckpt_dir / f"patchtst_{run_name(tcfg)}_{size}.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg.patience:
                print(f"  early stop: no validation improvement for {tcfg.patience} epochs")
                break
    return history, best


def run_training(cfg: Config, tcfg: TrainConfig) -> pd.DataFrame:
    device = best_device()
    feats = feature_list(cfg, tcfg.features)
    processed = cfg.path("processed")
    train_ds = WindowDataset(processed, "train", target=TARGET, features=feats)
    val_ds = WindowDataset(processed, "val", target=TARGET, features=feats)
    ckpt_dir = cfg.path("raw").parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    out = cfg.path("raw").parent / "results"
    print(f"training on {device}: {len(train_ds):,} train / {len(val_ds):,} val windows, "
          f"{len(train_ds.features)} features, sizes {tcfg.sizes}")

    all_hist, bests = [], []
    t0 = time.time()
    for size in tcfg.sizes:
        hist, best = train_one_size(size, tcfg, train_ds, val_ds, device, ckpt_dir)
        all_hist += hist
        bests.append(best)
        pd.DataFrame(all_hist).to_csv(out / f"patchtst_history_{run_name(tcfg)}.csv", index=False)
        if device.type == "mps":
            torch.mps.empty_cache()
    summary = pd.DataFrame(bests).set_index("size")
    summary.to_csv(out / f"patchtst_best_{run_name(tcfg)}.csv")
    winner = summary["val_loss"].idxmin()
    (out / f"patchtst_winner_{run_name(tcfg)}.json").write_text(
        json.dumps({"size": winner, "features": tcfg.features, "target_mode": tcfg.target_mode}))
    print(f"\ntotal training time {(time.time() - t0) / 60:.1f} min. Best size on validation: {winner}")
    return summary
