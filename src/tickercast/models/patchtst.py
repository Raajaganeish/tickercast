"""PatchTST-style Transformer for next-day direction.

    x: (batch, 128 days, F features)
      -> cut into 16 patches of 8 days            (batch, 16, 8*F)
      -> linear projection to d_model             (batch, 16, d)
      + learned position embedding (which patch)
      + ticker embedding + sector embedding        (same vector added to every patch)
      -> Transformer encoder (pre-norm)            (batch, 16, d)
      -> flatten -> linear                         (batch,) logit for P(up tomorrow)

Note: the original PatchTST paper treats each feature as a separate series
("channel-independent"). With 66 related features we mix them inside each patch instead,
which is cheaper (16 tokens, not 16 x 66) and lets attention combine features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class PatchTSTConfig:
    n_features: int
    n_tickers: int
    n_sectors: int
    window: int = 128
    patch_len: int = 8
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.2
    use_ticker_embedding: bool = True

    @property
    def n_patches(self) -> int:
        assert self.window % self.patch_len == 0, "window must be a multiple of patch_len"
        return self.window // self.patch_len

    def to_dict(self) -> dict:
        return asdict(self)


class PatchTST(nn.Module):
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.patch_len * cfg.n_features, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.ticker_emb = nn.Embedding(cfg.n_tickers, cfg.d_model) if cfg.use_ticker_embedding else None
        self.sector_emb = nn.Embedding(cfg.n_sectors, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(cfg.dropout), nn.Linear(cfg.n_patches * cfg.d_model, 1))

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor, sector_id: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        patches = x.reshape(b, self.cfg.n_patches, self.cfg.patch_len * self.cfg.n_features)
        h = self.patch_embed(patches) + self.pos
        ident = self.sector_emb(sector_id)
        if self.ticker_emb is not None:
            ident = ident + self.ticker_emb(ticker_id)
        h = self.drop(h + ident[:, None, :])
        h = self.norm(self.encoder(h))
        return self.head(h).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
