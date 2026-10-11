"""Which stocks we train on: current S&P 500 members with their GICS sector.

Source: the Wikipedia table "List of S&P 500 companies".
Known limitation: these are *today's* members (survivorship bias). See docs/02_universe.md.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from tickercast.config import Config

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def to_yahoo_symbol(symbol: str) -> str:
    # Wikipedia writes share classes with a dot (BRK.B); Yahoo uses a dash (BRK-B).
    return symbol.strip().replace(".", "-")


def fetch_sp500() -> pd.DataFrame:
    html = requests.get(WIKI_URL, headers={"User-Agent": "tickercast/0.1 (research)"}, timeout=30).text
    table = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
    out = pd.DataFrame(
        {
            "ticker": table["Symbol"].map(to_yahoo_symbol),
            "name": table["Security"],
            "sector": table["GICS Sector"],
            "sub_industry": table["GICS Sub-Industry"],
            "date_added": pd.to_datetime(table["Date added"], errors="coerce"),
        }
    )
    return out.sort_values("ticker").reset_index(drop=True)


def build_universe(cfg: Config) -> pd.DataFrame:
    uni = fetch_sp500()
    extra = [t for t in cfg["universe"]["extra_tickers"] if t not in set(uni["ticker"])]
    if extra:
        uni = pd.concat([uni, pd.DataFrame({"ticker": extra, "sector": "Unknown"})], ignore_index=True)
    out = cfg.path("raw") / "universe.parquet"
    uni.to_parquet(out, index=False)
    print(f"universe: {len(uni)} tickers, {uni['sector'].nunique()} sectors -> {out}")
    return uni
