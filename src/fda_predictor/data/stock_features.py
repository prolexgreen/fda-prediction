"""Leakage-safe pre-event sponsor stock features via yfinance.

CTO's post-completion slope is NEVER used as input — only ticker linkage
and completion/start dates from CTO/CTGov. All features are computed from
daily bars strictly BEFORE the trial completion date (bet-time proxy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fda_predictor.utils.paths import STOCK_CACHE_DIR, ensure_dirs

STOCK_FEATURE_NAMES = (
    "ret_30d",
    "ret_60d",
    "ret_90d",
    "vol_30d_ann",
    "max_drawdown_90d",
    "rel_volume_trend_30d",
    "momentum_30_90",
)

N_STOCK_FEATURES = len(STOCK_FEATURE_NAMES)


@dataclass
class StockFeatureStats:
    mean: np.ndarray = field(default_factory=lambda: np.zeros(N_STOCK_FEATURES))
    std: np.ndarray = field(default_factory=lambda: np.ones(N_STOCK_FEATURES))

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StockFeatureStats":
        return cls(mean=np.array(d["mean"], dtype=np.float64), std=np.array(d["std"], dtype=np.float64))


def _parse_date(value) -> datetime | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%B %Y", "%Y"):
        try:
            return datetime.strptime(s[: len(fmt.replace("%B", "January"))], fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, errors="coerce").to_pydatetime()
    except Exception:
        return None


def _cache_path(ticker: str) -> Path:
    safe = str(ticker).upper().replace("/", "_")
    return STOCK_CACHE_DIR / f"{safe}.json"


def fetch_price_history(
    ticker: str,
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Daily OHLCV sliced to [start, end]; full history cached per ticker.

    The cache stores the ticker's ENTIRE available history (period=max).
    Multiple trials share a ticker with different event dates; caching a
    single request window would feed later trials the first trial's window
    -- possibly post-event data (leakage). Slice locally instead.
    """
    ensure_dirs()
    STOCK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(ticker)
    hist: pd.DataFrame | None = None
    loaded_from_cache = False

    if use_cache and cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            df = pd.DataFrame(payload["rows"])
            if not df.empty:
                df["Date"] = pd.to_datetime(df["Date"])
                hist = df.set_index("Date").sort_index()
                loaded_from_cache = True
        except (json.JSONDecodeError, KeyError, ValueError):
            hist = None  # corrupt cache -> refetch below

    if hist is None:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("install yfinance: uv sync") from exc

        tkr = yf.Ticker(ticker)
        fetched = tkr.history(period="max", auto_adjust=True)
        if fetched is None or fetched.empty:
            # definitive miss (delisted/unknown symbol): safe to persist
            cache.write_text(json.dumps({"ticker": ticker, "rows": []}), encoding="utf-8")
            return None
        fetched = fetched.reset_index()
        fetched["Date"] = fetched["Date"].dt.tz_localize(None)
        rows = fetched[["Date", "Close", "Volume"]].to_dict(orient="records")
        for r in rows:
            r["Date"] = r["Date"].isoformat()
        cache.write_text(json.dumps({"ticker": ticker, "rows": rows}), encoding="utf-8")
        hist = fetched.set_index("Date").sort_index()[["Close", "Volume"]]

    sub = hist.loc[(hist.index >= pd.Timestamp(start)) & (hist.index <= pd.Timestamp(end))]
    if sub.empty and loaded_from_cache:
        # Cached history doesn't cover this window (legacy narrow caches):
        # refresh once from the source before giving up.
        return fetch_price_history(ticker, start, end, use_cache=False)
    if sub.empty:
        return None
    return sub


def _window_return(closes: pd.Series, end_idx: int, days: int) -> float:
    if end_idx < days or len(closes) <= days:
        return 0.0
    start_p = float(closes.iloc[end_idx - days])
    end_p = float(closes.iloc[end_idx])
    if start_p <= 0:
        return 0.0
    return (end_p / start_p) - 1.0


def _ann_vol(closes: pd.Series, end_idx: int, days: int = 30) -> float:
    if end_idx < days + 1:
        return 0.0
    window = closes.iloc[end_idx - days : end_idx + 1]
    rets = window.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    return float(rets.std() * np.sqrt(252))


def _max_drawdown(closes: pd.Series, end_idx: int, days: int = 90) -> float:
    start = max(0, end_idx - days)
    window = closes.iloc[start : end_idx + 1]
    if len(window) < 2:
        return 0.0
    peak = window.cummax()
    dd = (window / peak) - 1.0
    return float(dd.min())


def _rel_volume_trend(volumes: pd.Series, end_idx: int, days: int = 30) -> float:
    if end_idx < days * 2:
        return 0.0
    recent = volumes.iloc[end_idx - days + 1 : end_idx + 1].mean()
    prior = volumes.iloc[end_idx - 2 * days + 1 : end_idx - days + 1].mean()
    if prior <= 0:
        return 0.0
    return float(recent / prior - 1.0)


def compute_pre_event_features(
    ticker: str | None,
    completion_date,
    start_date=None,
    lookback_days: int = 180,
) -> tuple[np.ndarray, int]:
    """Return (feature_vector, mask). mask=0 when data unavailable."""
    zeros = np.zeros(N_STOCK_FEATURES, dtype=np.float32)
    if not isinstance(ticker, str):
        return zeros, 0  # None / float('nan') after concat with CTO ticker column
    ticker = ticker.strip()
    if not ticker or ticker.upper() in ("NAN", "NONE", "NULL"):
        return zeros, 0

    event = _parse_date(completion_date)
    if event is None:
        return zeros, 0

    # Features must end strictly BEFORE completion (no post-event leakage).
    end = event - timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    hist = fetch_price_history(ticker, start, end)
    if hist is None or len(hist) < 35:
        return zeros, 0

    closes = hist["Close"]
    volumes = hist["Volume"]
    end_idx = len(closes) - 1

    r30 = _window_return(closes, end_idx, 30)
    r60 = _window_return(closes, end_idx, 60)
    r90 = _window_return(closes, end_idx, 90)
    feats = np.array(
        [
            r30,
            r60,
            r90,
            _ann_vol(closes, end_idx, 30),
            _max_drawdown(closes, end_idx, 90),
            _rel_volume_trend(volumes, end_idx, 30),
            r30 - r90,  # momentum_30_90
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(feats)):
        return zeros, 0
    return feats, 1


def fit_stock_normalizer(train_frame: pd.DataFrame) -> StockFeatureStats:
    """Z-score stats from TRAIN split only (masked rows)."""
    rows = []
    for _, row in train_frame.iterrows():
        if int(row.get("stock_mask", 0)) != 1:
            continue
        vec = row.get("stock_feats")
        if vec is not None:
            rows.append(np.asarray(vec, dtype=np.float64))
    if not rows:
        return StockFeatureStats()
    mat = np.stack(rows)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std[std < 1e-6] = 1.0
    return StockFeatureStats(mean=mean, std=std)


def normalize_stock_features(feats: np.ndarray, stats: StockFeatureStats) -> np.ndarray:
    return ((feats - stats.mean) / stats.std).astype(np.float32)


def attach_stock_features(
    frame: pd.DataFrame,
    stats: StockFeatureStats | None = None,
    fit_stats: bool = False,
) -> pd.DataFrame:
    """Add stock_feats (list) and stock_mask columns to a trial frame."""
    out = frame.copy()
    feats_col = []
    mask_col = []
    for _, row in out.iterrows():
        vec, mask = compute_pre_event_features(
            row.get("ticker"),
            row.get("completion_date") or row.get("chronology_date"),
            row.get("start_date"),
        )
        feats_col.append(vec.tolist())
        mask_col.append(mask)
    out["stock_feats"] = feats_col
    out["stock_mask"] = mask_col
    if fit_stats:
        stats = fit_stock_normalizer(out)
    if stats is not None:
        normed = []
        for vec, mask in zip(feats_col, mask_col):
            arr = np.array(vec, dtype=np.float32)
            if mask:
                arr = normalize_stock_features(arr, stats)
            normed.append(arr.tolist())
        out["stock_feats"] = normed
    return out
