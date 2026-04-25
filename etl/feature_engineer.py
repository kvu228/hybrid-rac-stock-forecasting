"""Feature engineering: normalization, windowing, and labeling for pattern embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

OHLCV_CHANNELS = ["open", "high", "low", "close", "volume"]
# Feature channels used by encoder input.
# 6 channels: OHLCV + close_ret (technical indicators removed — scale mismatch degraded accuracy)
FEATURE_CHANNELS = ["open", "high", "low", "close", "volume", "close_ret"]
WINDOW_SIZE = 30
LABEL_HORIZON = 5  # T+5 forward return for labeling


@dataclass(frozen=True)
class WindowRecord:
    """Metadata + normalized data for a single sliding window."""

    symbol: str
    window_start: datetime
    window_end: datetime
    label: int  # 0=Down, 1=Neutral, 2=Up
    future_return: float  # raw T+5 close-to-close %
    data: np.ndarray  # shape (WINDOW_SIZE, n_channels) — z-score normalized features


def forward_fill_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill gaps in per-symbol OHLCV so every trading day has a row.

    Uses a business-day calendar (Mon-Fri) to fill gaps.  Price columns are
    forward-filled; volume is filled with 0 (no trading on missing days).

    Expects *sorted* input with columns: time, symbol, open, high, low, close, volume.
    """
    if df.empty:
        return df

    frames: list[pd.DataFrame] = []
    for symbol, grp in df.groupby("symbol", sort=False):
        grp = grp.set_index("time").sort_index()
        bday_idx = pd.bdate_range(start=grp.index.min(), end=grp.index.max(), freq="B")
        grp = grp.reindex(bday_idx)
        grp["symbol"] = symbol
        for col in ["open", "high", "low", "close"]:
            grp[col] = grp[col].ffill()
        grp["volume"] = grp["volume"].fillna(0).astype("int64")
        grp = grp.dropna(subset=["close"])
        grp = grp.reset_index().rename(columns={"index": "time"})
        frames.append(grp)

    if not frames:
        return df.iloc[:0]

    return pd.concat(frames, ignore_index=True)


def relative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``close_ret`` column: session-over-session close-to-close % return.

    Computed per symbol.  First row of each symbol gets 0.0.
    """
    out = df.copy()
    out["close_ret"] = out.groupby("symbol")["close"].pct_change().fillna(0.0)
    return out


def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI-14, range [0, 100]. Normalised to [-1, 1] by (rsi/50 - 1)."""
    n = len(close)
    rsi = np.full(n, 50.0)  # default neutral
    if n < period + 1:
        return (rsi / 50.0) - 1.0
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Wilder's smoothed moving average (EMA with alpha=1/period)
    alpha = 1.0 / period
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = gain[1 : period + 1].mean()
    avg_loss[period] = loss[1 : period + 1].mean()
    for i in range(period + 1, n):
        avg_gain[i] = avg_gain[i - 1] * (1 - alpha) + gain[i] * alpha
        avg_loss[i] = avg_loss[i - 1] * (1 - alpha) + loss[i] * alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
        rsi = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))
    rsi[:period] = 50.0  # warm-up → neutral
    return (rsi / 50.0) - 1.0  # rescale to [-1, 1]


def _compute_bb_pct(close: np.ndarray, period: int = 20, n_std: float = 2.0) -> np.ndarray:
    """Bollinger Band %B: position of price within the band.

    %B = (price - lower) / (upper - lower), clipped to [-1, 2] then scaled to [-1, 1] via (bb - 0.5) * 2.
    0 = lower band, 0.5 = middle, 1 = upper band.
    """
    n = len(close)
    bb_pct = np.full(n, 0.5)
    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        ma = window.mean()
        std = window.std(ddof=0)
        if std < 1e-8:
            bb_pct[i] = 0.5
        else:
            lower = ma - n_std * std
            upper = ma + n_std * std
            bb_pct[i] = (close[i] - lower) / (upper - lower)
    return (np.clip(bb_pct, -1.0, 2.0) - 0.5) * 2.0  # rescale to [-3, 3]


def _compute_macd_signal(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD Signal line normalised by the first close price (so units are comparable across symbols)."""
    n = len(close)
    if n < slow:
        return np.zeros(n)

    def _ema(arr: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        out = np.zeros(len(arr))
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = arr[i] * alpha + out[i - 1] * (1 - alpha)
        return out

    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    sig_line = _ema(macd_line, signal)
    base = close[0] if close[0] != 0 else 1.0
    return sig_line / base * 100.0  # express as % of first close


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rsi_14, bb_pct, macd_signal per symbol and add as columns.

    All indicators are computed on the *raw* close price before any
    window-level normalisation so that they carry cross-window information.
    """
    frames: list[pd.DataFrame] = []
    for symbol, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("time").copy()
        c = grp["close"].values.astype(np.float64)
        grp["rsi_14"] = _compute_rsi(c)
        grp["bb_pct"] = _compute_bb_pct(c)
        grp["macd_signal"] = _compute_macd_signal(c)
        frames.append(grp)
    if not frames:
        return df
    return pd.concat(frames, ignore_index=True)


def zscore_normalize_window(window: np.ndarray, channel_names: list[str] | None = None) -> np.ndarray:
    """Normalize channels within a window.

    Price columns (OHLC) are converted to percentage changes relative to the
    first 'open' price in the window, scaled by 100. This preserves both shape
    and ABSOLUTE volatility (e.g. 2.0 = +2% move).
    Volume is z-scored. 'close_ret' is scaled by 100.

    Args:
        window: shape (WINDOW_SIZE, n_channels)
        channel_names: List of channel names.

    Returns:
        Normalized array of the same shape.
    """
    x = np.asarray(window, dtype=np.float64)
    out = np.zeros_like(x)

    if channel_names is None:
        channel_names = ["open", "high", "low", "close", "volume", "close_ret"]

    # Channels that are already pre-normalised to a bounded range — pass through as-is.
    _PASSTHROUGH = {"rsi_14", "bb_pct", "macd_signal"}
    price_indices = [i for i, c in enumerate(channel_names) if c in ["open", "high", "low", "close"]]

    for i, col_name in enumerate(channel_names):
        if col_name in ["open", "high", "low", "close"]:
            continue  # Handled below via price_indices
        elif col_name in _PASSTHROUGH:
            out[:, i] = x[:, i]  # already bounded, no further scaling needed
        elif col_name == "volume":
            v = x[:, i]
            v_mean = np.nanmean(v)
            v_std = np.nanstd(v)
            if v_std == 0 or ~np.isfinite(v_std):
                v_std = 1.0
            out[:, i] = (v - v_mean) / v_std
        elif col_name == "close_ret" or "ret" in col_name:
            out[:, i] = x[:, i] * 100.0
        else:
            v = x[:, i]
            v_mean = np.nanmean(v)
            v_std = np.nanstd(v)
            if v_std == 0 or ~np.isfinite(v_std):
                v_std = 1.0
            out[:, i] = (v - v_mean) / v_std

    if price_indices:
        if "open" in channel_names:
            base_price = x[0, channel_names.index("open")]
        else:
            base_price = x[0, price_indices[0]]

        if base_price == 0 or ~np.isfinite(base_price):
            base_price = 1.0

        price_data = x[:, price_indices]
        out[:, price_indices] = ((price_data - base_price) / base_price) * 100.0

    out[~np.isfinite(out)] = 0.0
    return out


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range (ATR) — measure of daily volatility.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    Smoothed with Wilder's EMA (alpha=1/period).
    Returns array of same length as input; first ``period`` values are NaN.
    """
    n = len(close)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    valid = np.where(~np.isnan(tr))[0]
    if valid.size < period:
        return atr
    atr[valid[period - 1]] = np.nanmean(tr[valid[:period]])
    alpha = 1.0 / period
    for i in range(valid[period - 1] + 1, n):
        if not np.isnan(atr[i - 1]):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


def generate_windows(
    df: pd.DataFrame,
    *,
    window_size: int = WINDOW_SIZE,
    horizon: int = LABEL_HORIZON,
    stride: int = 1,
    up_threshold: float = 0.02,
    down_threshold: float = -0.02,
    channels: list[str] | tuple[str, ...] = tuple(FEATURE_CHANNELS),
    use_atr_threshold: bool = False,
    atr_multiplier: float = 1.5,
    atr_period: int = 14,
    dead_zone_pct: float = 0.0,
) -> list[WindowRecord]:
    """Create labeled sliding windows from cleaned OHLCV data.

    For each symbol the function slides a ``window_size``-session window with
    the given ``stride`` and computes a ``horizon``-day forward return from the
    last close in the window.  Each window's feature matrix is z-score
    normalized independently.

    By default the feature matrix has 6 channels (OHLCV + close_ret). The
    ``close_ret`` channel is session-over-session percentage return; unlike
    the raw OHLCV it is already differenced, so z-scoring within the window
    preserves directional information (up-trend vs down-trend) and helps the
    encoder distinguish momentum.

    Pass ``channels=OHLCV_CHANNELS`` for the legacy 5-channel behavior.

    ATR-based dynamic threshold:
        When ``use_atr_threshold=True``, the up/down thresholds are computed
        per window as ``atr_multiplier × ATR(atr_period) / last_close``.  This
        accounts for per-symbol, per-period volatility — volatile stocks need a
        larger move to be labelled Up/Down than stable ones.

    Dead-zone filtering:
        When ``dead_zone_pct > 0``, windows whose intra-horizon returns fall
        within ``(threshold - dead_zone_pct * threshold)`` of the label boundary
        are **discarded** (not added to the dataset).  This removes borderline
        "borderline" windows whose label flips easily from noise, improving the
        signal quality of the remaining samples.

    Args:
        df: Cleaned OHLCV DataFrame (must contain time, symbol, OHLCV cols).
        window_size: Number of sessions per window.
        horizon: Number of sessions ahead for the label return.
        stride: Step size between consecutive windows.
        up_threshold: Forward return >= this → label 2 (Up). Used when use_atr_threshold=False.
        down_threshold: Forward return <= this → label 0 (Down). Used when use_atr_threshold=False.
        channels: Columns to include as feature channels.
        use_atr_threshold: If True, compute per-window thresholds from ATR.
        atr_multiplier: ATR multiplier for dynamic threshold (default 1.5).
        atr_period: Look-back period for ATR (default 14).
        dead_zone_pct: Fraction of threshold to use as dead zone (0 = disabled).

    Returns:
        List of ``WindowRecord`` with normalized data + label. ``data`` has
        shape ``(window_size, len(channels))``.
    """
    channel_list = list(channels)
    records: list[WindowRecord] = []

    # Indicator columns that may need to be computed on-the-fly per symbol
    _INDICATOR_COLS = {"rsi_14", "bb_pct", "macd_signal"}
    _needs_indicators = bool(_INDICATOR_COLS & set(channel_list))

    for symbol, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("time").reset_index(drop=True)
        # Compute close_ret on demand if requested and not already present.
        if "close_ret" in channel_list and "close_ret" not in grp.columns:
            cr = grp["close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
            grp = grp.assign(close_ret=cr)
        # Compute technical indicators on demand.
        if _needs_indicators:
            c = grp["close"].values.astype(np.float64)
            if "rsi_14" in channel_list and "rsi_14" not in grp.columns:
                grp = grp.assign(rsi_14=_compute_rsi(c))
            if "bb_pct" in channel_list and "bb_pct" not in grp.columns:
                grp = grp.assign(bb_pct=_compute_bb_pct(c))
            if "macd_signal" in channel_list and "macd_signal" not in grp.columns:
                grp = grp.assign(macd_signal=_compute_macd_signal(c))
        closes = grp["close"].values
        highs = grp["high"].values
        lows = grp["low"].values
        times = grp["time"].values
        feature_matrix = grp[channel_list].values  # (N, C)

        # Pre-compute ATR array for this symbol if using dynamic thresholds.
        atr_arr: np.ndarray | None = None
        if use_atr_threshold:
            atr_arr = _compute_atr(highs, lows, closes, period=atr_period)

        n = len(grp)
        max_start = n - window_size - horizon
        if max_start < 0:
            continue

        for i in range(0, max_start + 1, stride):
            w_end = i + window_size  # exclusive for slice, last index = w_end-1
            future_idx = w_end + horizon - 1

            last_close = closes[w_end - 1]
            if last_close == 0:
                continue

            # Resolve thresholds: ATR-based or fixed.
            if use_atr_threshold and atr_arr is not None:
                atr_val = atr_arr[w_end - 1]
                if np.isnan(atr_val) or atr_val <= 0 or last_close <= 0:
                    # Fall back to fixed threshold if ATR not yet warm.
                    eff_up = up_threshold
                    eff_down = down_threshold
                else:
                    eff_up = atr_multiplier * atr_val / last_close
                    eff_down = -eff_up
            else:
                eff_up = up_threshold
                eff_down = down_threshold

            # Triple-Barrier Labeling using effective thresholds.
            label = 1  # Default to Neutral (Time stop)
            for j in range(w_end, future_idx + 1):
                # Check pessimistic case first: assume low hits stop-loss before high hits take-profit in the same session
                low_ret = (lows[j] - last_close) / last_close
                high_ret = (highs[j] - last_close) / last_close

                if low_ret <= eff_down:
                    label = 0  # Down (Stop-loss hit)
                    break
                elif high_ret >= eff_up:
                    label = 2  # Up (Take-profit hit)
                    break

            # Dead-zone filtering: discard borderline samples.
            if dead_zone_pct > 0.0 and label != 1:
                # Compute actual extreme return during horizon.
                horizon_lows = lows[w_end: future_idx + 1]
                horizon_highs = highs[w_end: future_idx + 1]
                if label == 0:  # Down — check if low is close to threshold
                    extreme = (horizon_lows.min() - last_close) / last_close if len(horizon_lows) else eff_down
                    margin = abs(eff_down) * dead_zone_pct
                    if abs(extreme - eff_down) < margin:
                        continue  # borderline Down — skip
                elif label == 2:  # Up — check if high is close to threshold
                    extreme = (horizon_highs.max() - last_close) / last_close if len(horizon_highs) else eff_up
                    margin = eff_up * dead_zone_pct
                    if abs(extreme - eff_up) < margin:
                        continue  # borderline Up — skip

            # Keep future_return as the point-in-time return for logging/metadata
            future_return = (closes[future_idx] - last_close) / last_close

            raw_window = feature_matrix[i:w_end].astype(np.float64)
            normed = zscore_normalize_window(raw_window, channel_names=channel_list)

            records.append(
                WindowRecord(
                    symbol=str(symbol),
                    window_start=pd.Timestamp(times[i]).to_pydatetime(),
                    window_end=pd.Timestamp(times[w_end - 1]).to_pydatetime(),
                    label=label,
                    future_return=float(future_return),
                    data=normed,
                )
            )

    return records


def train_test_split_by_time(
    records: list[WindowRecord],
    train_ratio: float = 0.8,
) -> tuple[list[WindowRecord], list[WindowRecord]]:
    """Split window records chronologically (NOT random).

    Windows are sorted by ``window_end`` across all symbols, then the first
    ``train_ratio`` fraction goes to the train set and the rest to test.

    Returns:
        (train_records, test_records)
    """
    if not records:
        return [], []

    sorted_recs = sorted(records, key=lambda r: r.window_end)
    split_idx = int(len(sorted_recs) * train_ratio)
    return sorted_recs[:split_idx], sorted_recs[split_idx:]