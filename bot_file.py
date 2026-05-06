import os
import time
import hmac
import json
import hashlib
import logging
import requests
import pandas as pd
import numpy as np

from math import sqrt
from urllib.parse import urlencode
from typing import Optional, Dict, List, Any, Tuple


try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ============================================================
# 1. KONFIGURATION - BVH x GETABOT COMPETITION
# ============================================================

SYMBOLS = ["BTCEUR", "ETHEUR", "DOGEEUR", "XRPEUR"]

INTERVAL = "1h"
LIMIT = 50000

INITIAL_CAPITAL = 10_000.0

# Regelwerk: keine simulierten Transaktionskosten
FEE_RATE = 0.0
SLIPPAGE = 0.0

# GetaBot Expert-Modus
COMPETITION_MODE = False
GETABOT_WEBHOOK_URL = ""  # später: https://getabot.eu/webhook/{team_id}

# Regelwerk: Signal spätestens 90 Sekunden nach Stundenschluss
SIGNAL_DELAY_SECONDS_AFTER_HOUR = 10
MAX_SIGNAL_SECONDS_AFTER_HOUR = 90
LIVE_POLL_SECONDS = 5

STATE_FILE = "getabot_live_state.json"

# Nur Kaufsignale / Spot-Logik
ALLOW_SHORTS = False

# Risiko
RISK_PER_TRADE = 0.0075
MAX_TOTAL_OPEN_RISK = 0.025
RRR = 2.0

MAX_TRADES_PER_DAY = 8
MAX_OPEN_POSITIONS = 4
MAX_DAILY_LOSS_PCT = 0.05

ENTRY_DELAY_CANDLES = 1

SWING_LOOKBACK = 2
FVG_MAX_AGE = 300

ATR_PERIOD = 14
MIN_IMPULSE_ATR_MULT = 0.6
OB_SEARCH_BACK = 15
OB_MAX_AGE = 300
OB_PROXIMITY_PCT = 0.03
OB_MAX_WIDTH_ATR_MULT = 2.0

USE_VOLUME_FILTER = True
VOLUME_LOOKBACK = 20
MIN_VOLUME_MULT = 1.0

LIQUIDITY_LOOKBACK = 20
SWEEP_MAX_AGE = 120
MIN_WICK_RATIO = 0.3
MAX_SWEEP_DISTANCE_ATR_MULT = 2.0

MIN_RISK_REWARD = 1.5
MIN_STOP_DISTANCE_ATR_MULT = 0.10
MAX_STOP_DISTANCE_ATR_MULT = 3.0

ENTRY_TOLERANCE_PCT_OF_FVG = 0.6

"""SYMBOL_PARAM_OVERRIDES = {
    "BTCEUR": {
        "MIN_WICK_RATIO": 0.35,
        "MIN_IMPULSE_ATR_MULT": 1.0,
        "OB_PROXIMITY_PCT": 0.010
    },
    "ETHEUR": {
        "MIN_WICK_RATIO": 0.30,
        "MIN_IMPULSE_ATR_MULT": 0.90,
        "OB_PROXIMITY_PCT": 0.012
    },
    "XRPEUR": {
        "MIN_WICK_RATIO": 0.25,
        "MIN_IMPULSE_ATR_MULT": 0.80,
        "OB_PROXIMITY_PCT": 0.015
    },
    "DOGEEUR": {
        "MIN_WICK_RATIO": 0.25,
        "MIN_IMPULSE_ATR_MULT": 0.80,
        "OB_PROXIMITY_PCT": 0.015
    }
}"""
SYMBOL_PARAM_OVERRIDES = {}

RISK_FREE_RATE_ANNUAL = 0.0
SHARPE_PERIODS_PER_YEAR = 24 * 365
OUT_OF_SAMPLE_SPLIT = 0.70

DATA_BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com"
]

BINANCE_LIVE_URL = "https://api.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binance.vision"


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    filename="smc_getabot_bot.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ============================================================
# 3. PARAMETER-HILFSFUNKTIONEN
# ============================================================

def get_symbol_param(symbol: str, param_name: str, default_value: Any) -> Any:
    return SYMBOL_PARAM_OVERRIDES.get(symbol, {}).get(param_name, default_value)


# ============================================================
# 4. BINANCE DATA
# ============================================================

def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    endpoint = "/api/v3/klines"
    max_per_request = 1000

    for base_url in DATA_BASE_URLS:
        try:
            all_rows = []
            end_time = None
            remaining = limit

            while remaining > 0:
                request_limit = min(max_per_request, remaining)

                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": request_limit
                }

                if end_time is not None:
                    params["endTime"] = end_time

                response = requests.get(base_url + endpoint, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                all_rows = data + all_rows
                oldest_open_time = data[0][0]
                end_time = oldest_open_time - 1
                remaining -= len(data)

                if len(data) < request_limit:
                    break

            if not all_rows:
                raise RuntimeError("Keine Daten erhalten.")

            df = pd.DataFrame(all_rows, columns=[
                "open_time", "open", "high", "low", "close",
                "volume", "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df[["date", "open", "high", "low", "close", "volume"]]
            df = df.dropna()
            df = df.drop_duplicates(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)

            return df.tail(limit).reset_index(drop=True)

        except Exception as e:
            print(f"{symbol}: Fehler über {base_url}: {e}")
            logging.warning(f"{symbol}: Fehler über {base_url}: {e}")

    raise RuntimeError(f"Konnte keine Daten für {symbol} laden.")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()

    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["volume_ma"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()

    return df


# ============================================================
# 5. SWINGS / STRUKTUR / BOS / CHOCH
# ============================================================

def confirm_swing(df: pd.DataFrame, idx: int, lookback: int) -> Optional[Dict[str, Any]]:
    if idx < lookback * 2:
        return None

    swing_idx = idx - lookback
    start = swing_idx - lookback
    end = swing_idx + lookback

    window = df.iloc[start:end + 1]
    candle = df.iloc[swing_idx]

    if candle["high"] == window["high"].max():
        return {
            "type": "high",
            "idx": swing_idx,
            "price": float(candle["high"]),
            "date": candle["date"]
        }

    if candle["low"] == window["low"].min():
        return {
            "type": "low",
            "idx": swing_idx,
            "price": float(candle["low"]),
            "date": candle["date"]
        }

    return None


def get_market_structure(swing_highs: List[Dict], swing_lows: List[Dict]) -> str:
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "neutral"

    h1, h2 = swing_highs[-2]["price"], swing_highs[-1]["price"]
    l1, l2 = swing_lows[-2]["price"], swing_lows[-1]["price"]

    if h2 > h1 and l2 > l1:
        return "bullish"

    if h2 < h1 and l2 < l1:
        return "bearish"

    return "neutral"


def detect_bos_choch(
    close: float,
    structure: str,
    last_high: Optional[Dict],
    last_low: Optional[Dict]
) -> Dict[str, bool]:

    result = {
        "bullish_bos": False,
        "bearish_bos": False,
        "bullish_choch": False,
        "bearish_choch": False
    }

    if last_high and close > last_high["price"]:
        if structure == "bearish":
            result["bullish_choch"] = True
        else:
            result["bullish_bos"] = True

    if last_low and close < last_low["price"]:
        if structure == "bullish":
            result["bearish_choch"] = True
        else:
            result["bearish_bos"] = True

    return result


# ============================================================
# 6. FVG
# ============================================================

def detect_fvg(df: pd.DataFrame, idx: int) -> Optional[Dict[str, Any]]:
    if idx < 2:
        return None

    c1 = df.iloc[idx - 2]
    c3 = df.iloc[idx]

    if c3["low"] > c1["high"]:
        lower = float(c1["high"])
        upper = float(c3["low"])

        return {
            "direction": "long",
            "created_idx": idx,
            "active_from_idx": idx + ENTRY_DELAY_CANDLES,
            "created_time": c3["date"],
            "lower": lower,
            "upper": upper,
            "entry": (lower + upper) / 2,
            "width": upper - lower
        }

    if c3["high"] < c1["low"]:
        lower = float(c3["high"])
        upper = float(c1["low"])

        return {
            "direction": "short",
            "created_idx": idx,
            "active_from_idx": idx + ENTRY_DELAY_CANDLES,
            "created_time": c3["date"],
            "lower": lower,
            "upper": upper,
            "entry": (lower + upper) / 2,
            "width": upper - lower
        }

    return None


def is_fvg_entry_touched(fvg: Dict, candle: pd.Series) -> bool:
    tolerance = fvg["width"] * ENTRY_TOLERANCE_PCT_OF_FVG
    return candle["low"] <= fvg["entry"] + tolerance and candle["high"] >= fvg["entry"] - tolerance


# ============================================================
# 7. ORDER BLOCK
# ============================================================

def find_strict_order_block(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    break_type: str,
    symbol: str
) -> Optional[Dict[str, Any]]:

    if idx <= ATR_PERIOD:
        return None

    atr = df.iloc[idx]["atr"]
    if pd.isna(atr) or atr <= 0:
        return None

    min_impulse_atr = get_symbol_param(symbol, "MIN_IMPULSE_ATR_MULT", MIN_IMPULSE_ATR_MULT)

    impulse_start = max(0, idx - 3)
    impulse_move = abs(df.iloc[idx]["close"] - df.iloc[impulse_start]["open"])

    if impulse_move < min_impulse_atr * atr:
        return None

    start = max(0, idx - OB_SEARCH_BACK)

    for j in range(idx - 1, start - 1, -1):
        candle = df.iloc[j]

        if direction == "long":
            valid_candle = candle["close"] < candle["open"]
        else:
            valid_candle = candle["close"] > candle["open"]

        if not valid_candle:
            continue

        ob_low = float(candle["low"])
        ob_high = float(candle["high"])
        ob_width = ob_high - ob_low

        if ob_width <= 0:
            continue

        if ob_width > OB_MAX_WIDTH_ATR_MULT * atr:
            continue

        if USE_VOLUME_FILTER:
            vol_ma = candle["volume_ma"]
            if pd.notna(vol_ma) and vol_ma > 0:
                if candle["volume"] < MIN_VOLUME_MULT * vol_ma:
                    continue

        return {
            "direction": "bullish" if direction == "long" else "bearish",
            "idx": j,
            "date": candle["date"],
            "low": ob_low,
            "high": ob_high,
            "mid": (ob_low + ob_high) / 2,
            "break_idx": idx,
            "break_time": df.iloc[idx]["date"],
            "break_type": break_type,
            "impulse_atr_multiple": impulse_move / atr
        }

    return None


def is_order_block_mitigated(df: pd.DataFrame, ob: Dict, current_idx: int) -> bool:
    if ob is None:
        return True

    start = ob["idx"] + 1
    end = current_idx

    if end <= start:
        return False

    future = df.iloc[start:end + 1]

    if ob["direction"] == "bullish":
        return bool((future["low"] <= ob["low"]).any())

    if ob["direction"] == "bearish":
        return bool((future["high"] >= ob["high"]).any())

    return True


def price_near_order_block(price: float, ob: Optional[Dict], symbol: str) -> bool:
    if ob is None:
        return False

    ob_proximity_pct = get_symbol_param(symbol, "OB_PROXIMITY_PCT", OB_PROXIMITY_PCT)

    if ob["low"] <= price <= ob["high"]:
        return True

    distance = min(abs(price - ob["low"]), abs(price - ob["high"])) / price
    return distance <= ob_proximity_pct


def valid_order_block_for_entry(
    df: pd.DataFrame,
    ob: Optional[Dict],
    current_idx: int,
    direction: str,
    entry_price: float,
    symbol: str
) -> bool:

    if ob is None:
        return False

    if current_idx - ob["idx"] > OB_MAX_AGE:
        return False

    if direction == "long" and ob["direction"] != "bullish":
        return False

    if direction == "short" and ob["direction"] != "bearish":
        return False

    if is_order_block_mitigated(df, ob, current_idx):
        return False

    return price_near_order_block(entry_price, ob, symbol)


# ============================================================
# 8. LIQUIDITY SWEEP
# ============================================================

def detect_strict_liquidity_sweep(
    df: pd.DataFrame,
    idx: int,
    swing_highs: List[Dict],
    swing_lows: List[Dict],
    symbol: str
) -> Optional[Dict[str, Any]]:

    if idx <= ATR_PERIOD:
        return None

    candle = df.iloc[idx]
    atr = candle["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    min_wick_ratio = get_symbol_param(symbol, "MIN_WICK_RATIO", MIN_WICK_RATIO)

    candle_range = candle["high"] - candle["low"]
    if candle_range <= 0:
        return None

    recent_lows = swing_lows[-LIQUIDITY_LOOKBACK:]
    recent_highs = swing_highs[-LIQUIDITY_LOOKBACK:]

    for low in recent_lows:
        swept_level = low["price"]

        if candle["low"] < swept_level and candle["close"] > swept_level:
            lower_wick = min(candle["open"], candle["close"]) - candle["low"]
            wick_ratio = lower_wick / candle_range
            distance = abs(swept_level - candle["low"])

            if wick_ratio >= min_wick_ratio and distance <= MAX_SWEEP_DISTANCE_ATR_MULT * atr:
                return {
                    "direction": "bullish",
                    "idx": idx,
                    "date": candle["date"],
                    "swept_level": swept_level,
                    "wick_ratio": wick_ratio,
                    "distance_atr": distance / atr
                }

    for high in recent_highs:
        swept_level = high["price"]

        if candle["high"] > swept_level and candle["close"] < swept_level:
            upper_wick = candle["high"] - max(candle["open"], candle["close"])
            wick_ratio = upper_wick / candle_range
            distance = abs(candle["high"] - swept_level)

            if wick_ratio >= min_wick_ratio and distance <= MAX_SWEEP_DISTANCE_ATR_MULT * atr:
                return {
                    "direction": "bearish",
                    "idx": idx,
                    "date": candle["date"],
                    "swept_level": swept_level,
                    "wick_ratio": wick_ratio,
                    "distance_atr": distance / atr
                }

    return None


def valid_recent_sweep(sweep: Optional[Dict], current_idx: int, direction: str) -> bool:
    if sweep is None:
        return False

    if current_idx - sweep["idx"] > SWEEP_MAX_AGE:
        return False

    if direction == "long":
        return sweep["direction"] == "bullish"

    if direction == "short":
        return sweep["direction"] == "bearish"

    return False


# ============================================================
# 9. PREMIUM / DISCOUNT
# ============================================================

def get_range_zone(last_high: Optional[Dict], last_low: Optional[Dict], price: float) -> Optional[str]:
    if not last_high or not last_low:
        return None

    high = last_high["price"]
    low = last_low["price"]

    if high <= low:
        return None

    equilibrium = (high + low) / 2

    if price < equilibrium:
        return "discount"

    if price > equilibrium:
        return "premium"

    return "equilibrium"


# ============================================================
# 10. RISIKO
# ============================================================

def current_total_open_risk(state: Dict[str, Dict]) -> float:
    total_risk = 0.0

    for s in state.values():
        trade = s.get("open_trade")
        if trade is not None:
            total_risk += float(trade["risk_eur"])

    return total_risk


def can_open_new_trade(state: Dict[str, Dict], capital: float, new_trade_risk: float) -> bool:
    total_open_risk = current_total_open_risk(state)
    max_allowed_risk = capital * MAX_TOTAL_OPEN_RISK

    return total_open_risk + new_trade_risk <= max_allowed_risk


# ============================================================
# 11. TRADE-SETUP
# ============================================================

def create_trade_from_fvg(
    symbol: str,
    fvg: Dict,
    candle: pd.Series,
    current_idx: int,
    structure: str,
    break_flags: Dict[str, bool],
    recent_sweep: Optional[Dict],
    last_high: Optional[Dict],
    last_low: Optional[Dict],
    order_block: Optional[Dict],
    df: pd.DataFrame,
    capital: float
) -> Optional[Dict[str, Any]]:

    direction = fvg["direction"]

    if direction == "short" and not ALLOW_SHORTS:
        return None

    entry = float(fvg["entry"])

    if direction == "long":
        entry = entry * (1 + SLIPPAGE)
    else:
        entry = entry * (1 - SLIPPAGE)
        
    atr = candle["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    zone = get_range_zone(last_high, last_low, entry)

    if direction == "long":
        valid_structure = structure == "bullish" or break_flags["bullish_choch"]
        valid_break = break_flags["bullish_bos"] or break_flags["bullish_choch"]
        valid_zone = zone == "discount"
        valid_sweep = valid_recent_sweep(recent_sweep, current_idx, "long")
        valid_ob = valid_order_block_for_entry(df, order_block, current_idx, "long", entry, symbol)

        if not (valid_structure and valid_break and valid_zone and valid_sweep and valid_ob):
            return None

        stop_candidates = [fvg["lower"], order_block["low"]]
        if last_low:
            stop_candidates.append(last_low["price"])

        stop_loss = min(stop_candidates)
        risk_per_unit = entry - stop_loss
        take_profit = entry + RRR * risk_per_unit

    else:
        valid_structure = structure == "bearish" or break_flags["bearish_choch"]
        valid_break = break_flags["bearish_bos"] or break_flags["bearish_choch"]
        valid_zone = zone == "premium"
        valid_sweep = valid_recent_sweep(recent_sweep, current_idx, "short")
        valid_ob = valid_order_block_for_entry(df, order_block, current_idx, "short", entry, symbol)

        if not (valid_structure and valid_break and valid_zone and valid_sweep and valid_ob):
            return None

        stop_candidates = [fvg["upper"], order_block["high"]]
        if last_high:
            stop_candidates.append(last_high["price"])

        stop_loss = max(stop_candidates)
        risk_per_unit = stop_loss - entry
        take_profit = entry - RRR * risk_per_unit

    if risk_per_unit <= 0:
        return None

    rr = abs(take_profit - entry) / risk_per_unit

    if rr < MIN_RISK_REWARD:
        return None

    if risk_per_unit < MIN_STOP_DISTANCE_ATR_MULT * atr:
        return None

    if risk_per_unit > MAX_STOP_DISTANCE_ATR_MULT * atr:
        return None

    risk_eur = capital * RISK_PER_TRADE
    position_size = risk_eur / risk_per_unit

    if position_size <= 0 or not np.isfinite(position_size):
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "entry_time": candle["date"],
        "entry_idx": current_idx,
        "entry_price": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_size": position_size,
        "risk_eur": risk_eur,
        "setup_reasons": (
            f"{direction}, structure={structure}, BOS/CHoCH, FVG, OB, Sweep, "
            f"zone={zone}, RR={round(rr, 2)}, SL_ATR={round(risk_per_unit / atr, 2)}"
        )
    }


# ============================================================
# 12. EXIT / PNL
# ============================================================

def apply_slippage(price: float, direction: str, action: str) -> float:
    if action == "entry":
        return price * (1 + SLIPPAGE) if direction == "long" else price * (1 - SLIPPAGE)

    if action == "exit":
        return price * (1 - SLIPPAGE) if direction == "long" else price * (1 + SLIPPAGE)

    return price


"""def check_exit(open_trade: Dict, candle: pd.Series) -> Tuple[Optional[float], Optional[str]]:
    direction = open_trade["direction"]

    if direction == "long":
        if candle["low"] <= open_trade["stop_loss"]:
            return open_trade["stop_loss"], "Stop Loss"

        if candle["high"] >= open_trade["take_profit"]:
            return open_trade["take_profit"], "Take Profit"

    if direction == "short":
        if candle["high"] >= open_trade["stop_loss"]:
            return open_trade["stop_loss"], "Stop Loss"

        if candle["low"] <= open_trade["take_profit"]:
            return open_trade["take_profit"], "Take Profit"

    return None, None"""
"""def check_exit(open_trade: Dict, candle: pd.Series) -> Tuple[Optional[float], Optional[str]]:
    direction = open_trade["direction"]

    if direction == "long":
        hit_sl = candle["low"] <= open_trade["stop_loss"]
        hit_tp = candle["high"] >= open_trade["take_profit"]

        # Konservativ: Wenn SL und TP in derselben Kerze erreicht werden,
        # wird angenommen, dass zuerst der Stop Loss getroffen wurde.
        if hit_sl and hit_tp:
            return open_trade["stop_loss"], "Stop Loss - same candle conservative"

        if hit_sl:
            return open_trade["stop_loss"], "Stop Loss"

        if hit_tp:
            return open_trade["take_profit"], "Take Profit"

    if direction == "short":
        hit_sl = candle["high"] >= open_trade["stop_loss"]
        hit_tp = candle["low"] <= open_trade["take_profit"]

        if hit_sl and hit_tp:
            return open_trade["stop_loss"], "Stop Loss - same candle conservative"

        if hit_sl:
            return open_trade["stop_loss"], "Stop Loss"

        if hit_tp:
            return open_trade["take_profit"], "Take Profit"

    return None, None"""
def check_exit(open_trade: Dict, candle: pd.Series) -> Tuple[Optional[float], Optional[str]]:
    direction = open_trade["direction"]

    if direction == "long":
        hit_sl = candle["low"] <= open_trade["stop_loss"]
        hit_tp = candle["high"] >= open_trade["take_profit"]

        if hit_sl and hit_tp:
            return open_trade["stop_loss"], "Stop Loss - same candle conservative"

        if hit_sl:
            return open_trade["stop_loss"], "Stop Loss"

        if hit_tp:
            return open_trade["take_profit"], "Take Profit"

    if direction == "short":
        hit_sl = candle["high"] >= open_trade["stop_loss"]
        hit_tp = candle["low"] <= open_trade["take_profit"]

        if hit_sl and hit_tp:
            return open_trade["stop_loss"], "Stop Loss - same candle conservative"

        if hit_sl:
            return open_trade["stop_loss"], "Stop Loss"

        if hit_tp:
            return open_trade["take_profit"], "Take Profit"

    return None, None

def close_trade(
    open_trade: Dict,
    exit_price: float,
    exit_time: pd.Timestamp,
    capital: float,
    exit_reason: str
) -> Tuple[Dict[str, Any], float]:

    direction = open_trade["direction"]

    entry_price = apply_slippage(open_trade["entry_price"], direction, "entry")
    exit_price = apply_slippage(exit_price, direction, "exit")

    gross_entry_value = open_trade["position_size"] * entry_price
    gross_exit_value = open_trade["position_size"] * exit_price
    fees = (gross_entry_value + gross_exit_value) * FEE_RATE

    if direction == "long":
        pnl = open_trade["position_size"] * (exit_price - entry_price) - fees
    else:
        pnl = open_trade["position_size"] * (entry_price - exit_price) - fees

    r_multiple = pnl / open_trade["risk_eur"] if open_trade["risk_eur"] != 0 else 0
    capital_after = capital + pnl

    return {
        "Symbol": open_trade["symbol"],
        "Richtung": direction,
        "Entry-Zeitpunkt": open_trade["entry_time"],
        "Exit-Zeitpunkt": exit_time,
        "Entry-Preis": round(entry_price, 6),
        "Stop Loss": round(open_trade["stop_loss"], 6),
        "Take Profit": round(open_trade["take_profit"], 6),
        "Exit-Preis": round(exit_price, 6),
        "Exit-Grund": exit_reason,
        "Positionsgröße": round(open_trade["position_size"], 8),
        "Gebühren EUR": round(fees, 2),
        "Risiko EUR": round(open_trade["risk_eur"], 2),
        "Gewinn/Verlust EUR": round(pnl, 2),
        "R-Multiple": round(r_multiple, 2),
        "Kapital nach Trade": round(capital_after, 2),
        "Setup-Gründe": open_trade["setup_reasons"]
    }, capital_after


# ============================================================
# 13. EQUITY CURVE
# ============================================================

def mark_to_market_equity(
    capital: float,
    open_trades_by_symbol: Dict[str, Optional[Dict]],
    candles_by_symbol_at_time: Dict[str, pd.Series]
) -> float:

    unrealized = 0.0

    for symbol, trade in open_trades_by_symbol.items():
        if trade is None:
            continue

        candle = candles_by_symbol_at_time.get(symbol)
        if candle is None:
            continue

        close_price = candle["close"]

        if trade["direction"] == "long":
            unrealized += trade["position_size"] * (close_price - trade["entry_price"])
        else:
            unrealized += trade["position_size"] * (trade["entry_price"] - close_price)

    return capital + unrealized


# ============================================================
# 14. BACKTEST
# ============================================================

def backtest(data_by_symbol: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    capital = INITIAL_CAPITAL
    trades = []
    equity_records = []

    state = {}

    for symbol, df in data_by_symbol.items():
        state[symbol] = {
            "df": df,
            "swing_highs": [],
            "swing_lows": [],
            "structure": "neutral",
            "pending_fvgs": [],
            "open_trade": None,
            "last_break_flags": {
                "bullish_bos": False,
                "bearish_bos": False,
                "bullish_choch": False,
                "bearish_choch": False
            },
            "last_order_block_long": None,
            "last_order_block_short": None,
            "last_sweep": None,
            "last_entry_hour": None
        }

    all_dates = sorted(set(
        date
        for df in data_by_symbol.values()
        for date in df["date"].tolist()
    ))

    for current_time in all_dates:
        candles_at_time = {}

        for symbol in list(state.keys()):
            s = state[symbol]
            df = s["df"]

            rows = df.index[df["date"] == current_time].tolist()
            if not rows:
                continue

            i = rows[0]
            candle = df.iloc[i]
            candles_at_time[symbol] = candle

            if s["open_trade"] is not None:
                exit_price, exit_reason = check_exit(s["open_trade"], candle)

                if exit_price is not None:
                    result, capital = close_trade(
                        s["open_trade"],
                        exit_price,
                        candle["date"],
                        capital,
                        exit_reason
                    )
                    trades.append(result)
                    s["open_trade"] = None

            swing = confirm_swing(df, i, SWING_LOOKBACK)

            if swing:
                if swing["type"] == "high":
                    s["swing_highs"].append(swing)
                else:
                    s["swing_lows"].append(swing)

            s["structure"] = get_market_structure(s["swing_highs"], s["swing_lows"])

            last_high = s["swing_highs"][-1] if s["swing_highs"] else None
            last_low = s["swing_lows"][-1] if s["swing_lows"] else None

            break_flags = detect_bos_choch(
                candle["close"],
                s["structure"],
                last_high,
                last_low
            )
            s["last_break_flags"] = break_flags

            sweep = detect_strict_liquidity_sweep(
                df,
                i,
                s["swing_highs"],
                s["swing_lows"],
                symbol
            )

            if sweep:
                s["last_sweep"] = sweep

            new_fvg = detect_fvg(df, i)

            if new_fvg:
                s["pending_fvgs"].append(new_fvg)

            s["pending_fvgs"] = [
                fvg for fvg in s["pending_fvgs"]
                if i - fvg["created_idx"] <= FVG_MAX_AGE
            ]

            if break_flags["bullish_bos"] or break_flags["bullish_choch"]:
                ob = find_strict_order_block(
                    df,
                    i,
                    "long",
                    "bullish_choch" if break_flags["bullish_choch"] else "bullish_bos",
                    symbol
                )
                if ob:
                    s["last_order_block_long"] = ob

            if break_flags["bearish_bos"] or break_flags["bearish_choch"]:
                ob = find_strict_order_block(
                    df,
                    i,
                    "short",
                    "bearish_choch" if break_flags["bearish_choch"] else "bearish_bos",
                    symbol
                )
                if ob:
                    s["last_order_block_short"] = ob

            current_entry_hour = candle["date"].floor("h")

            if s["open_trade"] is None and s["last_entry_hour"] != current_entry_hour:
                for fvg in list(s["pending_fvgs"]):
                    if i < fvg.get("active_from_idx", fvg["created_idx"] + ENTRY_DELAY_CANDLES):
                        continue

                    if not is_fvg_entry_touched(fvg, candle):
                        continue

                    relevant_ob = (
                        s["last_order_block_long"]
                        if fvg["direction"] == "long"
                        else s["last_order_block_short"]
                    )

                    trade = create_trade_from_fvg(
                        symbol=symbol,
                        fvg=fvg,
                        candle=candle,
                        current_idx=i,
                        structure=s["structure"],
                        break_flags=s["last_break_flags"],
                        recent_sweep=s["last_sweep"],
                        last_high=last_high,
                        last_low=last_low,
                        order_block=relevant_ob,
                        df=df,
                        capital=capital
                    )

                    if trade:
                        if not can_open_new_trade(state, capital, trade["risk_eur"]):
                            continue

                        s["open_trade"] = trade
                        s["last_entry_hour"] = current_entry_hour
                        s["pending_fvgs"].remove(fvg)
                        break

        open_trades = {
            symbol: state[symbol]["open_trade"]
            for symbol in state
        }

        equity_records.append({
            "date": current_time,
            "equity": mark_to_market_equity(capital, open_trades, candles_at_time)
        })

    for symbol, s in state.items():
        if s["open_trade"] is None:
            continue

        last_candle = s["df"].iloc[-1]

        result, capital = close_trade(
            s["open_trade"],
            float(last_candle["close"]),
            last_candle["date"],
            capital,
            "Backtest-Ende"
        )

        trades.append(result)
        s["open_trade"] = None

    equity_df = pd.DataFrame(equity_records)

    if not equity_df.empty:
        equity_df.loc[equity_df.index[-1], "equity"] = capital

    return pd.DataFrame(trades), equity_df, capital


# ============================================================
# 15. OUT-OF-SAMPLE SPLIT
# ============================================================

"""def split_data_out_of_sample(
    data_by_symbol: Dict[str, pd.DataFrame],
    split_ratio: float = OUT_OF_SAMPLE_SPLIT
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:

    train_data = {}
    test_data = {}

    for symbol, df in data_by_symbol.items():
        split_idx = int(len(df) * split_ratio)

        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        if len(train_df) > ATR_PERIOD + 100:
            train_data[symbol] = train_df

        if len(test_df) > ATR_PERIOD + 100:
            test_data[symbol] = test_df

    return train_data, test_data"""

def walk_forward_splits(
    data_by_symbol: Dict[str, pd.DataFrame],
    train_size: float = 0.6,
    test_size: float = 0.2,
    step_size: float = 0.2
):
    splits = []

    for symbol, df in data_by_symbol.items():
        n = len(df)

        train_len = int(n * train_size)
        test_len = int(n * test_size)
        step_len = int(n * step_size)

        start = 0

        while start + train_len + test_len <= n:
            train_df = df.iloc[start:start + train_len].reset_index(drop=True)
            test_df = df.iloc[start + train_len:start + train_len + test_len].reset_index(drop=True)

            splits.append({
                "symbol": symbol,
                "train": train_df,
                "test": test_df,
                "start_idx": start
            })

            start += step_len

    return splits
# ============================================================
# 16. PERFORMANCE
# ============================================================

def max_drawdown_from_equity(equity_df: pd.DataFrame) -> float:
    if equity_df.empty or len(equity_df) < 2:
        return 0.0

    equity = equity_df["equity"].astype(float).values
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks

    return float(drawdowns.min() * 100)


def sharpe_ratio_time_based(
    equity_df: pd.DataFrame,
    risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL,
    periods_per_year: int = SHARPE_PERIODS_PER_YEAR
) -> float:

    if equity_df.empty or len(equity_df) < 3:
        return 0.0

    returns = equity_df["equity"].astype(float).pct_change().dropna()

    if len(returns) < 2:
        return 0.0

    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = returns - rf_per_period
    std = excess.std(ddof=1)

    if std == 0 or np.isnan(std):
        return 0.0

    return float(excess.mean() / std * sqrt(periods_per_year))


def performance_summary(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    final_capital: float
) -> pd.DataFrame:

    if trades_df.empty:
        return pd.DataFrame([{
            "Gruppe": "Gesamt",
            "Trades": 0,
            "Gewinner": 0,
            "Verlierer": 0,
            "Winrate %": 0,
            "Ø R": 0,
            "Profit Factor": 0,
            "G/V EUR": 0,
            "Endkapital": round(final_capital, 2),
            "Max Drawdown %": 0,
            "Sharpe Ratio": 0,
            "Return %": round((final_capital / INITIAL_CAPITAL - 1) * 100, 2)
        }])

    summaries = []
    groups = list(trades_df["Symbol"].unique()) + ["Gesamt"]

    for group in groups:
        df = trades_df if group == "Gesamt" else trades_df[trades_df["Symbol"] == group]

        pnl = df["Gewinn/Verlust EUR"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf

        summaries.append({
            "Gruppe": group,
            "Trades": len(df),
            "Gewinner": len(wins),
            "Verlierer": len(losses),
            "Winrate %": round(len(wins) / len(df) * 100, 2) if len(df) else 0,
            "Ø R": round(df["R-Multiple"].mean(), 2) if len(df) else 0,
            "Profit Factor": round(profit_factor, 2) if profit_factor != np.inf else "inf",
            "G/V EUR": round(pnl.sum(), 2),
            "Endkapital": round(final_capital, 2) if group == "Gesamt" else "-",
            "Max Drawdown %": round(max_drawdown_from_equity(equity_df), 2) if group == "Gesamt" else "-",
            "Sharpe Ratio": round(sharpe_ratio_time_based(equity_df), 2) if group == "Gesamt" else "-",
            "Return %": round((final_capital / INITIAL_CAPITAL - 1) * 100, 2) if group == "Gesamt" else "-"
        })

    return pd.DataFrame(summaries)


# ============================================================
# 17. LIVE-TRADING-HILFSFUNKTIONEN
# ============================================================

def get_binance_base_url() -> str:
    return BINANCE_TESTNET_URL if TESTNET else BINANCE_LIVE_URL


def sign_params(params: Dict[str, Any], secret_key: str) -> Dict[str, Any]:
    query = urlencode(params)
    signature = hmac.new(
        secret_key.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    params["signature"] = signature
    return params


def binance_signed_request(
    method: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError("BINANCE_API_KEY oder BINANCE_SECRET_KEY fehlt.")

    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    params = sign_params(params, secret_key)

    headers = {"X-MBX-APIKEY": api_key}
    url = get_binance_base_url() + endpoint

    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        timeout=20
    )

    if response.status_code >= 400:
        logging.error(f"Binance Fehler {response.status_code}: {response.text}")
        raise RuntimeError(f"Binance API Fehler: {response.text}")

    return response.json()


def get_account_info() -> Dict[str, Any]:
    return binance_signed_request("GET", "/api/v3/account")


def get_symbol_info(symbol: str) -> Dict[str, Any]:
    url = get_binance_base_url() + "/api/v3/exchangeInfo"
    response = requests.get(url, params={"symbol": symbol}, timeout=20)
    response.raise_for_status()
    return response.json()["symbols"][0]


def get_filter_value(symbol_info: Dict, filter_type: str, key: str) -> float:
    for f in symbol_info["filters"]:
        if f["filterType"] == filter_type:
            return float(f[key])
    raise RuntimeError(f"Filter {filter_type}:{key} nicht gefunden.")


def round_step_size(quantity: float, step_size: float) -> float:
    return np.floor(quantity / step_size) * step_size


def round_tick_size(price: float, tick_size: float) -> float:
    return np.floor(price / tick_size) * tick_size


def calculate_live_position_size(capital: float, entry: float, stop_loss: float) -> float:
    risk_eur = capital * RISK_PER_TRADE
    risk_per_unit = abs(entry - stop_loss)

    if risk_per_unit <= 0:
        return 0.0

    return risk_eur / risk_per_unit


def place_spot_market_order(symbol: str, side: str, quantity: float) -> Dict[str, Any]:
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity
    }

    logging.info(f"Live Order: {params}")

    if not COMPETITION_MODE:
        print(f"[DRY RUN] Order nicht gesendet: {params}")
        return {"dry_run": True, "params": params}

    return binance_signed_request("POST", "/api/v3/order", params)


# ============================================================
# 18. GETABOT LIVE COMPETITION
# ============================================================

def load_live_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "last_signal_hour": {},
            "open_positions": {},
            "trades_today": 0,
            "daily_pnl": 0.0,
            "current_day": None
        }

    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_live_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4, default=str)


def reset_daily_limits_if_new_day(state: Dict[str, Any]) -> Dict[str, Any]:
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    if state.get("current_day") != today:
        state["current_day"] = today
        state["trades_today"] = 0
        state["daily_pnl"] = 0.0

    return state


def seconds_after_hour() -> int:
    now = pd.Timestamp.utcnow()
    return now.minute * 60 + now.second


def is_signal_window() -> bool:
    sec = seconds_after_hour()
    return SIGNAL_DELAY_SECONDS_AFTER_HOUR <= sec <= MAX_SIGNAL_SECONDS_AFTER_HOUR


def competition_safety_check(state: Dict[str, Any]) -> bool:
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        return False

    if len(state["open_positions"]) >= MAX_OPEN_POSITIONS:
        return False

    if state["daily_pnl"] <= -INITIAL_CAPITAL * MAX_DAILY_LOSS_PCT:
        return False

    return True


def build_getabot_payload(trade: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": trade["symbol"],
        "side": "BUY",
        "entry": round(float(trade["entry_price"]), 8),
        "stop_loss": round(float(trade["stop_loss"]), 8),
        "take_profit": round(float(trade["take_profit"]), 8),
        "timestamp": str(trade["entry_time"]),
        "strategy": "SMC_FVG_OB_SWEEP",
        "risk_eur": round(float(trade["risk_eur"]), 2),
        "position_size": round(float(trade["position_size"]), 8),
        "reason": trade["setup_reasons"]
    }


def send_getabot_signal(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not GETABOT_WEBHOOK_URL:
        print("[DRY RUN] GetaBot-Webhook leer. Signal wäre:")
        print(payload)
        return {"dry_run": True, "payload": payload}

    response = requests.post(
        GETABOT_WEBHOOK_URL,
        json=payload,
        timeout=20
    )

    if response.status_code >= 400:
        logging.error(f"GetaBot Fehler {response.status_code}: {response.text}")
        raise RuntimeError(f"GetaBot Fehler: {response.text}")

    return response.json() if response.text else {"status": "sent"}


def generate_latest_signal_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    df = fetch_binance_klines(symbol, INTERVAL, LIMIT)
    df = add_indicators(df)

    if len(df) < ATR_PERIOD + 100:
        return None

    trades_df, equity_df, final_capital = backtest({symbol: df})

    if trades_df.empty:
        return None

    latest_trade = trades_df.iloc[-1].to_dict()

    last_closed_candle_time = df["date"].iloc[-1]
    trade_time = pd.Timestamp(latest_trade["Entry-Zeitpunkt"])

    if trade_time < last_closed_candle_time - pd.Timedelta(hours=1):
        return None

    if latest_trade["Richtung"] != "long":
        return None

    return {
        "symbol": latest_trade["Symbol"],
        "direction": latest_trade["Richtung"],
        "entry_time": latest_trade["Entry-Zeitpunkt"],
        "entry_price": latest_trade["Entry-Preis"],
        "stop_loss": latest_trade["Stop Loss"],
        "take_profit": latest_trade["Take Profit"],
        "position_size": latest_trade["Positionsgröße"],
        "risk_eur": latest_trade["Risiko EUR"],
        "setup_reasons": latest_trade["Setup-Gründe"]
    }


def main_getabot_live_loop() -> None:
    print("Starte BVH x GetaBot Expert-Modus Bot...")
    print(f"Webhook gesetzt: {bool(GETABOT_WEBHOOK_URL)}")
    print(f"Handelspaare: {SYMBOLS}")
    print("Modus: DRY RUN" if not GETABOT_WEBHOOK_URL else "Modus: LIVE WEBHOOK")

    state = load_live_state()

    while True:
        try:
            state = reset_daily_limits_if_new_day(state)

            if not is_signal_window():
                time.sleep(LIVE_POLL_SECONDS)
                continue

            for symbol in SYMBOLS:
                if not competition_safety_check(state):
                    continue

                signal = generate_latest_signal_for_symbol(symbol)

                if signal is None:
                    continue

                signal_hour = pd.Timestamp(signal["entry_time"]).floor("h")
                last_signal_hour = state["last_signal_hour"].get(symbol)

                if last_signal_hour == str(signal_hour):
                    continue

                payload = build_getabot_payload(signal)
                result = send_getabot_signal(payload)

                state["last_signal_hour"][symbol] = str(signal_hour)
                state["trades_today"] += 1

                state["open_positions"][symbol] = {
                    "entry_time": str(signal["entry_time"]),
                    "entry_price": signal["entry_price"],
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "risk_eur": signal["risk_eur"],
                    "direction": signal["direction"]
                }

                save_live_state(state)

                print(f"Signal für {symbol} gesendet:")
                print(result)

            time.sleep(LIVE_POLL_SECONDS)

        except KeyboardInterrupt:
            print("Bot manuell gestoppt.")
            save_live_state(state)
            break

        except Exception as e:
            logging.error(f"Live Loop Fehler: {e}")
            print(f"Live Loop Fehler: {e}")
            time.sleep(LIVE_POLL_SECONDS)


# ============================================================
# 19. AUSGABE
# ============================================================

def print_table(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if df.empty:
        print("Keine Daten vorhanden.")
        return

    if HAS_TABULATE:
        print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    else:
        print(df.to_string(index=False))

def run_named_backtest(name: str, data_by_symbol: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "#" * 100)
    print(name)
    print("#" * 100)

    trades_df, equity_df, final_capital = backtest(data_by_symbol)

    print_table(trades_df, f"TRADE-TABELLE - {name}")

    summary_df = performance_summary(trades_df, equity_df, final_capital)
    print_table(summary_df, f"PERFORMANCE-ZUSAMMENFASSUNG - {name}")

    print(f"\n{name} abgeschlossen.")
    print(f"Endkapital: {final_capital:.2f} EUR")
    print(f"Gesamtrendite: {(final_capital / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print(f"Zeitbasierte Sharpe Ratio: {sharpe_ratio_time_based(equity_df):.2f}")
    
# ============================================================
# 20. MAIN BACKTEST
# ============================================================

"""def run_named_backtest(name: str, data_by_symbol: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "#" * 100)
    print(name)
    print("#" * 100)

    trades_df, equity_df, final_capital = backtest(data_by_symbol)

    print_table(trades_df, f"TRADE-TABELLE - {name}")

    summary_df = performance_summary(trades_df, equity_df, final_capital)
    print_table(summary_df, f"PERFORMANCE-ZUSAMMENFASSUNG - {name}")

    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    trades_df.to_csv(f"trades_{safe_name}.csv", index=False)
    equity_df.to_csv(f"equity_curve_{safe_name}.csv", index=False)

    print(f"\n{name} abgeschlossen.")
    print(f"Endkapital: {final_capital:.2f} EUR")
    print(f"Gesamtrendite: {(final_capital / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print(f"Zeitbasierte Sharpe Ratio: {sharpe_ratio_time_based(equity_df):.2f}")"""

def run_walk_forward(data_by_symbol: Dict[str, pd.DataFrame]):

    print("\n" + "#" * 100)
    print("WALK-FORWARD ANALYSE")
    print("#" * 100)

    splits = walk_forward_splits(data_by_symbol)

    results = []

    for i, split in enumerate(splits):
        symbol = split["symbol"]

        print(f"\n--- Split {i+1} | {symbol} ---")

        train_data = {symbol: split["train"]}
        test_data = {symbol: split["test"]}

        # Train (nur optional zur Kontrolle)
        trades_train, eq_train, cap_train = backtest(train_data)

        # Test (WICHTIG)
        trades_test, eq_test, cap_test = backtest(test_data)

        summary = performance_summary(trades_test, eq_test, cap_test)

        print(summary)

        results.append({
            "split": i + 1,
            "symbol": symbol,
            "trades": len(trades_test),
            "return": (cap_test / INITIAL_CAPITAL - 1) * 100,
            "sharpe": sharpe_ratio_time_based(eq_test),
            "max_dd": max_drawdown_from_equity(eq_test)
        })

    return pd.DataFrame(results)
    
def main_backtest() -> None:
    print("Starte SMC Crypto Bot Backtest...")
    print(f"Coins: {SYMBOLS}")
    print(f"Startkapital: {INITIAL_CAPITAL:.2f} EUR")
    print(f"Risiko pro Trade: {RISK_PER_TRADE * 100:.2f}%")
    print(f"Max. Gesamtrisiko offen: {MAX_TOTAL_OPEN_RISK * 100:.2f}%")
    print(f"Gebühr pro Order: {FEE_RATE * 100:.3f}%")
    print(f"Slippage-Annahme: {SLIPPAGE * 100:.3f}%")
    print(f"Shorts erlaubt: {ALLOW_SHORTS}")
    print("Validierung: Walk-Forward Analyse")

    data_by_symbol = {}

    for symbol in SYMBOLS:
        try:
            print(f"\nLade Daten für {symbol}...")
            df = fetch_binance_klines(symbol, INTERVAL, LIMIT)
            df = add_indicators(df)

            if len(df) < ATR_PERIOD + 100:
                print(f"{symbol}: zu wenig Daten, übersprungen.")
                continue

            data_by_symbol[symbol] = df

            print(
                f"{symbol}: {len(df)} Kerzen | "
                f"{df['date'].iloc[0]} bis {df['date'].iloc[-1]}"
            )

        except Exception as e:
            print(f"{symbol}: übersprungen wegen Fehler: {e}")
            logging.warning(f"{symbol}: übersprungen wegen Fehler: {e}")

        if not data_by_symbol:
            raise RuntimeError("Keine gültigen Daten geladen.")

        run_named_backtest("FULL SAMPLE", data_by_symbol)

        walk_forward_results = run_walk_forward(data_by_symbol)

        print("\n" + "=" * 100)
        print("WALK-FORWARD GESAMTÜBERSICHT")
        print("=" * 100)

        if HAS_TABULATE:
            print(tabulate(walk_forward_results, headers="keys", tablefmt="fancy_grid", showindex=False))
        else:
            print(walk_forward_results.to_string(index=False))
# ============================================================
# 21. START
# ============================================================

if __name__ == "__main__":
    if COMPETITION_MODE:
        main_getabot_live_loop()
    else:
        main_backtest()