import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ============================================================
# 1. KONFIGURATION
# ============================================================

SYMBOLS = ["BTCEUR", "ETHEUR", "DOGEEUR", "XRPEUR"]

INTERVAL = "1h"
LIMIT = 100000

INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01
RRR = 3.0

SWING_LOOKBACK = 2
FVG_MAX_AGE = 100
OB_PROXIMITY_PCT = 0.01
LIQUIDITY_LOOKBACK = 20

BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com"
]


# ============================================================
# 2. BINANCE-DATENABRUF
# ============================================================

def fetch_binance_klines(symbol, interval="1h", limit=1000):
    endpoint = "/api/v3/klines"
    max_per_request = 1000

    for base_url in BASE_URLS:
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

                response = requests.get(base_url + endpoint, params=params, timeout=15)
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

            df = pd.DataFrame(all_rows, columns=[
                "open_time", "open", "high", "low", "close",
                "volume", "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df[["date", "open", "high", "low", "close"]]
            df = df.dropna()
            df = df.drop_duplicates(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)

            return df.tail(limit).reset_index(drop=True)

        except Exception as e:
            print(f"Fehler bei {symbol} über {base_url}: {e}")

    raise RuntimeError(f"Konnte keine Daten für {symbol} laden.")


# ============================================================
# 3. SWING-HIGH / SWING-LOW
# ============================================================

def confirm_swing(df, idx, lookback):
    if idx < lookback * 2:
        return None

    swing_idx = idx - lookback
    start = swing_idx - lookback
    end = swing_idx + lookback

    window = df.iloc[start:end + 1]
    candle = df.iloc[swing_idx]

    is_swing_high = candle["high"] == window["high"].max()
    is_swing_low = candle["low"] == window["low"].min()

    if is_swing_high:
        return {
            "type": "high",
            "idx": swing_idx,
            "price": candle["high"],
            "date": candle["date"]
        }

    if is_swing_low:
        return {
            "type": "low",
            "idx": swing_idx,
            "price": candle["low"],
            "date": candle["date"]
        }

    return None


def get_market_structure(swing_highs, swing_lows):
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "neutral"

    h1, h2 = swing_highs[-2]["price"], swing_highs[-1]["price"]
    l1, l2 = swing_lows[-2]["price"], swing_lows[-1]["price"]

    if h2 > h1 and l2 > l1:
        return "bullish"

    if h2 < h1 and l2 < l1:
        return "bearish"

    return "neutral"


# ============================================================
# 4. BOS / CHoCH
# ============================================================

def detect_bos_choch(close, structure, last_high, last_low):
    bullish_bos = False
    bearish_bos = False
    bullish_choch = False
    bearish_choch = False

    if last_high and close > last_high["price"]:
        if structure == "bearish":
            bullish_choch = True
        else:
            bullish_bos = True

    if last_low and close < last_low["price"]:
        if structure == "bullish":
            bearish_choch = True
        else:
            bearish_bos = True

    return bullish_bos, bearish_bos, bullish_choch, bearish_choch


# ============================================================
# 5. FAIR VALUE GAPS
# ============================================================

def detect_fvg(df, idx):
    if idx < 2:
        return None

    c1 = df.iloc[idx - 2]
    c3 = df.iloc[idx]

    if c3["low"] > c1["high"]:
        lower = c1["high"]
        upper = c3["low"]
        entry = (lower + upper) / 2

        return {
            "direction": "long",
            "created_idx": idx,
            "created_time": c3["date"],
            "lower": lower,
            "upper": upper,
            "entry": entry
        }

    if c3["high"] < c1["low"]:
        lower = c3["high"]
        upper = c1["low"]
        entry = (lower + upper) / 2

        return {
            "direction": "short",
            "created_idx": idx,
            "created_time": c3["date"],
            "lower": lower,
            "upper": upper,
            "entry": entry
        }

    return None


# ============================================================
# 6. ORDER BLOCKS
# ============================================================

def find_order_block(df, idx, direction, search_back=10):
    start = max(0, idx - search_back)

    if direction == "long":
        for j in range(idx - 1, start - 1, -1):
            candle = df.iloc[j]
            if candle["close"] < candle["open"]:
                return {
                    "direction": "bullish",
                    "idx": j,
                    "low": candle["low"],
                    "high": candle["high"],
                    "date": candle["date"]
                }

    if direction == "short":
        for j in range(idx - 1, start - 1, -1):
            candle = df.iloc[j]
            if candle["close"] > candle["open"]:
                return {
                    "direction": "bearish",
                    "idx": j,
                    "low": candle["low"],
                    "high": candle["high"],
                    "date": candle["date"]
                }

    return None


def fvg_near_order_block(fvg, ob):
    if ob is None:
        return False

    fvg_mid = fvg["entry"]
    ob_mid = (ob["high"] + ob["low"]) / 2
    distance = abs(fvg_mid - ob_mid) / fvg_mid

    return distance <= OB_PROXIMITY_PCT


# ============================================================
# 7. LIQUIDITY SWEEPS
# ============================================================

def detect_liquidity_sweep(df, idx, swing_highs, swing_lows):
    candle = df.iloc[idx]

    bullish_sweep = False
    bearish_sweep = False

    recent_lows = swing_lows[-LIQUIDITY_LOOKBACK:]
    recent_highs = swing_highs[-LIQUIDITY_LOOKBACK:]

    for low in recent_lows:
        if candle["low"] < low["price"] and candle["close"] > low["price"]:
            bullish_sweep = True
            break

    for high in recent_highs:
        if candle["high"] > high["price"] and candle["close"] < high["price"]:
            bearish_sweep = True
            break

    return bullish_sweep, bearish_sweep


# ============================================================
# 8. PREMIUM / DISCOUNT
# ============================================================

def get_range_zone(last_high, last_low, price):
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
# 9. TRADE-SETUP
# ============================================================

def create_trade_from_fvg(
    symbol,
    fvg,
    candle,
    structure,
    bullish_bos,
    bearish_bos,
    bullish_choch,
    bearish_choch,
    bullish_sweep,
    bearish_sweep,
    last_high,
    last_low,
    order_block,
    capital
):
    direction = fvg["direction"]

    # Aktuell nur Long-Trades
    if direction != "long":
        return None

    entry = fvg["entry"]
    zone = get_range_zone(last_high, last_low, entry)

    reasons = []

    valid_structure = structure == "bullish" or bullish_choch
    valid_break = bullish_bos or bullish_choch
    valid_zone = zone == "discount"

    if not (valid_structure and valid_break and valid_zone):
        return None

    sl_candidates = [fvg["lower"]]

    if last_low:
        sl_candidates.append(last_low["price"])

    stop_loss = min(sl_candidates)

    if stop_loss >= entry:
        return None

    risk_per_unit = entry - stop_loss
    take_profit = entry + RRR * risk_per_unit

    reasons.append("bullish structure/CHoCH")
    reasons.append("bullish BOS/CHoCH")
    reasons.append("bullish FVG")
    reasons.append("entry in discount")

    if fvg_near_order_block(fvg, order_block):
        reasons.append("bullish OB nearby")

    if bullish_sweep:
        reasons.append("bullish liquidity sweep")

    risk_eur = capital * RISK_PER_TRADE
    position_size = risk_eur / risk_per_unit

    return {
        "symbol": symbol,
        "direction": direction,
        "entry_time": candle["date"],
        "entry_price": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_size": position_size,
        "risk_eur": risk_eur,
        "setup_reasons": ", ".join(reasons)
    }


# ============================================================
# 10. EXIT-LOGIK
# ============================================================

def check_exit(open_trade, candle):
    direction = open_trade["direction"]

    if direction == "long":
        sl_hit = candle["low"] <= open_trade["stop_loss"]
        tp_hit = candle["high"] >= open_trade["take_profit"]

        # Konservativ: Wenn SL und TP in derselben Kerze erreicht werden,
        # wird angenommen, dass zuerst der Stop Loss getroffen wurde.
        if sl_hit:
            return open_trade["stop_loss"]

        if tp_hit:
            return open_trade["take_profit"]

    if direction == "short":
        sl_hit = candle["high"] >= open_trade["stop_loss"]
        tp_hit = candle["low"] <= open_trade["take_profit"]

        if sl_hit:
            return open_trade["stop_loss"]

        if tp_hit:
            return open_trade["take_profit"]

    return None


def close_trade(open_trade, exit_price, exit_time, capital):
    direction = open_trade["direction"]

    if direction == "long":
        pnl = open_trade["position_size"] * (exit_price - open_trade["entry_price"])
    else:
        pnl = open_trade["position_size"] * (open_trade["entry_price"] - exit_price)

    r_multiple = pnl / open_trade["risk_eur"] if open_trade["risk_eur"] != 0 else 0
    capital_after = capital + pnl

    return {
        "Symbol": open_trade["symbol"],
        "Richtung": direction,
        "Entry-Zeitpunkt": open_trade["entry_time"],
        "Exit-Zeitpunkt": exit_time,
        "Entry-Preis": round(open_trade["entry_price"], 6),
        "Stop Loss": round(open_trade["stop_loss"], 6),
        "Take Profit": round(open_trade["take_profit"], 6),
        "Exit-Preis": round(exit_price, 6),
        "Positionsgröße": round(open_trade["position_size"], 8),
        "Risiko EUR": round(open_trade["risk_eur"], 2),
        "Gewinn/Verlust EUR": round(pnl, 2),
        "R-Multiple": round(r_multiple, 2),
        "Kapital nach Trade": round(capital_after, 2),
        "Setup-Gründe": open_trade["setup_reasons"]
    }, capital_after


# ============================================================
# 11. BACKTEST-ENGINE
# ============================================================

def backtest(data_by_symbol):
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []

    state = {}

    for symbol, df in data_by_symbol.items():
        state[symbol] = {
            "df": df,
            "swing_highs": [],
            "swing_lows": [],
            "structure": "neutral",
            "pending_fvgs": [],
            "open_trade": None,
            "last_bullish_bos": False,
            "last_bearish_bos": False,
            "last_bullish_choch": False,
            "last_bearish_choch": False,
            "last_bullish_sweep": False,
            "last_bearish_sweep": False,
            "last_order_block": None
        }

    all_dates = sorted(set(
        date
        for df in data_by_symbol.values()
        for date in df["date"].tolist()
    ))

    for current_time in all_dates:
        for symbol in SYMBOLS:
            s = state[symbol]
            df = s["df"]

            rows = df.index[df["date"] == current_time].tolist()

            if not rows:
                continue

            i = rows[0]
            candle = df.iloc[i]

            # 1. Offenen Trade prüfen
            if s["open_trade"] is not None:
                exit_price = check_exit(s["open_trade"], candle)

                if exit_price is not None:
                    trade_result, capital = close_trade(
                        s["open_trade"],
                        exit_price,
                        candle["date"],
                        capital
                    )

                    trades.append(trade_result)
                    equity_curve.append(capital)
                    s["open_trade"] = None

            # 2. Swing Point bestätigen
            swing = confirm_swing(df, i, SWING_LOOKBACK)

            if swing:
                if swing["type"] == "high":
                    s["swing_highs"].append(swing)
                elif swing["type"] == "low":
                    s["swing_lows"].append(swing)

            s["structure"] = get_market_structure(
                s["swing_highs"],
                s["swing_lows"]
            )

            last_high = s["swing_highs"][-1] if s["swing_highs"] else None
            last_low = s["swing_lows"][-1] if s["swing_lows"] else None

            # 3. BOS / CHoCH erkennen
            bullish_bos, bearish_bos, bullish_choch, bearish_choch = detect_bos_choch(
                candle["close"],
                s["structure"],
                last_high,
                last_low
            )

            s["last_bullish_bos"] = bullish_bos
            s["last_bearish_bos"] = bearish_bos
            s["last_bullish_choch"] = bullish_choch
            s["last_bearish_choch"] = bearish_choch

            # 4. Liquidity Sweep erkennen
            bullish_sweep, bearish_sweep = detect_liquidity_sweep(
                df,
                i,
                s["swing_highs"],
                s["swing_lows"]
            )

            s["last_bullish_sweep"] = bullish_sweep
            s["last_bearish_sweep"] = bearish_sweep

            # 5. FVG erkennen
            new_fvg = detect_fvg(df, i)

            if new_fvg:
                s["pending_fvgs"].append(new_fvg)

            # Alte FVGs entfernen
            s["pending_fvgs"] = [
                fvg for fvg in s["pending_fvgs"]
                if i - fvg["created_idx"] <= FVG_MAX_AGE
            ]

            # 6. Order Block erkennen
            if bullish_bos or bullish_choch:
                s["last_order_block"] = find_order_block(df, i, "long")

            if bearish_bos or bearish_choch:
                s["last_order_block"] = find_order_block(df, i, "short")

            # 7. Entry prüfen
            if s["open_trade"] is None:
                for fvg in list(s["pending_fvgs"]):
                    if i <= fvg["created_idx"]:
                        continue

                    entry = fvg["entry"]

                    tolerance = (fvg["upper"] - fvg["lower"]) * 0.3

                    entry_touched = (
                        candle["low"] <= entry + tolerance and
                        candle["high"] >= entry - tolerance
                    )

                    if not entry_touched:
                        continue

                    trade = create_trade_from_fvg(
                        symbol=symbol,
                        fvg=fvg,
                        candle=candle,
                        structure=s["structure"],
                        bullish_bos=s["last_bullish_bos"],
                        bearish_bos=s["last_bearish_bos"],
                        bullish_choch=s["last_bullish_choch"],
                        bearish_choch=s["last_bearish_choch"],
                        bullish_sweep=s["last_bullish_sweep"],
                        bearish_sweep=s["last_bearish_sweep"],
                        last_high=last_high,
                        last_low=last_low,
                        order_block=s["last_order_block"],
                        capital=capital
                    )

                    if trade:
                        s["open_trade"] = trade
                        s["pending_fvgs"].remove(fvg)
                        break

    return pd.DataFrame(trades), equity_curve, capital


# ============================================================
# 12. PERFORMANCE-AUSWERTUNG
# ============================================================

def max_drawdown(equity_curve):
    equity = np.array(equity_curve, dtype=float)

    if len(equity) < 2:
        return 0

    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks

    return drawdowns.min() * 100


def sharpe_ratio(equity_curve, risk_free_rate=0.0):
    """
    Berechnet die Sharpe Ratio auf Basis der Trade-Equity-Curve.

    Wichtig:
    Diese Sharpe Ratio ist trade-basiert.
    Das bedeutet: Jeder geschlossene Trade wird wie eine Periode behandelt.
    """

    equity = np.array(equity_curve, dtype=float)

    if len(equity) < 2:
        return 0

    returns = np.diff(equity) / equity[:-1]

    if len(returns) < 2:
        return 0

    excess_returns = returns - risk_free_rate

    std = np.std(excess_returns, ddof=1)

    if std == 0 or np.isnan(std):
        return 0

    return np.mean(excess_returns) / std


def performance_summary(trades_df, equity_curve, final_capital):
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
        if group == "Gesamt":
            df = trades_df.copy()
        else:
            df = trades_df[trades_df["Symbol"] == group]

        pnl = df["Gewinn/Verlust EUR"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        profit_factor = (
            wins.sum() / abs(losses.sum())
            if abs(losses.sum()) > 0 else np.inf
        )

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
            "Max Drawdown %": round(max_drawdown(equity_curve), 2) if group == "Gesamt" else "-",
            "Sharpe Ratio": round(sharpe_ratio(equity_curve), 2) if group == "Gesamt" else "-",
            "Return %": round((final_capital / INITIAL_CAPITAL - 1) * 100, 2) if group == "Gesamt" else "-"
        })

    return pd.DataFrame(summaries)


# ============================================================
# 13. AUSGABE
# ============================================================

def print_table(df, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if df.empty:
        print("Keine Daten vorhanden.")
        return

    if HAS_TABULATE:
        print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    else:
        print(df.to_string(index=False))


# ============================================================
# 14. MAIN
# ============================================================

def main():
    print("Starte SMC Crypto Trading Bot Backtest...")
    print(f"Startkapital: {INITIAL_CAPITAL:.2f} EUR")
    print(f"Risiko pro Trade: {RISK_PER_TRADE * 100:.2f}%")
    print(f"Risk-Reward-Ratio: 1:{RRR}")

    data_by_symbol = {}

    for symbol in SYMBOLS:
        print(f"\nLade Binance-Daten für {symbol}...")
        df = fetch_binance_klines(symbol, INTERVAL, LIMIT)
        data_by_symbol[symbol] = df

        print(
            f"{len(df)} Kerzen geladen. Zeitraum: "
            f"{df['date'].iloc[0]} bis {df['date'].iloc[-1]}"
        )

    trades_df, equity_curve, final_capital = backtest(data_by_symbol)

    print_table(trades_df, "TRADE-TABELLE")

    summary_df = performance_summary(trades_df, equity_curve, final_capital)

    print_table(summary_df, "PERFORMANCE-ZUSAMMENFASSUNG")

    print("\nBacktest abgeschlossen.")
    print(f"Endkapital: {final_capital:.2f} EUR")
    print(f"Gesamtrendite: {(final_capital / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio(equity_curve):.2f}")


if __name__ == "__main__":
    main()