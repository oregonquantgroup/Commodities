"""
Publishes the GPRD-Gold backtest as JSON for the OQG website.

Pulls fresh inputs (Iacoviello's daily GPR index, gold and S&P 500 closes from
Yahoo), runs the same signal -> state machine -> backtest pipeline as
gprd_gold_strategy.py, and writes site/gprd_gold.json.

Run on a schedule by .github/workflows/gprd-site-data.yml; safe to run by hand.
"""
import json
import math
from pathlib import Path

import pandas as pd
import yfinance as yf

from gprd_gold_strategy import build_signals, state_machine, backtest

GPRD_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta"
GPRD_START = "2008-01-01"   # z-score history the research workbook starts from
TRADE_START = "2010-01-01"  # its gold and S&P series begin here, so the backtest does too
RF = 0.0274                 # same risk-free rate the research uses
OUT = Path(__file__).parent / "site" / "gprd_gold.json"


def load_gprd():
    g = pd.read_stata(GPRD_URL)
    g["Date"] = pd.to_datetime(g["DAY"].astype(str), format="%Y%m%d")
    g = g.loc[g["Date"] >= GPRD_START, ["Date", "GPRD"]].dropna()
    return g.sort_values("Date").reset_index(drop=True)


def load_close(ticker, col):
    s = yf.Ticker(ticker).history(start=TRADE_START, auto_adjust=False)["Close"]
    s.index = s.index.tz_localize(None).normalize()
    return s.rename(col).rename_axis("Date").reset_index().dropna()


def check(df, name, max_age_days=10):
    # fail loudly instead of publishing a broken or stale chart — the site keeps the last good file
    if df.empty:
        raise SystemExit(f"{name}: no rows returned")
    age = (pd.Timestamp.now().normalize() - df["Date"].max()).days
    if age > max_age_days:
        raise SystemExit(f"{name}: newest row is {age} days old")


def num(x, digits):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), digits)


def summary(value, ret):
    # same formulas as metrics() in gprd_gold_strategy.py, kept numeric for the site
    years = len(value) / 252
    growth = value.iloc[-1] / value.iloc[0]
    rd = (1 + RF) ** (1 / 252) - 1
    r = ret.dropna()
    return {
        "totalReturn": num(growth - 1, 4),
        "cagr": num(growth ** (1 / years) - 1, 4),
        "sharpe": num((r.mean() - rd) / r.std() * math.sqrt(252), 3),
        "maxDrawdown": num((value / value.cummax() - 1).min(), 4),
    }


def main():
    gprd = load_gprd()
    gold = load_close("GC=F", "Gold_Close")
    sp = load_close("^GSPC", "SP500_Level")
    for df, name in [(gprd, "GPRD"), (gold, "Gold"), (sp, "S&P 500")]:
        check(df, name)

    # The research workbook keeps every calendar day of GPRD through 2009, then only
    # trading days. Match it, so the 504-row z-score window means the same thing.
    gprd = gprd[(gprd["Date"] < TRADE_START) | gprd["Date"].isin(gold["Date"])].reset_index(drop=True)

    sigs = state_machine(build_signals(gprd, gold["Date"]))
    res = backtest(gold, sp, sigs)

    res["Strategy_DD"] = res["Strategy_Value"] / res["Strategy_Value"].cummax() - 1
    res["Gold_DD"] = res["Gold_BH_Value"] / res["Gold_BH_Value"].cummax() - 1
    res["Day"] = res["Date"]

    # one point per week keeps the file small; drawdowns keep the week's worst day
    wk = (
        res.groupby(pd.Grouper(key="Date", freq="W-FRI"))
        .agg({
            "Day": "last",
            "Strategy_Value": "last", "Gold_BH_Value": "last", "SP500_BH_Value": "last",
            "Strategy_DD": "min", "Gold_DD": "min",
            "Position": "last", "Z_score": "last",
        })
        .dropna(subset=["Day"])
    )

    last = res.iloc[-1]
    payload = {
        "asOf": last["Date"].date().isoformat(),
        "stats": {
            "strategy": summary(res["Strategy_Value"], res["Strategy_Return"]),
            "gold": summary(res["Gold_BH_Value"], res["Gold_1d_return"]),
            "sp500": summary(res["SP500_BH_Value"], res["SP500_1d_return"]),
            "inMarket": num((res["Position"] == 1).mean(), 4),
        },
        "current": {"position": int(last["Position"]), "z": num(last["Z_score"], 2)},
        "series": {
            "date": [d.date().isoformat() for d in wk["Day"]],
            "strategy": [num(v, 2) for v in wk["Strategy_Value"]],
            "gold": [num(v, 2) for v in wk["Gold_BH_Value"]],
            "sp500": [num(v, 2) for v in wk["SP500_BH_Value"]],
            "strategyDD": [num(v, 4) for v in wk["Strategy_DD"]],
            "goldDD": [num(v, 4) for v in wk["Gold_DD"]],
            "position": [int(v) for v in wk["Position"]],
            "z": [num(v, 2) for v in wk["Z_score"]],
        },
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    s = payload["stats"]
    print(f"as of {payload['asOf']}: strategy {s['strategy']['totalReturn']:+.1%}, "
          f"gold {s['gold']['totalReturn']:+.1%}, {len(wk)} weeks -> {OUT}")


if __name__ == "__main__":
    main()
