# /// script
# requires-python = ">=3.10"
# dependencies = ["statsmodels>=0.14", "pandas>=2.0", "numpy>=1.24"]
# ///
"""pairs_backtest.py — walk-forward cointegration pairs / stat-arb strategy + honest backtest.

The FUNCTIONAL payoff of the stationarity skill. Cointegration isn't a verdict, it's a trade: two series
whose spread is mean-reverting can be traded market-neutral (long one leg, short the other) around the
spread's own mean. This builds that strategy and BACKTESTS it — the equivalent of the GARCH skill's
compare.py: it doesn't just diagnose, it produces a decision (long/short spread) and proves the P&L.

Zero look-ahead, the whole point:
  - Hedge ratio beta_t = OLS slope of y on x over the TRAILING window (data before t only).
  - Spread_t = y_t - beta_t * x_t.  z_t = (spread_t - trailing_mean) / trailing_std  (trailing only).
  - Position chosen at the CLOSE of day t is applied to day t+1's returns. Nothing peeks forward.

Signal (classic Bollinger-style spread reversion):
  z > +entry  -> SHORT the spread (short y, long beta*x)   [spread rich, bet it falls]
  z < -entry  -> LONG  the spread (long y, short beta*x)    [spread cheap, bet it rises]
  |z| < exit  -> flat.    |z| > stop -> bail (divergence stop; a broken relationship, not a fade).

Costs modeled per leg turnover (bps of notional). Reports Sharpe / MaxDD / profit factor / CAGR /
#trades / win rate — honestly, including when the answer is "no edge here."

Usage:
  uv run scripts/pairs_backtest.py --y ES.parquet --x NQ.parquet
  uv run scripts/pairs_backtest.py --y A.csv --x B.csv --entry 2 --exit 0.5 --stop 4 --win 60 --json
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import coint


def load_series(path, price_col="close"):
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    dcol = next((c for c in ["date", "datetime", "time", "et", "timestamp"] if c in df.columns), None)
    tcol = "time" if ("time" in df.columns and dcol != "time") else None
    if dcol is not None:
        stamp = df[dcol].astype(str) + ((" " + df[tcol].astype(str)) if tcol else "")
        df.index = pd.to_datetime(stamp, errors="coerce", format="mixed")
    pc = price_col.lower()
    if pc not in df.columns:
        pc = "close" if "close" in df.columns else df.select_dtypes("number").columns[-1]
    return df[pc].astype(float).dropna().sort_index()


def stats(daily_ret, ann=252):
    r = daily_ret.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"sharpe": 0.0, "cagr_pct": 0.0, "maxdd_pct": 0.0, "profit_factor": 0.0}
    eq = (1 + r).cumprod()
    dd = 1 - eq / eq.cummax()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"sharpe": round(r.mean() / r.std() * np.sqrt(ann), 2),
            "cagr_pct": round((eq.iloc[-1] ** (ann / len(r)) - 1) * 100, 1),
            "maxdd_pct": round(dd.max() * 100, 1),
            "profit_factor": round(gains / losses, 2) if losses > 0 else float("inf")}


def backtest_pair(y, x, win=60, zwin=60, entry=2.0, exit=0.5, stop=4.0, cost_bps=1.0, ann=252):
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1, sort=True).dropna()
    ry, rx = np.log(j["y"]).diff(), np.log(j["x"]).diff()

    # trailing-window hedge ratio beta_t = cov(y,x)/var(x) over [t-win, t-1]  (causal: shift 1)
    cov = j["y"].rolling(win).cov(j["x"]).shift(1)
    var = j["x"].rolling(win).var().shift(1)
    beta = (cov / var).clip(-10, 10)
    spread = j["y"] - beta * j["x"]
    z = (spread - spread.rolling(zwin).mean()) / spread.rolling(zwin).std()

    # position state machine on z known at close t
    pos = np.zeros(len(j)); cur = 0.0
    zv = z.values
    for t in range(len(j)):
        zt = zv[t]
        if not np.isfinite(zt):
            pos[t] = 0.0; cur = 0.0; continue
        if cur == 0.0:
            if zt > entry:   cur = -1.0
            elif zt < -entry: cur = 1.0
        else:
            if abs(zt) < exit or abs(zt) > stop or np.sign(zt) == -np.sign(cur):
                cur = 0.0
        pos[t] = cur
    pos = pd.Series(pos, index=j.index)

    # spread return per unit, gross-normalized by (1+|beta|); position applied next day (no lookahead)
    spread_ret = (ry - beta * rx) / (1.0 + beta.abs())
    strat = pos.shift(1) * spread_ret
    turnover = pos.diff().abs().fillna(0.0)                       # legs traded on each change
    strat = strat - turnover * (cost_bps / 1e4)                   # transaction cost drag

    trades = int((turnover > 0).sum() // 1)
    entries = int(((pos != 0) & (pos.shift(1) == 0)).sum())
    wins = strat[strat != 0]
    st = stats(strat, ann)
    st.update({"n_entries": entries, "days_in_market_pct": round((pos != 0).mean() * 100, 1),
               "win_day_pct": round((wins > 0).mean() * 100, 1) if len(wins) else 0.0,
               "avg_abs_beta": round(beta.abs().mean(), 3)})
    return st, strat, pos, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", required=True); ap.add_argument("--x", required=True)
    ap.add_argument("--price-col", default="close")
    ap.add_argument("--win", type=int, default=60); ap.add_argument("--zwin", type=int, default=60)
    ap.add_argument("--entry", type=float, default=2.0); ap.add_argument("--exit", type=float, default=0.5)
    ap.add_argument("--stop", type=float, default=4.0); ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--ann", type=int, default=252); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    y, x = load_series(a.y, a.price_col), load_series(a.x, a.price_col)
    j = pd.concat([y, x], axis=1, sort=True).dropna()
    ct, cp, _ = coint(j.iloc[:, 0], j.iloc[:, 1], trend="c", autolag="AIC")
    st, strat, pos, z = backtest_pair(y, x, a.win, a.zwin, a.entry, a.exit, a.stop, a.cost_bps, a.ann)
    yn, xn = Path(a.y).stem, Path(a.x).stem

    if a.json:
        print(json.dumps({"pair": f"{yn}~{xn}", "n_obs": int(len(j)),
                          "full_sample_coint_p": round(float(cp), 4),
                          "params": {"win": a.win, "zwin": a.zwin, "entry": a.entry, "exit": a.exit,
                                     "stop": a.stop, "cost_bps": a.cost_bps},
                          "backtest": st,
                          "note": "Market-neutral spread reversion, walk-forward, zero look-ahead, costs modeled."},
                         indent=2, default=float))
        return

    print("PAIRS BACKTEST  %s ~ %s | %d obs | full-sample coint p=%.4f" % (yn, xn, len(j), cp))
    print("params: win=%d zwin=%d entry=%.1f exit=%.1f stop=%.1f cost=%.1fbps" %
          (a.win, a.zwin, a.entry, a.exit, a.stop, a.cost_bps))
    print("-> Sharpe %.2f | CAGR %.1f%% | MaxDD %.1f%% | PF %.2f | entries %d | in-mkt %.0f%% | win-day %.0f%% | avg|beta| %.2f"
          % (st["sharpe"], st["cagr_pct"], st["maxdd_pct"], st["profit_factor"],
             st["n_entries"], st["days_in_market_pct"], st["win_day_pct"], st["avg_abs_beta"]))
    verdict = ("TRADEABLE EDGE" if st["sharpe"] >= 1.0 and st["profit_factor"] >= 1.3
               else "MARGINAL" if st["sharpe"] >= 0.5 else "NO EDGE (diagnostic was right)")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
