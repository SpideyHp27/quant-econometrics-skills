# /// script
# requires-python = ">=3.10"
# dependencies = ["statsmodels>=0.14", "pandas>=2.0", "numpy>=1.24", "scipy>=1.10"]
# ///
"""arima_forecast.py -- Box-Jenkins ARIMA, walk-forward, judged against the random walk.

The honest question for markets is not "can ARIMA fit?" (it always fits) but "does it FORECAST
better than the naive random walk?" (Fama says usually no). This tool:
  1. Picks d by ADF on the TRAIN window only; picks (p,q)<=(3,3) by AIC on train only.
  2. Walk-forward 1-step forecasts across the TEST window (refit every --refit bars, zero look-ahead).
  3. Judges: RMSE vs naive (no-change) forecast + directional accuracy vs 50% + Ljung-Box on
     residuals (did the model capture the autocorrelation it claims?).
  4. PROOF HARNESS: trade sign(forecast return) with costs vs buy & hold -- the expected result on
     liquid markets is an HONEST NULL, and the tool says so plainly when it finds one.

Usage: uv run scripts/arima_forecast.py --csv prices.csv [--test-frac .3] [--refit 20] [--cost-bp 1] [--json]
"""
import argparse, json, warnings
from math import sqrt
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")


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


def pick_order(y_train, max_p=3, max_q=3):
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA
    d = 0; s = y_train.copy()
    while d < 2 and adfuller(s.dropna(), autolag="AIC")[1] >= 0.05:
        s = s.diff(); d += 1
    best = (0, d, 0); best_aic = np.inf
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == q == 0:
                continue
            try:
                aic = ARIMA(y_train, order=(p, d, q)).fit().aic
                if aic < best_aic:
                    best_aic, best = aic, (p, d, q)
            except Exception:
                pass
    return best, round(float(best_aic), 1)


def walk_forward(y, order, n_test, refit=20):
    """1-step-ahead forecasts of y over the last n_test points. Zero look-ahead.
    Positional index internally: calendar gaps (holidays) break statsmodels' append."""
    from statsmodels.tsa.arima.model import ARIMA
    y = pd.Series(np.asarray(y, float))                     # RangeIndex
    preds = np.full(n_test, np.nan)
    res = None
    for i in range(n_test):
        t = len(y) - n_test + i
        if res is None or i % refit == 0:
            res = ARIMA(y.iloc[:t], order=order).fit()
        else:
            res = res.append(y.iloc[t - 1:t], refit=False)
        preds[i] = res.forecast(1).iloc[0]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--price-col", default="close")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--refit", type=int, default=20)
    ap.add_argument("--cost-bp", type=float, default=1.0)
    ap.add_argument("--max-p", type=int, default=3); ap.add_argument("--max-q", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    px = load_series(a.csv, a.price_col)
    logp = np.log(px)
    n_test = int(len(logp) * a.test_frac)
    train = logp.iloc[:-n_test]

    order, aic = pick_order(train, a.max_p, a.max_q)
    preds = walk_forward(logp, order, n_test, a.refit)

    actual = logp.iloc[-n_test:].to_numpy()
    prev = logp.iloc[-n_test - 1:-1].to_numpy()
    rmse_arima = float(np.sqrt(np.mean((preds - actual) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((prev - actual) ** 2)))
    fret, aret = preds - prev, actual - prev                      # forecast vs realized log-returns
    live = np.abs(fret) > 1e-12
    dir_acc = float((np.sign(fret[live]) == np.sign(aret[live])).mean()) if live.any() else 0.5

    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_p = float(acorr_ljungbox(aret - fret, lags=[10], return_df=True)["lb_pvalue"].iloc[0])

    strat = np.sign(fret) * aret - np.abs(np.diff(np.concatenate([[0], np.sign(fret)]))) * a.cost_bp / 1e4
    def perf(r):
        r = np.asarray(r, float); sd = r.std()
        sh = float(r.mean() / sd * sqrt(252)) if sd > 0 else 0.0
        return {"ann_ret_pct": round(float(r.mean()) * 252 * 100, 1), "sharpe": round(sh, 2)}
    h = {"arima_sign_trading": perf(strat), "buyhold_test": perf(aret)}

    skill = rmse_naive / rmse_arima - 1
    verdict = ("BEATS the random walk (rare -- check for data quirks before believing)"
               if skill > 0.01 and dir_acc > 0.53 else
               "does NOT beat the random walk (the expected honest null on liquid markets)")
    out = {"order_pdq": list(order), "train_aic": aic, "n_test": n_test,
           "test_span": f"{str(px.index[-n_test])[:10]} -> {str(px.index[-1])[:10]}",
           "rmse_arima": round(rmse_arima, 6), "rmse_naive": round(rmse_naive, 6),
           "rmse_skill_pct": round(100 * skill, 2), "directional_acc_pct": round(100 * dir_acc, 1),
           "resid_ljungbox_p": round(lb_p, 4), "harness": h, "verdict": verdict,
           "note": "Order chosen on train only; forecasts walk-forward; judged vs naive no-change."}
    if a.json:
        print(json.dumps(out, indent=2)); return
    print("ARIMA%s walk-forward | test %s (%d bars, refit %d)" % (tuple(order), out["test_span"], n_test, a.refit))
    print("  RMSE arima %.6f vs naive %.6f  -> skill %+.2f%%" % (rmse_arima, rmse_naive, 100 * skill))
    print("  directional accuracy %.1f%% (benchmark 50%%) | resid Ljung-Box p=%.3f (want >0.05)"
          % (100 * dir_acc, lb_p))
    print("  harness: sign-trading %s | buy&hold %s" % (h["arima_sign_trading"], h["buyhold_test"]))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
