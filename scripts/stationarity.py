# /// script
# requires-python = ">=3.10"
# dependencies = ["statsmodels>=0.14", "arch>=6.0", "pandas>=2.0", "numpy>=1.24", "scipy>=1.10"]
# ///
"""stationarity.py — unit-root / stationarity battery + Engle-Granger cointegration.

The pre-flight every time-series model needs. A price with a UNIT ROOT (it is I(1), a random walk
with drift) has no fixed level to revert to, and regressing one such series on another manufactures
fake significance (spurious regression, Granger & Newbold 1974). This answers three questions with
real hypothesis tests:

  1. Is this series stationary?            -> ADF + KPSS + Phillips-Perron  (three tests, one verdict)
  2. If not, how many diffs make it so?    -> integration order d           (the "I" in ARIMA)
  3. Are these TWO series cointegrated?     -> Engle-Granger                 (=> spread is tradeable)

ADF null  = "has a unit root" (non-stationary); small p => STATIONARY.  (Dickey & Fuller 1979)
KPSS null = "is stationary";                     small p => NON-stationary. (KPSS 1992)  Opposite
null on purpose: agreement of the two is far stronger than either alone. PP (Phillips-Perron 1988)
corroborates ADF, robust to the heteroskedasticity + serial correlation financial data always has.

What this does NOT do: predict direction, or size a trade. It says whether a mean exists to revert to.

Usage:
  uv run scripts/stationarity.py --csv prices.csv                 # date + close columns
  uv run scripts/stationarity.py --csv A.csv --coint B.parquet    # Engle-Granger pairs test
  uv run scripts/stationarity.py --csv prices.csv --json          # machine-readable
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, coint

warnings.filterwarnings("ignore")

try:
    from arch.unitroot import PhillipsPerron
    _HAVE_ARCH = True
except Exception:
    _HAVE_ARCH = False


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


def adf_test(x, alpha=0.05):
    stat, p, usedlag, nobs, crit, _ = adfuller(x.dropna(), autolag="AIC")
    return {"test": "ADF", "stat": float(stat), "p": float(p), "lags": int(usedlag), "n": int(nobs),
            "stationary": bool(p < alpha), "null": "unit root (non-stationary)"}


def kpss_test(x, alpha=0.05, regression="c"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, lags, crit = kpss(x.dropna(), regression=regression, nlags="auto")
    return {"test": "KPSS", "stat": float(stat), "p": float(p), "lags": int(lags),
            "stationary": bool(p >= alpha), "null": "stationary"}


def pp_test(x, alpha=0.05):
    if not _HAVE_ARCH:
        return None
    pp = PhillipsPerron(x.dropna())
    return {"test": "Phillips-Perron", "stat": float(pp.stat), "p": float(pp.pvalue), "lags": int(pp.lags),
            "stationary": bool(pp.pvalue < alpha), "null": "unit root (non-stationary)"}


def combined_verdict(x, alpha=0.05):
    a, k, pp = adf_test(x, alpha), kpss_test(x, alpha), pp_test(x, alpha)
    adf_s, kpss_s = a["stationary"], k["stationary"]
    pp_s = pp["stationary"] if pp else adf_s
    if adf_s and kpss_s:
        v = "STATIONARY (I(0)) -- ADF & KPSS agree"
    elif (not adf_s) and (not kpss_s):
        v = "NON-STATIONARY (unit root, I(1)+) -- ADF & KPSS agree"
    elif adf_s and not kpss_s:
        v = "TREND-STATIONARY / borderline -- ADF rejects unit root but KPSS rejects stationarity"
    else:
        v = "INCONCLUSIVE -- tests disagree (likely near a unit root)"
    return {"verdict": v, "stationary_votes": int(adf_s) + int(kpss_s) + int(pp_s),
            "tests": [t for t in (a, k, pp) if t]}


def integration_order(x, max_d=2, alpha=0.05):
    s = x.dropna()
    for d in range(max_d + 1):
        if len(s) < 20:
            break
        if adfuller(s, autolag="AIC")[1] < alpha:
            return d
        s = s.diff().dropna()
    return None


def cointegration(y, x, alpha=0.05):
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1, sort=True).dropna()
    t, p, crit = coint(j["y"], j["x"], trend="c", autolag="AIC")
    beta = float(np.polyfit(j["x"], j["y"], 1)[0])
    spread = j["y"] - beta * j["x"]
    return {"coint_t": float(t), "p": float(p), "cointegrated": bool(p < alpha), "hedge_ratio": beta,
            "n": int(len(j)), "spread_adf_p": adf_test(spread, alpha)["p"]}


def _fmt(t):
    return "  %-16s stat=%9.3f  p=%.4f  -> %s" % (
        t["test"], t["stat"], t["p"], "stationary" if t["stationary"] else "NON-stationary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--price-col", default="close")
    ap.add_argument("--coint", default=None, help="second series CSV/parquet -> Engle-Granger pairs test")
    ap.add_argument("--max-d", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    px = load_series(a.csv, a.price_col)
    ret = np.log(px / px.shift(1)).dropna()
    lv, rv = combined_verdict(px, a.alpha), combined_verdict(ret, a.alpha)
    d = integration_order(px, a.max_d, a.alpha)
    coint_res = cointegration(px, load_series(a.coint, a.price_col), a.alpha) if a.coint else None

    if a.json:
        out = {"as_of": str(px.index[-1])[:10], "n_obs": int(len(px)),
               "levels": {"verdict": lv["verdict"], "tests": lv["tests"]},
               "log_returns": {"verdict": rv["verdict"], "tests": rv["tests"]},
               "integration_order_d": d,
               "note": "Descriptive stationarity tests -- say whether a mean exists to revert to, not a trade signal."}
        if coint_res:
            out["cointegration"] = coint_res
        print(json.dumps(out, indent=2, default=float))
        return

    print("stationarity battery | %s..%s | %d obs | alpha=%.2f  (arch/PP=%s)"
          % (str(px.index[0])[:10], str(px.index[-1])[:10], len(px), a.alpha, _HAVE_ARCH))
    print("\n[LEVELS]  (raw price)")
    for t in lv["tests"]:
        print(_fmt(t))
    print("  VERDICT:", lv["verdict"])
    print("\n[LOG-RETURNS]")
    for t in rv["tests"]:
        print(_fmt(t))
    print("  VERDICT:", rv["verdict"])
    print("\n[INTEGRATION ORDER]  d = %s  %s" % (
        d, "(price is I(%d); ARIMA d-term = %d)" % (d, d) if d is not None
        else "(still non-stationary after %d diffs -- check for structural breaks)" % a.max_d))
    if coint_res:
        c = coint_res
        print("\n[COINTEGRATION Engle-Granger]  %s ~ %s  | n=%d" % (Path(a.csv).stem, Path(a.coint).stem, c["n"]))
        print("  coint_t=%.3f  p=%.4f  hedge_ratio=%.4f  spread_ADF_p=%.4f  -> %s"
              % (c["coint_t"], c["p"], c["hedge_ratio"], c["spread_adf_p"],
                 "COINTEGRATED (spread tradeable)" if c["cointegrated"] else "not cointegrated"))


if __name__ == "__main__":
    main()
