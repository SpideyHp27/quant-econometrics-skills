# /// script
# requires-python = ">=3.10"
# dependencies = ["statsmodels>=0.14", "pandas>=2.0", "numpy>=1.24", "scipy>=1.10"]
# ///
"""seasonality.py -- time-series decomposition + calendar-edge detection + the honest TOM harness.

Three questions, one tool:
  1. DECOMPOSITION -- how much of this series is trend vs seasonal vs noise? (STL, Cleveland 1990)
     For markets the honest answer is usually "seasonality is ~0% of VARIANCE" -- but:
  2. CALENDAR CELLS -- tiny MEAN effects at specific calendar points can still be tradeable.
     Day-of-week (French 1980) and turn-of-month (Ariel 1987; Lakonishok & Smidt 1988) cells,
     each with a Newey-West HAC p-value. No cherry-picking: every cell is printed.
  3. PROOF HARNESS -- the classic turn-of-month strategy (long ONLY the last session and first 3
     sessions of each month, flat otherwise) vs buy & hold, with a pre-registered IS/OOS split.
     ~19% time in market; if it keeps a large share of B&H return, the calendar edge is real.

Zero look-ahead: calendar membership is known in advance by definition; the strategy holds
positions only through pre-scheduled windows. What this does NOT do: predict direction beyond
the calendar tendency; find YOUR instrument's best window (that would be mining -- the TOM window
is fixed from the 1987/1988 literature, not fitted).

Usage:
  uv run scripts/seasonality.py --csv prices.csv [--split 0.7] [--json]
"""
import argparse, json
from math import erf, sqrt
from pathlib import Path
import numpy as np, pandas as pd


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


def hac_p(x):
    """Newey-West HAC p for mean!=0 (Bartlett, L=0.75*n^(1/3))."""
    x = np.asarray(x, float); n = len(x)
    if n < 12:
        return None
    m = x.mean(); L = max(1, int(0.75 * n ** (1 / 3)))
    g0 = ((x - m) ** 2).mean(); s = g0
    for k in range(1, L + 1):
        gk = ((x[:-k] - m) * (x[k:] - m)).mean() * (n - k) / n
        s += 2 * (1 - k / (L + 1)) * gk
    se = sqrt(max(s, 1e-18) / n)
    t = m / se if se > 0 else 0.0
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def stl_shares(logpx, period=21):
    """STL variance shares (trend/seasonal/resid) on log price, monthly cycle by default."""
    from statsmodels.tsa.seasonal import STL
    r = STL(logpx, period=period, robust=True).fit()
    var = np.var(np.diff(logpx))
    parts = {"trend": np.var(np.diff(r.trend.to_numpy())),
             "seasonal": np.var(np.diff(r.seasonal.to_numpy())),
             "resid": np.var(np.diff(r.resid.dropna().to_numpy()))}
    tot = sum(parts.values())
    return {k: round(100 * v / tot, 1) for k, v in parts.items()} if tot > 0 else parts


def tom_mask(idx):
    """Turn-of-month: last session of the month + first 3 sessions of the next (T-1..T+3)."""
    per = idx.to_period("M")
    rank_start = pd.Series(idx, index=idx).groupby(per).cumcount()
    n_per = pd.Series(1, index=idx).groupby(per).transform("size")
    rank_end = n_per - 1 - rank_start
    return ((rank_end == 0) | (rank_start <= 2)).to_numpy()


def cell(r):
    r = np.asarray(r, float)
    if len(r) < 12:
        return None
    ann = float(r.mean()) * 252
    return {"n": int(len(r)), "mean_bp": round(1e4 * float(r.mean()), 2),
            "ann_pct": round(100 * ann, 1), "p_hac": round(hac_p(r), 4)}


def perf(r, ann=252):
    r = pd.Series(np.asarray(r, float))
    sd = r.std()
    sharpe = float(r.mean() / sd * sqrt(ann)) if sd > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((1 - eq / eq.cummax()).max() * 100)
    yrs = len(r) / ann
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
    return {"cagr_pct": round(cagr, 1), "sharpe": round(sharpe, 2), "maxdd_pct": round(dd, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--price-col", default="close")
    ap.add_argument("--split", type=float, default=0.7, help="IS fraction (pre-registered)")
    ap.add_argument("--stl-period", type=int, default=21)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    px = load_series(a.csv, a.price_col)
    ret = np.log(px / px.shift(1)).dropna()
    idx = ret.index
    out = {"as_of": str(idx[-1])[:10], "n_obs": int(len(ret)),
           "span": f"{str(idx[0])[:10]} -> {str(idx[-1])[:10]}"}

    # [1] decomposition -- with a shuffled-returns NULL so the pseudo-cycle STL always finds
    # at any period doesn't get read as real seasonality
    out["stl_variance_shares_pct"] = stl_shares(np.log(px), a.stl_period)
    rng = np.random.default_rng(42)
    sh = np.log(px.iloc[0]) + np.concatenate([[0], np.cumsum(rng.permutation(ret.to_numpy()))])
    null_share = stl_shares(pd.Series(sh, index=px.index), a.stl_period)["seasonal"]
    out["stl_seasonal_null_pct"] = null_share
    out["stl_seasonal_excess_pct"] = round(float(out["stl_variance_shares_pct"]["seasonal"]) - float(null_share), 1)

    # [2] calendar cells -- ALL printed, none hidden
    dows = {}
    for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        dows[name] = cell(ret[idx.dayofweek == d])
    tm = tom_mask(idx)
    out["day_of_week"] = dows
    out["tom_cell"] = cell(ret[tm]); out["non_tom_cell"] = cell(ret[~tm])
    out["tom_share_pct"] = round(100 * tm.mean(), 1)

    # [3] TOM harness vs buy&hold, IS/OOS
    strat = ret.where(pd.Series(tm, index=idx), 0.0)
    k = int(len(ret) * a.split)
    out["harness"] = {
        "buyhold_full": perf(ret), "tom_full": perf(strat),
        "buyhold_IS": perf(ret.iloc[:k]), "tom_IS": perf(strat.iloc[:k]),
        "buyhold_OOS": perf(ret.iloc[k:]), "tom_OOS": perf(strat.iloc[k:]),
        "split_date": str(idx[k])[:10]}
    tomc, ntc = out["tom_cell"], out["non_tom_cell"]
    verdict = ("CALENDAR EDGE: TOM window carries the returns"
               if tomc and ntc and tomc["p_hac"] < 0.05 and tomc["mean_bp"] > 4 * max(ntc["mean_bp"], 0.01)
               else "weak/no calendar edge (honest null)")
    out["verdict"] = verdict
    out["note"] = "TOM window fixed from Ariel 1987/Lakonishok-Smidt 1988 -- NOT fitted to this data."

    if a.json:
        print(json.dumps(out, indent=2, default=str)); return
    print("SEASONALITY | %s | %d obs" % (out["span"], out["n_obs"]))
    print("[1] STL variance shares:", out["stl_variance_shares_pct"], "(markets: seasonal~=0 is normal)")
    print("[2] day-of-week (mean bp/day, HAC p):")
    for k2, v in dows.items():
        if v:
            print("    %-4s %+6.2f bp  p=%s" % (k2, v["mean_bp"], v["p_hac"]))
    print("    TOM  %+6.2f bp  p=%s   | non-TOM %+6.2f bp p=%s  (TOM=%s%% of days)"
          % (tomc["mean_bp"], tomc["p_hac"], ntc["mean_bp"], ntc["p_hac"], out["tom_share_pct"]))
    print("[3] TOM harness (long only T-1..T+3, flat otherwise) vs buy&hold:")
    h = out["harness"]
    for tag in ("full", "IS", "OOS"):
        b, t = h["buyhold_" + tag], h["tom_" + tag]
        print("    %-4s B&H %6.1f%%/yr Sh %.2f DD %5.1f%%   |  TOM %6.1f%%/yr Sh %.2f DD %5.1f%%"
              % (tag, b["cagr_pct"], b["sharpe"], b["maxdd_pct"], t["cagr_pct"], t["sharpe"], t["maxdd_pct"]))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
