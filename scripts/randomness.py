# /// script
# requires-python = ">=3.10"
# dependencies = ["statsmodels>=0.14", "pandas>=2.0", "numpy>=1.24", "scipy>=1.10"]
# ///
"""randomness.py -- is there ANY exploitable structure, or is this white noise?

The gate that should run before any model: if returns are indistinguishable from white noise,
every backtest on them is curve-fitting. Battery (all reported, fused verdict):
  Ljung-Box portmanteau (1978) at lags 5/10/20  -- linear autocorrelation
  Lo-MacKinlay variance ratio (1988), q=2/5/10, heteroskedasticity-robust z
      VR>1 => momentum-flavored, VR<1 => mean-reversion-flavored
  Runs test (Wald-Wolfowitz) on return signs      -- sign clustering
  Ljung-Box on |r| (proxy ARCH test)              -- volatility clustering (GARCH's food)
FUNCTIONAL OUTPUT: a strategy-archetype selector -- per series it answers "momentum / mean-rev /
nothing (in the mean); vol-structure yes/no". Feed multiple files to scan a universe.

Usage: uv run scripts/randomness.py --csv A.parquet [B.csv ...] [--returns-col] [--json]
"""
import argparse, json, warnings
from math import erf, sqrt
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")


def load_returns(path, price_col="close"):
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
    s = df[pc].astype(float).dropna().sort_index()
    return np.log(s / s.shift(1)).dropna()


def variance_ratio(r, q):
    """Lo-MacKinlay VR(q) with heteroskedasticity-robust z (their eq. A2/A3)."""
    r = np.asarray(r, float); n = len(r); mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = pd.Series(r).rolling(q).sum().dropna().to_numpy()
    m = (n - q + 1) * (1 - q / n)
    varq = ((rq - q * mu) ** 2).sum() / m
    vr = varq / (q * var1)
    # robust z
    dsq = (r - mu) ** 2
    theta = 0.0
    for k in range(1, q):
        dk = (dsq[k:] * dsq[:-k]).sum() / (dsq.sum() ** 2 / n)   # delta_k
        theta += (2 * (q - k) / q) ** 2 * dk
    z = (vr - 1) / sqrt(theta) if theta > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return round(float(vr), 3), round(float(z), 2), round(float(p), 4)


def runs_test(r):
    s = np.sign(np.asarray(r, float)); s = s[s != 0]
    n1, n2 = (s > 0).sum(), (s < 0).sum()
    runs = 1 + (np.diff(s) != 0).sum()
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / sqrt(var) if var > 0 else 0.0
    return round(float(z), 2), round(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))), 4)


def battery(r, name):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(r, lags=[5, 10, 20], return_df=True)["lb_pvalue"]
    lb_abs = acorr_ljungbox(np.abs(r), lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    vrs = {f"q{q}": variance_ratio(r, q) for q in (2, 5, 10)}
    rz, rp = runs_test(r)
    mean_struct = (lb < 0.05).any() or any(v[2] < 0.05 for v in vrs.values()) or rp < 0.05
    vr2 = vrs["q2"][0]
    if not mean_struct:
        archetype = "NOTHING in the mean (white-noise-like) -- do not fit directional models"
    elif vr2 > 1.02:
        archetype = "MOMENTUM-flavored (VR>1): trend/continuation archetypes"
    elif vr2 < 0.98:
        archetype = "MEAN-REVERSION-flavored (VR<1): fade/reversion archetypes"
    else:
        archetype = "weak mixed structure"
    return {"series": name, "n": int(len(r)),
            "ljung_box_p": {f"lag{k}": round(float(v), 4) for k, v in zip([5, 10, 20], lb)},
            "variance_ratio": {k: {"vr": v[0], "z": v[1], "p": v[2]} for k, v in vrs.items()},
            "runs": {"z": rz, "p": rp}, "vol_clustering_p": round(float(lb_abs), 4),
            "vol_clustering": bool(lb_abs < 0.05), "mean_structure": bool(mean_struct),
            "archetype": archetype}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True); ap.add_argument("--price-col", default="close")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = [battery(load_returns(f, a.price_col), Path(f).stem) for f in a.csv]
    if a.json:
        print(json.dumps(out, indent=2)); return
    for b in out:
        print("%s | n=%d" % (b["series"], b["n"]))
        print("  Ljung-Box p:", b["ljung_box_p"], "| runs p=%s" % b["runs"]["p"])
        print("  VR:", {k: v["vr"] for k, v in b["variance_ratio"].items()},
              "p:", {k: v["p"] for k, v in b["variance_ratio"].items()})
        print("  vol clustering: %s (p=%s)  <- GARCH's food" % (b["vol_clustering"], b["vol_clustering_p"]))
        print("  ARCHETYPE:", b["archetype"])
        print()


if __name__ == "__main__":
    main()
