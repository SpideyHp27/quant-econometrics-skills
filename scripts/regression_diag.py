# /// script
# requires-python = ">=3.10"
# dependencies = ["statsmodels>=0.14", "pandas>=2.0", "numpy>=1.24", "scipy>=1.10"]
# ///
"""regression_diag.py -- OLS with the full Gauss-Markov referee battery.

The use case: someone claims "X predicts Y" with a regression p-value. Financial data violates
OLS assumptions almost by default (heteroskedasticity, serial correlation), which makes naive
p-values LIE small. This tool fits the regression and then prosecutes it:

  heteroskedasticity : Breusch-Pagan 1979 + White 1980
  serial correlation : Durbin-Watson + Breusch-Godfrey (5 lags)
  multicollinearity  : VIF per regressor (>10 = trouble)
  normality          : Jarque-Bera
  functional form    : Ramsey RESET
  stability          : Chow break at the sample midpoint
  THE VERDICT COLUMN : naive p vs HAC (Newey-West) p side by side -- if significance only
                       exists in the naive column, the "edge" is an artifact.

Modes:
  --csv data.csv --y colY --x colX1,colX2     generic regression
  --y-prices A.csv --x-prices B.csv [--lag 1] returns-on-lagged-returns predictor test
Usage: uv run scripts/regression_diag.py ... [--json]
"""
import argparse, json, warnings
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


def diagnose(y, X, names):
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_breusch_godfrey, linear_reset
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    hac = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, int(0.75 * len(y) ** (1 / 3)))})

    coefs = []
    for i, nm in enumerate(["const"] + names):
        coefs.append({"var": nm, "beta": round(float(ols.params[i]), 6),
                      "p_naive": round(float(ols.pvalues[i]), 4),
                      "p_hac": round(float(hac.pvalues[i]), 4)})
    resid = ols.resid
    bp = het_breuschpagan(resid, Xc)
    wh = het_white(resid, Xc)
    bg = acorr_breusch_godfrey(ols, nlags=5)
    jb = jarque_bera(resid)
    try:
        reset_p = float(linear_reset(ols, power=2, use_f=True).pvalue)
    except Exception:
        reset_p = None
    Xc_arr = np.asarray(Xc)
    vif = {nm: round(float(variance_inflation_factor(Xc_arr, i + 1)), 2)
           for i, nm in enumerate(names)} if X.shape[1] > 1 else {names[0]: 1.0}
    # Chow at midpoint
    k = len(y) // 2
    rss = lambda yy, xx: float(sm.OLS(yy, xx).fit().ssr)
    rss_p = rss(y, Xc); rss_1 = rss(y[:k], Xc[:k]); rss_2 = rss(y[k:], Xc[k:])
    kp = Xc.shape[1]
    f_chow = ((rss_p - rss_1 - rss_2) / kp) / ((rss_1 + rss_2) / (len(y) - 2 * kp))
    from scipy.stats import f as fdist
    chow_p = float(1 - fdist.cdf(f_chow, kp, len(y) - 2 * kp))

    tests = {"breusch_pagan_p": round(float(bp[1]), 4), "white_p": round(float(wh[1]), 4),
             "durbin_watson": round(float(durbin_watson(resid)), 3),
             "breusch_godfrey_p": round(float(bg[1]), 4), "jarque_bera_p": round(float(jb[1]), 4),
             "reset_p": round(reset_p, 4) if reset_p is not None else None,
             "chow_break_p": round(chow_p, 4), "vif": vif,
             "r2": round(float(ols.rsquared), 4), "n": int(len(y))}
    violations = []
    if tests["breusch_pagan_p"] < 0.05 or tests["white_p"] < 0.05:
        violations.append("heteroskedasticity")
    if tests["breusch_godfrey_p"] < 0.05 or not (1.5 < tests["durbin_watson"] < 2.5):
        violations.append("serial correlation")
    if any(v > 10 for v in vif.values()):
        violations.append("multicollinearity")
    if tests["chow_break_p"] < 0.05:
        violations.append("parameter instability (Chow)")
    sig_naive = [c["var"] for c in coefs[1:] if c["p_naive"] < 0.05]
    sig_hac = [c["var"] for c in coefs[1:] if c["p_hac"] < 0.05]
    only_naive = [v for v in sig_naive if v not in sig_hac]
    verdict = ("ARTIFACT RISK: significance vanishes under HAC for " + ",".join(only_naive)
               if only_naive else
               ("edge survives HAC for " + ",".join(sig_hac) if sig_hac else "no significant predictors (honest null)"))
    return coefs, tests, violations, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--y"); ap.add_argument("--x")
    ap.add_argument("--y-prices"); ap.add_argument("--x-prices"); ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.y_prices:
        yp, xp = load_series(a.y_prices), load_series(a.x_prices)
        j = pd.concat([np.log(yp).diff().rename("y"), np.log(xp).diff().rename("x")], axis=1, sort=True).dropna()
        y = j["y"].iloc[a.lag:].to_numpy()
        X = j["x"].shift(a.lag).dropna().to_numpy().reshape(-1, 1)[:len(y)]
        names = [f"{Path(a.x_prices).stem}_ret_lag{a.lag}"]
        label = f"{Path(a.y_prices).stem}_ret ~ {names[0]}"
    else:
        df = pd.read_csv(a.csv)
        names = a.x.split(",")
        y = df[a.y].astype(float).to_numpy(); X = df[names].astype(float).to_numpy()
        label = f"{a.y} ~ {'+'.join(names)}"

    coefs, tests, violations, verdict = diagnose(y, X, names)
    out = {"model": label, "coefficients": coefs, "tests": tests,
           "violations": violations, "verdict": verdict,
           "note": "Naive vs HAC p side by side; if significance lives only in naive, it is an artifact."}
    if a.json:
        print(json.dumps(out, indent=2)); return
    print("REGRESSION REFEREE | %s | n=%d R2=%.4f" % (label, tests["n"], tests["r2"]))
    for c in coefs:
        print("  %-24s beta %+.6f  p_naive %-7s p_HAC %s" % (c["var"], c["beta"], c["p_naive"], c["p_hac"]))
    print("  tests:", {k: v for k, v in tests.items() if k not in ("vif", "n", "r2")})
    print("  VIF:", tests["vif"])
    print("  violations:", violations or "none")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
