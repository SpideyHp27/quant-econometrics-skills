# /// script
# requires-python = ">=3.10"
# dependencies = ["statsmodels>=0.14", "arch>=6.0", "pandas>=2.0", "numpy>=1.24", "scipy>=1.10"]
# ///
"""accuracy_audit.py -- the jar audits itself.

Every skill is tested against SYNTHETIC data where the true answer is known BY CONSTRUCTION.
A skill passes only if it recovers the planted truth AND stays quiet on planted noise.
Also times each tool on realistic input sizes. Run: uv run scripts/accuracy_audit.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = []


def check(name, cond, detail):
    RESULTS.append((name, bool(cond), detail))
    print("  %-58s %s  %s" % (name, "PASS" if cond else "FAIL", detail))


def bdays(n):
    return pd.date_range("2015-01-01", periods=n, freq="B")


def main():
    rng = np.random.default_rng(7)
    t0 = time.time()

    # ---------------- stationarity-tests ------------------------------------------------
    print("[stationarity-tests] planted truths: RW=I(1), AR(0.5)=I(0), built cointegration")
    import stationarity as S
    rw = pd.Series(100 + np.cumsum(rng.normal(0, 1, 1500)), index=bdays(1500))
    v = S.combined_verdict(rw)
    check("random walk read as NON-stationary", "NON-STATIONARY" in v["verdict"], v["verdict"][:40])
    ar = np.zeros(1500)
    for t in range(1, 1500):
        ar[t] = 0.5 * ar[t - 1] + rng.normal()
    v2 = S.combined_verdict(pd.Series(ar, index=bdays(1500)))
    check("AR(1) phi=0.5 read as STATIONARY", v2["verdict"].startswith("STATIONARY"), v2["verdict"][:40])
    check("integration order of RW = 1", S.integration_order(rw) == 1, f"d={S.integration_order(rw)}")
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, 1500)), index=bdays(1500))
    y = 2 * x + rng.normal(0, 1.5, 1500)
    co = S.cointegration(y, x)
    check("built cointegration detected, hedge~2", co["cointegrated"] and abs(co["hedge_ratio"] - 2) < 0.1,
          f"p={co['p']:.4f} hedge={co['hedge_ratio']:.2f}")
    xn = pd.Series(100 + np.cumsum(rng.normal(0, 1, 1500)), index=bdays(1500))
    co2 = S.cointegration(xn, x)
    check("independent RWs NOT cointegrated", not co2["cointegrated"], f"p={co2['p']:.4f}")

    # ---------------- garch-volatility ---------------------------------------------------
    print("[garch-volatility] planted truth: simulated GARCH(1,1) w=.05 a=.10 b=.85")
    import garch_forecast as G
    n = 1800; sig2 = np.zeros(n); r = np.zeros(n)
    sig2[0] = 0.05 / (1 - 0.10 - 0.85)
    for t in range(1, n):
        sig2[t] = 0.05 + 0.10 * r[t - 1] ** 2 + 0.85 * sig2[t - 1]
        r[t] = np.sqrt(sig2[t]) * rng.normal()
    px = pd.Series(1000 * np.exp(np.cumsum(r / 100)), index=bdays(n))
    tg = time.time()
    wf = G.walk_forward_garch(px, min_train=500, refit_every=60)
    garch_secs = time.time() - tg
    true_sig = pd.Series(np.sqrt(sig2) / 100, index=bdays(n)).reindex(wf.index)
    corr = float(np.corrcoef(wf["sigma_daily"], true_sig)[0, 1])
    check("forecast vol tracks TRUE vol (corr>0.85)", corr > 0.85, f"corr={corr:.3f}")
    iid = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.01, 1200))), index=bdays(1200))
    wf2 = G.walk_forward_garch(iid, min_train=500, refit_every=60)
    spread = float(wf2["sigma_ann"].std() / wf2["sigma_ann"].mean())
    check("on IID returns forecast vol ~flat (cv<0.15)", spread < 0.15, f"cv={spread:.3f}")

    # ---------------- ts-decomposition ---------------------------------------------------
    print("[ts-decomposition] planted truth: +25bp Mondays in white noise; no TOM effect")
    import seasonality as SE
    idx = bdays(1500)
    r3 = rng.normal(0, 0.01, 1500) + np.where(idx.dayofweek == 0, 0.0025, 0.0)
    mon = SE.cell(pd.Series(r3, index=idx)[idx.dayofweek == 0])
    thu = SE.cell(pd.Series(r3, index=idx)[idx.dayofweek == 3])
    check("planted Monday found (p<0.05)", mon["p_hac"] < 0.05, f"Mon p={mon['p_hac']}")
    check("clean Thursday stays quiet (p>0.05)", thu["p_hac"] > 0.05, f"Thu p={thu['p_hac']}")
    tm = SE.tom_mask(idx)
    tomc = SE.cell(pd.Series(r3, index=idx)[tm])
    check("no false TOM on noise (p>0.05)", tomc["p_hac"] > 0.05, f"TOM p={tomc['p_hac']}")

    # ---------------- arima-forecast ------------------------------------------------------
    print("[arima-forecast] planted truth: AR(1) phi=0.6 in returns -> must beat naive")
    import arima_forecast as AF
    ret = np.zeros(1000)
    for t in range(1, 1000):
        ret[t] = 0.6 * ret[t - 1] + rng.normal(0, 0.01)
    pxa = pd.Series(100 * np.exp(np.cumsum(ret)), index=bdays(1000))
    order, _ = AF.pick_order(np.log(pxa).iloc[:700])
    check("order picks AR terms (p>=1)", order[0] >= 1 or order[2] >= 1, f"order={order}")
    ta = time.time()
    preds = AF.walk_forward(np.log(pxa), order, 200, refit=50)
    arima_secs = time.time() - ta
    actual = np.log(pxa).iloc[-200:].to_numpy(); prev = np.log(pxa).iloc[-201:-1].to_numpy()
    skill = float(np.sqrt(np.mean((prev - actual) ** 2)) / np.sqrt(np.mean((preds - actual) ** 2)) - 1)
    check("beats naive on planted AR(1) (skill>5%)", skill > 0.05, f"skill={100*skill:.1f}%")

    # ---------------- regression-diagnostics ----------------------------------------------
    print("[regression-diagnostics] planted: heteroskedasticity, AR errors, collinearity")
    import regression_diag as RD
    n4 = 800; x4 = rng.normal(0, 1, n4)
    y4 = 1 + 2 * x4 + rng.normal(0, 1, n4) * (0.5 + np.abs(x4))          # planted heterosk.
    _, tests, viol, _ = RD.diagnose(y4, x4.reshape(-1, 1), ["x"])
    check("planted heteroskedasticity caught", "heteroskedasticity" in viol,
          f"BP p={tests['breusch_pagan_p']}")
    e = np.zeros(n4)
    for t in range(1, n4):
        e[t] = 0.6 * e[t - 1] + rng.normal()
    y5 = 1 + 2 * x4 + e                                                   # planted AR errors
    _, tests5, viol5, _ = RD.diagnose(y5, x4.reshape(-1, 1), ["x"])
    check("planted serial correlation caught", "serial correlation" in viol5,
          f"BG p={tests5['breusch_godfrey_p']} DW={tests5['durbin_watson']}")
    x2 = x4 + rng.normal(0, 0.05, n4)                                     # planted collinearity
    _, tests6, viol6, _ = RD.diagnose(y4, np.column_stack([x4, x2]), ["x1", "x2"])
    check("planted multicollinearity caught (VIF>10)", "multicollinearity" in viol6,
          f"VIF={tests6['vif']}")
    yc = 1 + 2 * x4 + rng.normal(0, 1, n4)                                # clean control
    _, tests7, viol7, _ = RD.diagnose(yc, x4.reshape(-1, 1), ["x"])
    check("clean regression stays clean", len(viol7) == 0, f"violations={viol7}")

    # ---------------- randomness-tests ----------------------------------------------------
    print("[randomness-tests] planted: IID passes; AR(1) flagged; GARCH vol-clustering flagged")
    import randomness as RN
    b_iid = RN.battery(rng.normal(0, 1, 2000), "iid")
    check("IID noise -> no mean structure", not b_iid["mean_structure"], b_iid["archetype"][:30])
    b_ar = RN.battery(ret, "ar1")
    check("AR(1) -> structure + momentum flavor", b_ar["mean_structure"] and "MOMENTUM" in b_ar["archetype"],
          f"VR2={b_ar['variance_ratio']['q2']['vr']}")
    b_g = RN.battery(r / 100, "garch")
    check("GARCH returns -> vol clustering flagged", b_g["vol_clustering"],
          f"|r| LB p={b_g['vol_clustering_p']}")
    check("GARCH returns -> mean stays unstructured", not b_g["mean_structure"], "")

    # ---------------- swarm-optimizer ------------------------------------------------------
    print("[swarm-optimizer] planted: 1 real-Sharpe asset among 4 noise assets")
    import swarm_optimizer as SW
    n6 = 1200
    good = rng.normal(0.0008, 0.01, n6)                                    # Sharpe ~1.3
    mat = np.column_stack([good] + [rng.normal(0, 0.01, n6) for _ in range(4)])
    ts_ = time.time()
    w, trials = SW.pso(mat[:800], wmax=0.35, particles=30, iters=60, lam_dd=0.02, seed=3)
    swarm_secs = time.time() - ts_
    check("weight concentrates on the real asset (>=0.30)", w[0] >= 0.30, f"w={np.round(w,2).tolist()}")
    check("cap respected (max<=0.35+eps)", w.max() <= 0.351, f"max={w.max():.3f}")

    # ---------------- timing summary --------------------------------------------------------
    total = time.time() - t0
    print()
    print("[timing] garch walk-forward 1800 obs: %.1fs | arima 200-step wf: %.1fs | pso 30x60: %.1fs | full audit: %.1fs"
          % (garch_secs, arima_secs, swarm_secs, total))
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print("\nAUDIT: %d/%d PASS" % (npass, len(RESULTS)))
    if npass < len(RESULTS):
        for nm, ok, det in RESULTS:
            if not ok:
                print("  FAILED:", nm, det)
        sys.exit(1)


if __name__ == "__main__":
    main()
