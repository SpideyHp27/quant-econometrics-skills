# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "numpy>=1.24", "scipy>=1.10"]
# ///
"""swarm_optimizer.py — PSO portfolio-weight optimizer with an anti-overfit gauntlet WELDED ON.

Particle Swarm Optimization (Kennedy & Eberhart 1995): a swarm of candidate weight vectors flies
through allocation space, each particle pulled toward its own best find and the swarm's best.
Great at finding optima fast — which is exactly the danger in finance: a stronger optimizer finds
OVERFITS faster. So this tool refuses to report an in-sample answer:

  1. Optimization sees ONLY the TRAIN window (first 1-test_frac of history).
  2. The TEST window (last test_frac) is untouched until the swarm is done — verdict comes from there.
  3. Every candidate evaluation is COUNTED (N_trials = particles x iterations) and the test Sharpe is
     DEFLATED for it: Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) — the probability the
     result beats what the best of N_trials random tries would show.
  4. Equal-weight is the benchmark. If PSO does not beat equal-weight OUT OF SAMPLE, the tool says
     "USE EQUAL WEIGHT" — that honest null is a feature, not a failure.

Input: per-strategy trade CSVs (columns incl. `date` + `profit`) — one file per strategy — or a
wide daily-returns CSV. Output: weights + train/test Sharpe/MaxDD for PSO vs equal-weight + DSR.

Usage:
  uv run scripts/swarm_optimizer.py --dir <folder-of-trade-csvs> [--glob "*.csv"]
  uv run scripts/swarm_optimizer.py --wide daily_returns.csv
  options: --equity 100000 --test-frac 0.30 --wmax 0.35 --particles 40 --iters 120
           --lambda-dd 0.02 --seed 42 --json
"""
import argparse, json, glob as globmod
from math import erf, sqrt, log, exp
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm, skew as sk_, kurtosis as ku_

ANN = 252


def load_dir(d, pat):
    files = sorted(globmod.glob(str(Path(d) / pat)))
    cols = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        if "profit" not in df.columns or "date" not in df.columns:
            continue
        day = pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.normalize()
        pnl = df.groupby(day)["profit"].sum()
        name = Path(f).stem
        cols[name] = pnl
    if not cols:
        raise SystemExit("no usable trade CSVs (need date+profit columns) in " + str(d))
    m = pd.DataFrame(cols).sort_index()
    full = pd.date_range(m.index.min(), m.index.max(), freq="B")
    return m.reindex(full).fillna(0.0)


def stats(ret):
    r = np.asarray(ret, float)
    mu, sd = r.mean(), r.std()
    sharpe = mu / sd * sqrt(ANN) if sd > 0 else 0.0
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = float((1 - eq / peak).max() * 100)
    return sharpe, dd


def dsr(sr_test, n_obs, trial_srs, ret_test):
    """Deflated Sharpe Ratio (Bailey & LdP 2014). trial_srs = train-Sharpes of all trials (proxy
    for the trial distribution). Returns probability the test Sharpe beats the expected max of
    N random trials with that dispersion."""
    N = max(len(trial_srs), 2)
    v = float(np.var(trial_srs))
    if v <= 0:
        v = 1e-9
    g = 0.5772156649
    sr0 = sqrt(v) * ((1 - g) * norm.ppf(1 - 1.0 / N) + g * norm.ppf(1 - 1.0 / (N * np.e)))
    s = float(sk_(ret_test)); k = float(ku_(ret_test, fisher=False))
    srd = sr_test / sqrt(ANN)                                   # per-period SR for the formula
    sr0d = sr0 / sqrt(ANN)
    denom = sqrt(max(1 - s * srd + (k - 1) / 4.0 * srd ** 2, 1e-9))
    z = (srd - sr0d) * sqrt(max(n_obs - 1, 1)) / denom
    return float(norm.cdf(z)), sr0


def project(w, wmax):
    """Exact projection onto {w>=0, w<=wmax, sum=1}: normalize, then iteratively FREEZE any
    entry at the cap (frozen entries leave the pool permanently) and scale the rest to absorb
    the excess. Each pass freezes >=1 entry -> converges in <=n passes, cap exact."""
    w = np.clip(np.asarray(w, float), 0.0, None)
    s = w.sum()
    w = w / s if s > 0 else np.full_like(w, 1.0 / len(w))
    active = np.ones(len(w), bool)
    for _ in range(len(w)):
        if not (w > wmax + 1e-12).any():
            break
        over = active & (w >= wmax - 1e-12)
        excess = float((w[over] - wmax).sum())
        w[over] = wmax
        active &= ~over
        pool = float(w[active].sum())
        if pool <= 0:
            break                                       # cap infeasible (n*wmax < 1) — leave capped
        w[active] *= (pool + excess) / pool
    return w


def pso(train, wmax, particles, iters, lam_dd, seed):
    rng = np.random.default_rng(seed)
    n = train.shape[1]
    X = np.array([project(rng.random(n), wmax) for _ in range(particles)])
    V = rng.normal(0, 0.05, (particles, n))
    trial_srs = []

    def fit(w):
        r = train @ w
        s, d = stats(r)
        trial_srs.append(s)
        return s - lam_dd * d

    P = X.copy(); Pf = np.array([fit(x) for x in X])
    gi = int(Pf.argmax()); G, Gf = P[gi].copy(), Pf[gi]
    for it in range(iters):
        r1, r2 = rng.random((particles, n)), rng.random((particles, n))
        V = 0.72 * V + 1.49 * r1 * (P - X) + 1.49 * r2 * (G - X)
        X = np.array([project(x, wmax) for x in X + V])
        F = np.array([fit(x) for x in X])
        up = F > Pf
        P[up], Pf[up] = X[up], F[up]
        if Pf.max() > Gf:
            gi = int(Pf.argmax()); G, Gf = P[gi].copy(), Pf[gi]
    return G, trial_srs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--glob", default="*.csv"); ap.add_argument("--wide")
    ap.add_argument("--equity", type=float, default=100000.0)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--wmax", type=float, default=0.35)
    ap.add_argument("--particles", type=int, default=40); ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lambda-dd", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.wide:
        m = pd.read_csv(a.wide, index_col=0, parse_dates=True)
        rets = m.astype(float)
    else:
        if not a.dir:
            raise SystemExit("need --dir or --wide")
        pnl = load_dir(a.dir, a.glob)
        rets = pnl / a.equity
    names = list(rets.columns); n = len(names)
    split = int(len(rets) * (1 - a.test_frac))
    train, test = rets.iloc[:split].to_numpy(), rets.iloc[split:].to_numpy()
    d0, d1, d2 = rets.index[0].date(), rets.index[split].date(), rets.index[-1].date()

    w_pso, trial_srs = pso(train, a.wmax, a.particles, a.iters, a.lambda_dd, a.seed)
    w_ew = np.full(n, 1.0 / n)

    rows = {}
    for tag, w in (("equal_weight", w_ew), ("pso", w_pso)):
        s_tr, dd_tr = stats(pd.Series(train @ w))
        s_te, dd_te = stats(pd.Series(test @ w))
        rows[tag] = {"train_sharpe": round(s_tr, 2), "train_maxdd_pct": round(dd_tr, 2),
                     "test_sharpe": round(s_te, 2), "test_maxdd_pct": round(dd_te, 2)}
    p_dsr, sr0 = dsr(rows["pso"]["test_sharpe"], len(test), trial_srs, test @ w_pso)
    beats = (rows["pso"]["test_sharpe"] > rows["equal_weight"]["test_sharpe"]
             and rows["pso"]["test_maxdd_pct"] <= rows["equal_weight"]["test_maxdd_pct"] * 1.25)
    verdict = ("PSO weights VALIDATED out-of-sample (and DSR-credible)" if beats and p_dsr >= 0.95 else
               "PSO beats equal-weight OOS but DSR-inconclusive (could be trial luck) — prefer equal weight" if beats else
               "USE EQUAL WEIGHT — PSO does not beat it out of sample (honest null)")

    out = {"strategies": names, "n_days": int(len(rets)),
           "windows": {"train": f"{d0}..{d1}", "test": f"{d1}..{d2}"},
           "n_trials_evaluated": len(trial_srs), "expected_max_sharpe_of_trials": round(sr0, 2),
           "weights_pso": {nm: round(float(w), 3) for nm, w in zip(names, w_pso)},
           "results": rows, "dsr_prob": round(p_dsr, 3), "verdict": verdict,
           "note": "Optimizer saw ONLY the train window. Test window untouched. DSR deflates for every trial evaluated."}
    if a.json:
        print(json.dumps(out, indent=2)); return
    print("SWARM OPTIMIZER (PSO + anti-overfit gauntlet) | %d strategies | %d days | train %s | test %s"
          % (n, len(rets), out["windows"]["train"], out["windows"]["test"]))
    print("trials evaluated: %d  -> expected max trial Sharpe (luck bar): %.2f\n"
          % (out["n_trials_evaluated"], sr0))
    print("%-14s %13s %13s %13s %13s" % ("", "train Sharpe", "train MaxDD", "TEST Sharpe", "TEST MaxDD"))
    for tag in ("equal_weight", "pso"):
        r = rows[tag]
        print("%-14s %13.2f %12.2f%% %13.2f %12.2f%%"
              % (tag, r["train_sharpe"], r["train_maxdd_pct"], r["test_sharpe"], r["test_maxdd_pct"]))
    print("\nPSO weights:")
    for nm, w in sorted(out["weights_pso"].items(), key=lambda kv: -kv[1]):
        print("  %-52s %.3f" % (nm, w))
    print("\nDSR (prob. test Sharpe beats best-of-%d luck): %.3f" % (out["n_trials_evaluated"], p_dsr))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
