# Roadmap & Build Journal

The living record of this repo — what it is, the plan, and every skill as it lands with its real-data validation numbers. Newest entries at the top of the log.

## Mission
Turn a graduate econometrics course + the seminal papers into a **jar of runnable Claude Code skills** for quant trading research — the same way the GARCH Method skill was built. Each skill: a real model, implemented cleanly on the reference library, walk-forward with zero look-ahead, wired into Claude Code as a plain-English skill, and **proven on real market data** (USTEC daily = the live MeanRev instrument; CME NQ/ES daily = Databento). This repo is also the portfolio track record — everything we build accumulates here, start to finish.

## Source of the theory
- Course: **"Modelling Economic Processes" (`D:\MEP`)**, extracted + atomized into the Obsidian **Econometrics & Time-Series KB** (107 atoms / 16 hubs) in the 2026-07-22 session.
- Hubs → skills mapping below. Each hub's atoms are the theory each skill implements; each skill cites the original paper (Prime Directive: cite, never invent).

## The plan (6 skills, dependency order)

| # | Skill | KB hubs folded in | Depends on | Status |
|---|---|---|---|---|
| 1 | garch-volatility | (volatility — built first, 2026-07-22) | — | ✅ built |
| 2 | stationarity-tests | stationarity, unit-root | — | ✅ built |
| 3 | arima-forecast | arma-arima, box-jenkins, acf-pacf, autoregressive, moving-average, forecasting | #2 (uses `d`) | ⏳ next |
| 4 | ts-decomposition | time-series-decomposition, deterministic/stochastic trend | — | ⏳ |
| 5 | regression-diagnostics | regression, ols-assumptions, multicollinearity, residual-diagnostics, hypothesis-testing | — | ⏳ |
| 6 | randomness-tests | forecast-evaluation, white-noise/randomness | — | ⏳ |

Versioning: `plugin.json` minor version = number of skills in the jar (0.2.0 = 2 built). Bumps by one per skill.

## Per-skill loop (repeated for every skill)
**The bar (non-negotiable): a skill is FUNCTIONAL like GARCH, not a doc wrapper around a test.** GARCH is functional because it closes the loop — forecast → sizing *decision* → a proof harness (`compare.py`) that shows real P&L impact (cut MaxDD 38.5%→24.2%). Every skill here must do the same: produce a **decision/signal** AND ship a **proof harness** that backtests it on real money — and reports the null honestly when there is no edge.

1. Read the KB hub + atoms; pull the seminal paper(s); research where it adds value.
2. Build the tool on the reference library (`statsmodels`/`arch`/`scipy`), walk-forward, `--json`, PEP 723 deps.
3. **Build the functional output + proof harness** (a backtest / decision, not just a verdict).
4. Validate on real data; capture the exact numbers — positive edge OR honest null.
5. Write `SKILL.md` (auto-trigger description + "how it plugs into the research") + `REPORT.md` (what it does, what it does for the research, validation numbers, caveat, references).
6. Commit + push. Journal it here.

---

## Visual layer (owner: "seeing something running is better than anything")
A **local validation dashboard** (`dashboard/`, served at `localhost:8000`) renders every skill's real output so caveats are eyeballable, not buried in text. `build_data.py` runs the skills → `data.json`; `index.html` (self-contained, dark/neon, vanilla SVG) draws skill cards, the GARCH equity chart (buy&hold vs vol-targeted), the pairs cointegration heatmap (green/amber/red like an assets heatmap), and a caveats panel. Re-run `build_data.py` to refresh. **This is v1** — the roadmap (inspired by owner's reference dashboards) is a richer always-on surface: a live "matrix" of configs being swept, per-arm trade tape, holdout/exam watch, and a discretion-trainer. Built incrementally alongside the skills.

## Build log

### 2026-07-23 — skill 4: ts-decomposition (v0.4.0)
- **ts-decomposition** — STL decomposition with a **shuffled-null calibration** (STL finds a 31.5%
  pseudo-seasonal share in a pure random walk — the null exposes it; NQ real excess = 0.3% = no cycle),
  all DOW/TOM calendar cells with HAC p, fixed-window TOM harness vs B&H with IS/OOS.
- Validated on back-adjusted CME ES+NQ (1,661 sessions each): **TOM anomaly = HONEST NULL** (TOM days
  earn less than non-TOM; harness OOS negative on both — the 1980s effect is arbed). **Monday = live
  finding**: ES +15.4bp p=.0096, NQ +22.1bp p=.0011, independently significant both instruments —
  flagged as weekend-hold-premium candidate, research lead NOT validated edge.


### 2026-07-22 (later) — skill 3: swarm-optimizer (v0.3.0)
- **swarm-optimizer** — PSO portfolio weights + anti-overfit gauntlet (train/test wall, DSR charged for
  all ~4,840 trials, equal-weight benchmark with an honest "use equal weight" verdict path, iterative
  simplex projection for the concentration cap). Origin: owner's swarm-intelligence article; assessment:
  PSO is real value here for exactly one job — bundle weights — and only caged.
- Validated on the REAL 7-strategy Pella bundle (2018→2026, 2,172 days): PSO test Sharpe **2.95 vs
  equal-weight 2.03**, DSR 1.00 vs luck-bar 0.54 → machinery validated at the weights level.
- **The tool demonstrated its own caveat on real data:** it max-capped TT_NDX — beautiful backtest,
  but live it produced a single −14.75% swing-tail day. Daily-P&L Sharpe can't see intraday tails →
  output is a proposal for the human + portfolio-MC, never a deploy order. Correctly zeroed the two
  weakest sleeves.

### 2026-07-22 — repo created + skills 1–2 + local dashboard
- **Repo scaffolded** exact-kind to garchmethod: `.claude-plugin/` marketplace + plugin manifests, MIT `LICENSE`, `README.md`, `requirements.txt`, this journal. Installable via `/plugin marketplace add SpideyHp27/quant-econometrics-skills`.
- **Skill 1 — garch-volatility** (migrated in; originally built 2026-07-22 in the main session). GARCH(1,1) walk-forward vol forecast → regime → vol-target size. Proven earlier on real NQ: vol-targeting cut MaxDD **38.5% → 24.2%** with higher Sharpe. Refs: Engle 1982, Bollerslev 1986.
- **Skill 2 — stationarity-tests** (new). ADF + KPSS + Phillips-Perron fused verdict, integration order `d`, Engle-Granger cointegration. Validated on **USTEC daily** (2150 bars, 2018→2026):
  - Price = **I(1), non-stationary** (ADF p=0.99, KPSS p=0.01, PP p=0.99 — all agree). **This is the MeanRev_NDX forensic made formal: the live strategy is not reverting to a level, it's leveraged bull-beta.**
  - Log-returns = **I(0), stationary** (all three agree). Integration order **d=1**. USTEC~NQ cointegrated (p=0.0000, hedge 0.99 — same-underlying sanity check).
- **Skill 2 FUNCTIONAL upgrade — `pairs_backtest.py`** (this is what makes it a working skill, not a diagnostic). Walk-forward market-neutral spread reversion, costs modeled, zero look-ahead. Screened + backtested 6 pairs:
  - **Cointegration screen is PREDICTIVE:** the only 2 pairs with PF>1 (YM~ES p=0.0065 → PF 1.16; GLD~GDX p=0.062 → PF 1.24) are the only 2 that (borderline-)cointegrated; every clearly-non-cointegrated pair (ES~NQ, GC~SI, EWA~EWC, KO~PEP) loses money. The filter works.
  - **Honest null:** no pair clears Sharpe≥1 — static-cointegration daily pairs on liquid instruments are marginal-to-dead. Textbook EWA/EWC has drifted out of cointegration since its original window (relationship non-stationarity). Not fabricating an edge that isn't robust — that's the overfitting this project rejects. The tool's worth = it proves this honestly + refuses to over-trade; hunt richer universes (sector baskets, crypto, intraday).
  - Refs: Dickey & Fuller 1979, KPSS 1992, Phillips & Perron 1988, Engle & Granger 1987, Granger & Newbold 1974.

---

## Next
Functional-first, honest about edge vs null. Two tracks:
- **Best shot at a demonstrable POSITIVE edge → `ts-decomposition` / seasonality** (skill 4, may pull forward). Turn-of-month + day-of-week effects in equity indices are among the most robust anomalies; a seasonality extractor + backtest is the likeliest GARCH-style "it makes money" win. Prioritize if we want a clear positive on the board.
- **`arima-forecast`** (skill 3, Box-Jenkins): auto (p,d,q) via AIC/BIC + ACF/PACF, walk-forward k-step forecast + intervals, Ljung-Box residual check. Functional harness = forecast-driven signal backtest vs random-walk benchmark + directional-accuracy. Expect a likely null (markets ≈ efficient) but proven honestly. Consumes `d` from skill 2.
- **`randomness-tests`** (skill 6): Ljung-Box + Lo-MacKinlay variance ratio as a **strategy selector** (momentum vs reversion vs noise) — a functional gate that kills white-noise strategies. Strong positive-utility candidate.
