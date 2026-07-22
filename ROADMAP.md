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
1. Read the KB hub + atoms; pull the seminal paper(s); research where it adds value.
2. Build the tool on the reference library (`statsmodels`/`arch`/`scipy`), walk-forward, `--json`, PEP 723 deps.
3. Validate on real data; capture the exact numbers.
4. Write `SKILL.md` (auto-trigger description + "how it plugs into the research") + `REPORT.md` (what it does, what it does for the research, validation numbers, caveat, references).
5. Commit + push. Journal it here.

---

## Build log

### 2026-07-22 — repo created + skills 1–2 landed
- **Repo scaffolded** exact-kind to garchmethod: `.claude-plugin/` marketplace + plugin manifests, MIT `LICENSE`, `README.md`, `requirements.txt`, this journal. Installable via `/plugin marketplace add SpideyHp27/quant-econometrics-skills`.
- **Skill 1 — garch-volatility** (migrated in; originally built 2026-07-22 in the main session). GARCH(1,1) walk-forward vol forecast → regime → vol-target size. Proven earlier on real NQ: vol-targeting cut MaxDD **38.5% → 24.2%** with higher Sharpe. Refs: Engle 1982, Bollerslev 1986.
- **Skill 2 — stationarity-tests** (new). ADF + KPSS + Phillips-Perron fused into one verdict, integration order `d`, Engle-Granger cointegration. Validated on **USTEC daily** (2150 bars, 2018→2026):
  - Price = **I(1), non-stationary** (ADF p=0.99, KPSS p=0.01, PP p=0.99 — all agree). **This is the MeanRev_NDX forensic made formal: the live strategy is not reverting to a level, it's leveraged bull-beta.**
  - Log-returns = **I(0), stationary** (all three agree). Integration order **d=1**.
  - **USTEC ~ NQ cointegrated** (p=0.0000, hedge ratio 0.99) — same underlying, spread tradeable (sanity check).
  - Refs: Dickey & Fuller 1979, KPSS 1992, Phillips & Perron 1988, Engle & Granger 1987, Granger & Newbold 1974 (spurious regression).

---

## Next
- **Skill 3 — arima-forecast** (Box-Jenkins): auto (p,d,q) via AIC/BIC + ACF/PACF, walk-forward k-step forecast + intervals, Ljung-Box residual white-noise check. Consumes `d` from skill 2. Validate on NQ/USTEC daily returns; compare naive vs ARIMA forecast error (honest, out-of-sample).
