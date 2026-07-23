# AUDIT — accuracy, efficiency, appropriateness of the jar

**Last run:** 2026-07-23 · `uv run scripts/accuracy_audit.py` · **22/22 PASS**
Method: every skill is tested against SYNTHETIC data where the true answer is planted BY
CONSTRUCTION — a skill passes only if it recovers the planted truth AND stays quiet on planted
noise (false-positive control). Re-runnable by anyone, any time.

## Accuracy grid (planted-truth recovery)

| Skill | Test | Result |
|---|---|---|
| stationarity-tests | random walk → I(1) | ✅ |
| | AR(1) φ=0.5 → I(0) | ✅ |
| | integration order of RW = 1 | ✅ |
| | built cointegration (β=2) detected, hedge recovered 2.00 | ✅ |
| | independent random walks NOT cointegrated (p=0.99) | ✅ false-positive control |
| garch-volatility | simulated GARCH(1,1): forecast-vol vs TRUE-vol **corr 0.991** | ✅ |
| | IID returns → forecast vol flat (cv 0.035) | ✅ false-positive control |
| ts-decomposition | planted +25bp Monday found (p=.018) | ✅ |
| | clean Thursday stays quiet | ✅ false-positive control |
| | no false TOM on pure noise | ✅ false-positive control |
| arima-forecast | planted AR(1): order picks AR terms; beats naive by **+30.1%** | ✅ |
| regression-diagnostics | planted heteroskedasticity / AR errors / collinearity all caught | ✅ ×3 |
| | clean regression → zero false flags | ✅ false-positive control |
| randomness-tests | IID → "nothing"; AR(1) → momentum flavor; GARCH → vol-clustering only | ✅ ×4 |
| swarm-optimizer | weight concentrates on the one real-Sharpe asset; cap exact ≤0.35 | ✅ ×2 |

**Bugs the audit itself caught before release:** numpy input path in VIF (regression), cap-projection
inflation in the PSO (weights could exceed the cap by ~0.3% — replaced with exact freeze-and-
redistribute water-filling), calendar-gapped indexes breaking ARIMA's append (real NQ data caught it).
That is the audit doing precisely its job.

## Efficiency (measured, this machine)

| Tool | Workload | Time |
|---|---|---|
| stationarity / randomness / seasonality / regression | full battery on 1,600–150,000 obs | **< 2s each** |
| garch-volatility | walk-forward, 1,800 obs | ~1–2s |
| swarm-optimizer | 30 particles × 60 iters (4,840-trial run: ~10s) | ~0.3s |
| arima-forecast | 200–500-step walk-forward | **30–40s** (statsmodels refits — the jar's one slow tool; use `--refit 40`) |
| full self-audit | all 22 checks | ~40–60s |

Token efficiency as skills: each SKILL.md ≈ 300–500 tokens loaded on invoke; no always-on cost
(skills load on demand). All tools support `--json` for programmatic use without re-parsing prose.

## Appropriateness as Claude Code skills

- **Right shape:** each answers a question a trader actually asks in plain English (descriptions
  are written for auto-triggering), runs in one command, returns a verdict — not a lecture.
- **Right division of labor:** the seven compose (stationarity feeds ARIMA's `d`; randomness gates
  everything; GARCH sizes anything; regression referees claims; swarm allocates) without overlap.
- **Right honesty:** every skill has a null path it takes proudly (ARIMA's random-walk verdict,
  seasonality's TOM null, pairs' no-edge, swarm's "use equal weight") — on real CME data the jar
  produced BOTH genuine positives (Monday effect, ES→NQ cross-predictor, GARCH DD-cut) and
  genuine nulls, which is exactly the base-rate an honest toolset should show.
- **Known limits (stated, not hidden):** linear tests can't see nonlinear structure; single-window
  harnesses are leads, not validations; ARIMA is slow; calendar cells need multiple-testing care.
