# Quant Econometrics Skills

A growing **jar of econometrics & time-series skills** for quant trading research — each one a runnable Claude Code skill that turns a piece of the classic literature into a **walk-forward, zero-look-ahead tool**, validated on real market data.

Built the same way the [GARCH Method](https://github.com/milesdeutscher/garchmethod) skill was: take a Nobel-grade or textbook-standard econometric model, implement it cleanly on the reference library, wire it into Claude Code as a plain-English skill, and prove it on real prices. The theory comes from a graduate econometrics course ("Modelling Economic Processes") plus the seminal papers each method is named after — cited, never invented.

**The through-line:** direction is the hard, crowded question. This jar answers the *other* questions that decide whether a strategy survives — **is this even stationary? how volatile is tomorrow? is there any structure here at all, or am I fitting noise?** Get those right and a modest directional edge becomes tradeable; get them wrong and a beautiful backtest is a mirage.

---

## Install (two commands, in Claude Code)

```
/plugin marketplace add SpideyHp27/quant-econometrics-skills
/plugin install quant-econometrics@quant-econometrics-skills
```

Then ask in plain English — *"is USTEC stationary or a random walk?"*, *"what's the vol forecast on NQ?"*, *"is this return series white noise or is there an edge?"* — and Claude fires the right skill.

Scripts run with [`uv`](https://docs.astral.sh/uv/) (dependencies resolve on first run via PEP 723 inline metadata — nothing to pip-install). Plain-pip users: `pip install -r requirements.txt`.

## The jar

Each skill is self-describing (`skills/<name>/SKILL.md`), has a runnable tool (`scripts/`), and ships a validation report on real data (`skills/<name>/REPORT.md`).

| # | Skill | Answers | Core method (literature) | Status |
|---|---|---|---|---|
| 1 | **garch-volatility** | How violent is tomorrow, and how big should I size? | GARCH(1,1) — Engle 1982, Bollerslev 1986 (Nobel 2003) | ✅ built |
| 2 | **stationarity-tests** | Is this mean-reverting or a random walk? Are these two a tradeable pair? | ADF (Dickey-Fuller 1979), KPSS 1992, Phillips-Perron 1988, Engle-Granger 1987 | ✅ built |
| 3 | **arima-forecast** | What's the model-based forecast of the next moves? | Box-Jenkins ARIMA (Box & Jenkins 1970) | ⏳ planned |
| 4 | **ts-decomposition** | What's trend vs seasonal vs noise? Is there a calendar edge? | STL decomposition, deterministic-trend models | ⏳ planned |
| 5 | **regression-diagnostics** | Is this regression edge real, or does it break its own assumptions? | OLS + Gauss-Markov battery, HAC/Newey-West, Chow | ⏳ planned |
| 6 | **randomness-tests** | Is there ANY exploitable structure, or is this white noise? | Ljung-Box portmanteau, runs, turning-points | ⏳ planned |

See [`ROADMAP.md`](ROADMAP.md) for the build journal — what's done, what's next, and the exact validation numbers as each skill lands.

## How they compose

The jar is designed to stack, not compete:

- **stationarity-tests → arima-forecast** — the integration order `d` from the unit-root test is exactly the differencing ARIMA needs.
- **garch-volatility on top of anything** — it never touches your entry logic; it scales exposure by forecast vol. The archetype-correct risk control for no-stop mean-reversion.
- **randomness-tests as a gate** — before modelling anything, ask whether there's structure to model. If the series is white noise, stop.
- **regression-diagnostics as the referee** — any cross-asset predictor regression must clear the unit-root + HAC checks or its p-value is a lie (spurious regression).

## Honesty rules (non-negotiable, inherited from the GARCH method)

1. **Nothing here predicts direction.** These are risk, structure, and validation tools. On their own they make no money; they make an existing directional edge survivable and honest.
2. **Report the drawdown and the failing test.** A skill that says "no edge here" is doing its job.
3. **Zero look-ahead, always.** Every walk-forward forecast uses only data that existed before the forecast day. Same discipline as a real research desk.
4. **Cite the source.** Every method names its paper. The math is the authors'; the implementation and the trading framing are mine.

## Credit

- **Methods:** the econometricians the tests are named after — Engle, Bollerslev, Dickey & Fuller, Kwiatkowski/Phillips/Schmidt/Shin, Phillips & Perron, Box & Jenkins, Granger. Read the originals.
- **Structural template:** Miles Deutscher's [garchmethod](https://github.com/milesdeutscher/garchmethod) — the plugin-marketplace shape and the "skill from a model" idea.
- **Implementation, trading framing, validation:** Hoysala Prasad.

## License

MIT.
