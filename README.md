> **Proprietary - All Rights Reserved.** Licensed to Sandeep Grover. No reuse, copying, modification, or distribution without prior written permission. See LICENSE.

# NQ / ES Regime-Detection Research Demo

Author: Sandeep Grover, PhD

A small, walk-forward study of whether market regimes detected on daily E-mini S&P 500 (ES) and
E-mini Nasdaq-100 (NQ) futures can be used to switch between a trend-following sleeve and a
mean-reversion sleeve. It was built as a research demonstration on real data, not as a trading
recommendation. Every number is produced by `main.py` and written to `results/results.json`; the
tables under "Headline results" were printed from that file by `report.py`.

## Headline results (out-of-sample, net of costs)

**Short answer: neither regime-switching strategy beat buy-and-hold on Sharpe ratio, on either
instrument.** Both have a lower point Sharpe than buy-and-hold, and the 95% bootstrap intervals of the
difference span zero, so the gap is not statistically distinguishable either way. The HMM switch also
did not beat the simpler volatility x trend rules switch.

What the switching strategies did deliver is a smaller maximum drawdown than buy-and-hold over this
path (ES: rules switch -22.8% and HMM switch -32.5% vs buy-and-hold -54.2%; NQ: rules switch -23.7% and
HMM switch -40.8% vs -50.6%). Read that with care: the switches were in the market only 65% to 74% of
days, ran at lower volatility, and earned a lower CAGR than buy-and-hold; maximum drawdown is a single
statistic of one historical path and no significance test was run on it.

Source: results/results.json, generated_utc 2026-10-15T19:17:52+00:00

#### ES (ES=F), out-of-sample 2008-01-02 to 2026-10-14, 4727 trading days, 19 yearly refits

| Strategy | Sharpe (net) | CAGR (net) | Ann. vol | Max drawdown | Time in market | Sharpe gross | Sharpe at 2x costs |
|---|---|---|---|---|---|---|---|
| `buy_and_hold` | 0.54 | 9.3% | 20.0% | -54.2% | 100.0% | 0.54 | 0.54 |
| `trend` | 0.29 | 3.8% | 20.0% | -43.4% | 100.0% | 0.29 | 0.28 |
| `meanrev` | 0.18 | 1.4% | 12.0% | -22.5% | 31.4% | 0.25 | 0.10 |
| `hmm_switch` | 0.40 | 5.1% | 15.5% | -32.5% | 70.9% | 0.43 | 0.37 |
| `rules_switch` | 0.48 | 6.1% | 14.7% | -22.8% | 65.1% | 0.52 | 0.43 |
| `hmm_switch_incl_long` | 0.32 | 4.2% | 17.8% | -42.2% | 82.5% | 0.34 | 0.30 |

Sharpe differences, paired stationary block bootstrap (95% interval):

| Comparison | Point | 95% CI |
|---|---|---|
| `hmm_switch_minus_buy_and_hold` | -0.14 | [-0.64, 0.30] |
| `rules_switch_minus_buy_and_hold` | -0.07 | [-0.55, 0.39] |
| `hmm_switch_minus_trend` | 0.11 | [-0.19, 0.43] |
| `hmm_switch_minus_rules_switch` | -0.08 | [-0.38, 0.24] |

* Verdict (HMM switch): not statistically distinguishable from buy-and-hold on Sharpe (95% bootstrap CI spans 0).
* Verdict (rules switch): not statistically distinguishable from buy-and-hold on Sharpe (95% bootstrap CI spans 0).
* HMM switch Sharpe: mean training 0.53, mean single-year OOS 0.47, full-period OOS 0.40; sleeve choice changed in 1 of 54 regime slots across refits.
* OOS HMM regime episodes: 122 (low_vol 45, mid_vol 61, high_vol 16).
* BIC-selected K: [4] across all refits.
* Lookahead spot check passed: True.

#### NQ (NQ=F), out-of-sample 2008-01-02 to 2026-10-14, 4727 trading days, 19 yearly refits

| Strategy | Sharpe (net) | CAGR (net) | Ann. vol | Max drawdown | Time in market | Sharpe gross | Sharpe at 2x costs |
|---|---|---|---|---|---|---|---|
| `buy_and_hold` | 0.74 | 15.3% | 22.6% | -50.6% | 100.0% | 0.74 | 0.74 |
| `trend` | 0.46 | 8.1% | 22.6% | -39.9% | 100.0% | 0.46 | 0.46 |
| `meanrev` | 0.09 | 0.3% | 13.6% | -33.7% | 33.4% | 0.14 | 0.05 |
| `hmm_switch` | 0.34 | 4.7% | 18.5% | -40.8% | 73.7% | 0.35 | 0.33 |
| `rules_switch` | 0.58 | 9.0% | 17.5% | -23.7% | 70.8% | 0.60 | 0.56 |
| `hmm_switch_incl_long` | 0.47 | 8.0% | 21.1% | -48.9% | 90.7% | 0.47 | 0.47 |

Sharpe differences, paired stationary block bootstrap (95% interval):

| Comparison | Point | 95% CI |
|---|---|---|
| `hmm_switch_minus_buy_and_hold` | -0.40 | [-0.92, 0.07] |
| `rules_switch_minus_buy_and_hold` | -0.16 | [-0.61, 0.27] |
| `hmm_switch_minus_trend` | -0.12 | [-0.39, 0.16] |
| `hmm_switch_minus_rules_switch` | -0.24 | [-0.52, 0.02] |

* Verdict (HMM switch): not statistically distinguishable from buy-and-hold on Sharpe (95% bootstrap CI spans 0).
* Verdict (rules switch): not statistically distinguishable from buy-and-hold on Sharpe (95% bootstrap CI spans 0).
* HMM switch Sharpe: mean training 0.45, mean single-year OOS 0.49, full-period OOS 0.34; sleeve choice changed in 5 of 54 regime slots across refits.
* OOS HMM regime episodes: 133 (low_vol 48, mid_vol 67, high_vol 18).
* BIC-selected K: [4] across all refits.
* Lookahead spot check passed: True.

## How to run

Python 3.11. Versions are pinned in `requirements.txt` (the exact versions used for the committed
results; they are also recorded in `results/results.json` under `environment`).

```bash
pip install -r requirements.txt && python3 main.py   # full study: download, walk-forward, outputs
python3 report.py                                     # reprint the headline tables from results.json
python3 -m pytest -q                                  # offline unit checks (synthetic data, no network)
```

`main.py` downloads data (internet needed), fits 38 yearly models (19 per instrument), backtests,
runs the lookahead spot check and writes all outputs. Expect a few minutes on a laptop CPU.

Raw Yahoo Finance downloads are **not** committed (`data/` is git-ignored); `main.py` re-fetches them.
A re-run will therefore use data that ends later than 2026-10-14 and may include vendor revisions, so
its numbers can differ slightly from the committed `results/`.

## Data

* Source: Yahoo Finance via `yfinance`, tickers `ES=F` and `NQ=F` (continuous front-month), daily,
  maximum history. The committed run used 6,583 clean daily closes per instrument, 2000-09-18 to
  2026-10-14 (`instruments.<ROOT>.data` in `results.json`); the out-of-sample period is 2008-01-02 to
  2026-10-14. Raw downloads are saved unmodified to `data/*_raw.csv` (not committed).
* Cleaning: drop non-positive or missing closes and duplicate dates; drop the bar dated "today"
  (New York time), because during the session Yahoo returns a partial bar that is still moving.
  Daily moves above 15% are listed in `results.json` (`data.abs_daily_move_gt_15pct`) for inspection.
* If a futures download fails, the code falls back to `^GSPC` / `^NDX` and records `is_proxy: true`.
  The actual ticker, row counts and date ranges used are stored under `instruments.<ROOT>.data`.

## Method

### Features (trailing windows only)
`r1` daily log return, `rv20` 20-day realised volatility (annualised), `r5`, `r20`, `r60`
log returns, `slope60` 60-day OLS slope of log price, `ma50`, `ma200`, and `dd252` drawdown from the
252-day high. Each value at date t uses closes up to and including t.

### Regime model 1: Gaussian HMM
* Inputs: standardised `[r1, log(rv20), r20]`; the scaler is fitted on the training window only.
* Model: `hmmlearn.GaussianHMM`, full covariance, K in {2, 3, 4}, best of 5 random initialisations
  per K by log-likelihood, K chosen by BIC on the training window only.
* Label consistency: states are sorted by the variance of the daily-return dimension. The lowest is
  `low_vol`, the highest `high_vol`, anything between `mid_vol`. This keeps labels comparable across
  refits even when K changes.
* Inference: regime at date t = argmax of the **forward-filtered** probability P(state_t | data up to t),
  computed by an explicit forward recursion (`regimes.forward_filter`). Smoothed posteriors
  (`predict_proba`) and Viterbi paths (`predict`) are never used, because both condition on future data.

### Regime model 2: rules baseline
20-day realised volatility tercile (cut points from the training window) crossed with the sign of the
60-day return: six regimes such as `highvol_down`.

### Strategy sleeves
* Trend: +1 when MA50 > MA200, otherwise -1 (time-series trend, long/short).
* Mean reversion: z = r5 / (daily vol x sqrt(5)); long when z < -1, short when z > +1, else flat.

### Regime-switching allocation
For each regime, the sleeve with the best Sharpe **on the training window** is chosen (it must be
positive and have at least 63 training days in that regime; otherwise the regime is traded flat).
That mapping is then applied, unchanged, to the following out-of-sample year. A sensitivity variant
`hmm_switch_incl_long` also allows a plain long position as a candidate.

### Walk-forward protocol
Expanding window, one refit per calendar year. The model for year Y sees only data before 1 January
of Y. The first out-of-sample year is 2008, so the first training window covers late 2001 to 2007
(after the 252-day feature warm-up). Fitted parameters, BIC tables, scaler values, rules thresholds and
sleeve choices for every refit are saved to `results/params/<ROOT>_<YEAR>.json`.

### Execution and costs
* Timing: a signal computed at the close of day t is filled at the close of day t+1 (1-day execution
  lag), so the first return it earns is close(t+1) to close(t+2). The regime label used to evaluate
  a return carries the same lag.
* Costs per unit of turnover and per side: 1 tick slippage plus USD 2.50 commission and fees, divided
  by contract notional at the previous close (ES: USD 50 x index, tick USD 12.50; NQ: USD 20 x index,
  tick USD 5.00). Cost in basis points therefore falls as the index level rises. Sharpe ratios at
  zero cost and at twice this cost are reported as sensitivity checks.
* Sharpe uses rf = 0: a futures return is already approximately an excess-of-cash return.

### No-lookahead check (automated, limited scope)
`main.lookahead_check` truncates the price series at four randomly chosen out-of-sample dates,
recomputes features, filtered regime probabilities and sleeve signals on the truncated data, and
confirms they are identical to the full-sample run at those dates. The outcome is stored in
`results.json` (`lookahead_check`).

What this check does and does not cover:
* It reuses the stored model for year T (HMM parameters, scaler, thresholds); it does not refit that
  model on the truncated data. It therefore cannot detect a leak in the training window (for example
  a model for year Y that was accidentally fitted on data from Y or later).
* It compares only features, filtered probabilities and sleeve signals, at four dates. It does not
  check regime labels, positions, the execution lag or returns, and four dates are a spot check, not
  a proof. Protection against training-window leakage rests on the walk-forward code itself (the
  training slice ends before 1 January of the test year), which this check does not exercise.

### Statistics reported (out-of-sample, net of costs)
CAGR, annualised volatility, Sharpe, maximum drawdown, % time in market (and long/short split),
annual turnover; per-regime annualised return, volatility, Sharpe and hit rate for each sleeve;
regime frequencies, episode counts, mean and median durations; empirical daily transition matrix;
paired stationary block bootstrap (mean block 20 days, 2000 resamples) confidence intervals for
Sharpe differences (each comparison also reports `share_boot_draws_le_0`, the share of bootstrap draws
of the difference that are <= 0; the draws are centred on the point estimate, not on zero, so this is
a descriptive share and NOT a p-value; significance is judged from the 95% interval only); training-versus-out-of-sample Sharpe for each refit; stability of the
regime-to-sleeve mapping across refits.

## Outputs

| File | Contents |
|---|---|
| `results/results.json` | `summary` (headline numbers, verdicts, flags), full per-instrument metrics, design notes, environment versions |
| `results/<ROOT>_per_regime_hmm.csv`, `..._rules.csv` | per-regime OOS performance of each sleeve and the switch strategy |
| `results/<ROOT>_daily_oos.csv` | daily close, regime label, filtered probabilities, positions and net returns for every strategy |
| `results/<ROOT>_refit_train_vs_oos.csv` | training Sharpe vs next-year OOS Sharpe per refit |
| `results/params/<ROOT>_<YEAR>.json` | fitted HMM parameters and all choices made at each refit |
| `report.py` | prints the headline tables in this README from `results/results.json` |
| `tests/` | offline pytest checks: 1-day execution lag, costs reduce returns, no future data in features / signals / filtered probabilities, drawdown definition |

## Limitations (read before drawing conclusions)

* **Small regime sample.** Over roughly 19 out-of-sample years there are only on the order of a hundred
  regime episodes per instrument, and far fewer high-volatility episodes. Per-regime Sharpe ratios
  have wide confidence intervals.
* **Daily bars only.** No intraday information, no overnight/session split, close-to-close fills.
* **Continuous-contract artefacts.** Yahoo's `ES=F`/`NQ=F` series splice front-month contracts without
  back-adjustment, so roll days contain the calendar spread as a spurious return; roll costs are not
  charged. A production study should use back-adjusted contracts with explicit roll dates.
* **ES=F expiry-day quirk.** Comparing `ES=F` with `^GSPC`, the gap between the two concentrates on
  the quarterly third-Friday expiries and the next trading day, and reverses on the following day.
  These spurious moves feed the `r1`, `rv20` and `r5` features. The 2-day lag between signal and the
  first return earned mostly neutralises the P&L effect. It is not lookahead (it uses only past data),
  but it is noise in the inputs. Fix in production: back-adjusted contracts, or mask the expiry bars.
* **BIC tends to the largest K on long daily samples**, so K is often boundary-limited rather than a
  sharp optimum (flagged automatically in `summary.<ROOT>.flags`). Part of the reason: the HMM inputs
  `rv20` and `r20` are built from overlapping 20-day windows, so consecutive days are far from
  independent, yet BIC uses n = number of days as if they were. That overstates the effective sample
  size, shrinks the relative complexity penalty and over-rewards extra states. K = 4, the top of the
  searched range, is therefore best read as a boundary artefact, not evidence of four real regimes.
* **ES and NQ are highly correlated**, so the two instruments are not independent confirmations.
* **Design choices were made by hand** (feature set, sleeve rules, thresholds, 2008 start). Nothing was
  tuned on out-of-sample data, but the set of variants reported still carries selection optimism.
* **No claim of tradability.** This is a research demonstration of a sound protocol, not a strategy
  recommendation. In this run the regime switches did not beat buy-and-hold, and the results file says so.
* **No deflated Sharpe ratio is computed** in this demo, although six variants per instrument were
  tried (`summary.multiple_testing_note` in `results.json`); Bailey and Lopez de Prado (2014) is cited
  as the appropriate next step.

## References

* Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the
  business cycle. *Econometrica*, 57(2), 357-384.
* Ang, A., and Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial
  Economics*, 4, 313-337.
* Moskowitz, T. J., Ooi, Y. H., and Pedersen, L. H. (2012). Time series momentum. *Journal of Financial
  Economics*, 104(2), 228-250.
* Politis, D. N., and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American
  Statistical Association*, 89(428), 1303-1313.
* Bailey, D. H., and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias,
  backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94-107.

## License

Proprietary - All Rights Reserved. Licensed to Sandeep Grover. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
