"""End-to-end regime-detection research demo on ES and NQ daily futures.

Run:  python3 main.py
Outputs: data/*.csv (raw downloads), results/results.json, results/*.csv, results/params/*.json

Design (no lookahead):
  * expanding-window walk-forward, one refit per calendar year (first out-of-sample year
    FIRST_OOS_YEAR); each refit sees data strictly before 1 January of the test year;
  * feature scaling, HMM fit, K selection (BIC), rules thresholds and sleeve choice per
    regime are all estimated on the training window only;
  * regimes in the test year come from forward-filtered probabilities only;
  * signals at close t are filled at close t+1 (1-day execution lag), with costs.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

# Small matrices: BLAS thread oversubscription only slows the many HMM fits down.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import backtest as bt
import regimes as rg
import strategies as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RES = ROOT / "results"
PARAMS = RES / "params"

INSTRUMENTS = {"ES": "ES=F", "NQ": "NQ=F"}
PROXIES = {"ES": "^GSPC", "NQ": "^NDX"}
FIRST_OOS_YEAR = 2008
MIN_TRAIN_OBS_PER_REGIME = 63  # a bucket needs >= one quarter of training days to get a sleeve
LOOKAHEAD_CHECK_DATES = 4


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def download(root: str) -> tuple[pd.Series, dict]:
    import yfinance as yf

    info = {"root": root, "yfinance_version": yf.__version__}
    for ticker, is_proxy in ((INSTRUMENTS[root], False), (PROXIES[root], True)):
        try:
            df = yf.download(ticker, period="max", interval="1d", auto_adjust=False,
                             progress=False, multi_level_index=False)
        except Exception as e:  # noqa: BLE001
            info.setdefault("errors", []).append(f"{ticker}: {e!r}")
            continue
        if df is None or len(df) < 2000:
            info.setdefault("errors", []).append(f"{ticker}: only {0 if df is None else len(df)} rows")
            continue
        raw_path = DATA / f"{ticker.replace('=', '_').replace('^', '')}_raw.csv"
        df.to_csv(raw_path)
        info.update({"ticker": ticker, "is_proxy": is_proxy, "raw_csv": str(raw_path.relative_to(ROOT)),
                     "raw_rows": int(len(df)), "raw_start": str(df.index.min().date()),
                     "raw_end": str(df.index.max().date())})
        close = df["Close"].astype(float)
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        close = close[~close.index.duplicated(keep="last")].sort_index()
        n0 = len(close)
        close = close[(close > 0) & close.notna()]
        info["dropped_nan_or_nonpositive"] = int(n0 - len(close))
        # Drop today's bar: during the session yfinance returns a partial, still-moving bar.
        today_ny = dt.datetime.now(ZoneInfo("America/New_York")).date()
        n1 = len(close)
        close = close[close.index.date < today_ny]
        info["dropped_partial_today_bar"] = int(n1 - len(close))
        r = close.pct_change()
        big = r[r.abs() > 0.15]
        info["abs_daily_move_gt_15pct"] = {str(k.date()): float(v) for k, v in big.items()}
        info.update({"clean_rows": int(len(close)), "clean_start": str(close.index.min().date()),
                     "clean_end": str(close.index.max().date())})
        return close, info
    raise RuntimeError(f"No data for {root}: {info.get('errors')}")


# ---------------------------------------------------------------------------
# Sleeve selection on training data
# ---------------------------------------------------------------------------
def choose_sleeves(labels: pd.Series, train_idx: pd.Index, net: dict, buckets: list[str],
                   candidates: list[str]) -> tuple[dict, list]:
    """For each regime bucket pick the candidate sleeve with the best TRAINING Sharpe.

    labels: decision-time regime label (known at close t). The label attached to the return
    of day t is labels[t - (1 + EXEC_LAG)], the same lag the positions use.
    """
    lagged = labels.reindex(net["flat"].index).shift(1 + bt.EXEC_LAG)
    rows, choice = [], {}
    tr = lagged.reindex(train_idx)
    for b in buckets:
        days = tr.index[tr == b]
        best, best_sh = "flat", 0.0
        for c in candidates:
            r = net[c].reindex(days)
            sh = bt.sharpe(r.to_numpy()) if len(r) > 2 and r.std() > 0 else 0.0
            rows.append({"bucket": b, "sleeve": c, "train_days": int(len(days)), "train_sharpe": sh})
            if len(days) >= MIN_TRAIN_OBS_PER_REGIME and np.isfinite(sh) and sh > best_sh:
                best, best_sh = c, sh
        choice[b] = best
    return choice, rows


def composite_signal(labels: pd.Series, choice: dict, sig: dict) -> pd.Series:
    out = pd.Series(0.0, index=labels.index)
    for b, c in choice.items():
        m = labels == b
        out[m] = sig[c].reindex(labels.index)[m]
    return out


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def walk_forward(root: str, close: pd.Series) -> dict:
    feat = rg.compute_features(close)
    ok = rg.valid_mask(feat)
    ret = close.pct_change().fillna(0.0)
    cost = bt.cost_rate_per_side(close, root)

    sig = {"trend": st.trend_signal(feat), "meanrev": st.meanrev_signal(feat),
           "long": st.long_signal(feat), "flat": st.flat_signal(feat)}
    pos = {k: bt.lag_positions(v) for k, v in sig.items()}
    net = {k: bt.net_returns(pos[k], ret, cost) for k in sig}

    idx = feat.index
    hmm_label = pd.Series(np.nan, index=idx, dtype=object)
    hmm_state = pd.Series(np.nan, index=idx)
    hmm_k = pd.Series(np.nan, index=idx)
    probs = pd.DataFrame(np.nan, index=idx, columns=[f"p_state{j}" for j in range(4)])
    rules_label = pd.Series(np.nan, index=idx, dtype=object)
    comp = {"hmm_switch": pd.Series(np.nan, index=idx), "rules_switch": pd.Series(np.nan, index=idx),
            "hmm_switch_incl_long": pd.Series(np.nan, index=idx)}
    refits, model_store = [], {}

    years = sorted({d.year for d in idx if d.year >= FIRST_OOS_YEAR})
    for Y in years:
        train_idx = idx[(idx.year < Y) & ok]
        test_idx = idx[(idx.year == Y) & ok]
        if len(test_idx) == 0:
            continue
        upto = idx[(idx.year <= Y) & ok]
        mu = feat.loc[train_idx, rg.HMM_FEATURES].mean()
        sd = feat.loc[train_idx, rg.HMM_FEATURES].std()
        Xtr = ((feat.loc[train_idx, rg.HMM_FEATURES] - mu) / sd).to_numpy()
        params, bic_table = rg.fit_hmm_bic(Xtr)
        k = params["k"]
        X = ((feat.loc[upto, rg.HMM_FEATURES] - mu) / sd).to_numpy()
        P = rg.forward_filter(params, X)  # row t uses rows <= t only
        state = P.argmax(axis=1)
        lab = pd.Series(rg.state_to_bucket(state, k), index=upto)

        q1, q2 = rg.rules_thresholds(feat.loc[train_idx])
        rlab = rg.rules_regime(feat.loc[upto], q1, q2)

        hmm_choice, hmm_tab = choose_sleeves(lab, train_idx, net, rg.HMM_BUCKETS, ["trend", "meanrev"])
        hmml_choice, _ = choose_sleeves(lab, train_idx, net, rg.HMM_BUCKETS, ["trend", "meanrev", "long"])
        rules_choice, rules_tab = choose_sleeves(rlab, train_idx, net, rg.RULE_BUCKETS, ["trend", "meanrev"])

        # In-sample (training) Sharpe of the allocation, for the overfitting diagnostic.
        is_sig = composite_signal(lab, hmm_choice, sig).reindex(idx).fillna(0.0)
        is_net = bt.net_returns(bt.lag_positions(is_sig), ret, cost).reindex(train_idx)
        is_sh = bt.sharpe(is_net.to_numpy())

        # Record decision-time labels and composite signals for the test year only.
        hmm_label[test_idx] = lab[test_idx]
        hmm_state[test_idx] = pd.Series(state, index=upto)[test_idx]
        hmm_k[test_idx] = k
        pdf = pd.DataFrame(P, index=upto, columns=[f"p_state{j}" for j in range(k)])
        probs.loc[test_idx, pdf.columns] = pdf.loc[test_idx].to_numpy()
        rules_label[test_idx] = rlab[test_idx]
        comp["hmm_switch"][test_idx] = composite_signal(lab[test_idx], hmm_choice, sig)
        comp["hmm_switch_incl_long"][test_idx] = composite_signal(lab[test_idx], hmml_choice, sig)
        comp["rules_switch"][test_idx] = composite_signal(rlab[test_idx], rules_choice, sig)

        rec = {
            "test_year": Y, "train_start": str(train_idx[0].date()), "train_end": str(train_idx[-1].date()),
            "train_days": int(len(train_idx)), "test_days": int(len(test_idx)), "k_selected": k,
            "converged": params["converged"], "bic_table": bic_table,
            "scaler_mean": mu.to_dict(), "scaler_std": sd.to_dict(),
            "startprob": params["startprob"].tolist(), "transmat": params["transmat"].tolist(),
            "means_scaled": params["means"].tolist(), "covars_scaled": params["covars"].tolist(),
            "state_daily_vol_ann": [float(np.sqrt(params["covars"][j][0, 0]) * sd["r1"] * np.sqrt(252))
                                    for j in range(k)],
            "rules_vol_terciles": [q1, q2],
            "hmm_sleeve_choice": hmm_choice, "hmm_incl_long_sleeve_choice": hmml_choice,
            "rules_sleeve_choice": rules_choice,
            "hmm_train_sleeve_table": hmm_tab, "rules_train_sleeve_table": rules_tab,
            "hmm_switch_train_sharpe": is_sh,
        }
        (PARAMS / f"{root}_{Y}.json").write_text(json.dumps(clean_json(rec), indent=1))
        refits.append(rec)
        model_store[Y] = {"params": params, "mu": mu, "sd": sd}

    return {"feat": feat, "ok": ok, "ret": ret, "cost": cost, "sig": sig, "pos": pos, "net": net,
            "hmm_label": hmm_label, "hmm_state": hmm_state, "hmm_k": hmm_k, "probs": probs,
            "rules_label": rules_label, "comp": comp, "refits": refits, "models": model_store}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(root: str, close: pd.Series, wf: dict) -> dict:
    ret, cost = wf["ret"], wf["cost"]
    oos_idx = wf["hmm_label"].dropna().index
    oos_start = oos_idx[0]
    oos = ret.index[ret.index >= oos_start]

    pos, net = {}, {}
    bh = pd.Series(0.0, index=ret.index)
    bh[oos] = 1.0  # long one unit for every OOS return day; entry cost charged on day one
    pos["buy_and_hold"] = bh
    for s in ("trend", "meanrev"):
        pos[s] = wf["pos"][s]
    for name, c in wf["comp"].items():
        pos[name] = bt.lag_positions(c.fillna(0.0))
    for name, p in pos.items():
        net[name] = bt.net_returns(p, ret, cost)

    strat_metrics = {n: bt.metrics(net[n][oos], pos[n][oos]) for n in net}
    cost_2x = {n: bt.metrics(bt.net_returns(pos[n], ret, 2 * cost)[oos], pos[n][oos])["sharpe"] for n in net}
    gross = {n: bt.metrics((pos[n] * ret)[oos])["sharpe"] for n in net}

    # Per-regime OOS performance, regime label lagged exactly like the positions.
    tables = {}
    for kind, labels, order in (("hmm", wf["hmm_label"], rg.HMM_BUCKETS),
                                ("rules", wf["rules_label"], rg.RULE_BUCKETS)):
        lag_lab = labels.shift(1 + bt.EXEC_LAG).reindex(oos)
        rows = []
        switch_name = "hmm_switch" if kind == "hmm" else "rules_switch"
        for b in order:
            days = lag_lab.index[lag_lab == b]
            for s in ("buy_and_hold", "trend", "meanrev", switch_name):
                r = net[s].reindex(days)
                sd = r.std()
                rows.append({"instrument": root, "regime_model": kind, "regime": b, "strategy": s,
                             "oos_days": int(len(days)), "share_of_oos": len(days) / len(oos),
                             "ann_return_arith": float(r.mean() * 252) if len(r) else np.nan,
                             "ann_vol": float(sd * np.sqrt(252)) if len(r) > 1 else np.nan,
                             "sharpe": float(r.mean() / sd * np.sqrt(252)) if len(r) > 2 and sd > 0 else np.nan,
                             "hit_rate": float((r > 0).mean()) if len(r) else np.nan})
        tab = pd.DataFrame(rows)
        tab.to_csv(RES / f"{root}_per_regime_{kind}.csv", index=False)
        tables[kind] = tab

    def regime_stats(labels: pd.Series, order: list[str]) -> dict:
        lab = labels.reindex(oos).dropna()
        ep = bt.run_lengths(lab)
        out = {}
        for b in order:
            e = ep[ep["label"] == b]
            out[b] = {"share_of_days": float((lab == b).mean()), "n_episodes": int(len(e)),
                      "mean_duration_days": float(e["n_days"].mean()) if len(e) else None,
                      "median_duration_days": float(e["n_days"].median()) if len(e) else None}
        tm = bt.transition_matrix(lab, order)
        return {"per_regime": out, "total_episodes": int(len(ep)),
                "empirical_daily_transition_matrix": {"order": order, "matrix": tm.round(4).values.tolist()}}

    hmm_stats = regime_stats(wf["hmm_label"], rg.HMM_BUCKETS)
    rules_stats = regime_stats(wf["rules_label"], rg.RULE_BUCKETS)

    boot = {
        "hmm_switch_minus_buy_and_hold": bt.bootstrap_sharpe_diff(net["hmm_switch"][oos], net["buy_and_hold"][oos]),
        "rules_switch_minus_buy_and_hold": bt.bootstrap_sharpe_diff(net["rules_switch"][oos], net["buy_and_hold"][oos]),
        "hmm_switch_minus_trend": bt.bootstrap_sharpe_diff(net["hmm_switch"][oos], net["trend"][oos]),
        "hmm_switch_minus_rules_switch": bt.bootstrap_sharpe_diff(net["hmm_switch"][oos], net["rules_switch"][oos]),
    }

    # Overfitting diagnostic: training (in-sample) Sharpe of the chosen allocation vs the
    # Sharpe realised in the following out-of-sample year.
    is_oos = []
    for rec in wf["refits"]:
        Y = rec["test_year"]
        yr = net["hmm_switch"][net["hmm_switch"].index.year == Y]
        is_oos.append({"year": Y, "k": rec["k_selected"], "train_sharpe": rec["hmm_switch_train_sharpe"],
                       "oos_year_sharpe": bt.sharpe(yr.to_numpy()) if yr.std() > 0 else None,
                       "choice": rec["hmm_sleeve_choice"]})
    is_oos_df = pd.DataFrame(is_oos)
    is_oos_df.to_csv(RES / f"{root}_refit_train_vs_oos.csv", index=False)
    changes = 0
    for a, b in zip(is_oos[:-1], is_oos[1:]):
        changes += sum(a["choice"].get(k) != b["choice"].get(k) for k in rg.HMM_BUCKETS)

    # Daily output
    daily = pd.DataFrame({"close": close, "ret": ret, "cost_rate_per_side": cost,
                          "hmm_regime": wf["hmm_label"], "hmm_state": wf["hmm_state"], "hmm_k": wf["hmm_k"],
                          "rules_regime": wf["rules_label"]}).join(wf["probs"])
    for n in pos:
        daily[f"pos_{n}"] = pos[n]
        daily[f"net_{n}"] = net[n]
    daily.loc[oos].to_csv(RES / f"{root}_daily_oos.csv", index_label="date")

    return {
        "oos_start": str(oos[0].date()), "oos_end": str(oos[-1].date()), "oos_days": int(len(oos)),
        "n_refits": len(wf["refits"]),
        "k_selected_by_year": {r["test_year"]: r["k_selected"] for r in wf["refits"]},
        "strategy_metrics_net": strat_metrics,
        "sharpe_gross_of_costs": gross,
        "sharpe_at_2x_costs": cost_2x,
        "bootstrap_sharpe_differences": boot,
        "hmm_regime_stats_oos": hmm_stats,
        "rules_regime_stats_oos": rules_stats,
        "hmm_sleeve_choice_by_year": {r["test_year"]: r["hmm_sleeve_choice"] for r in wf["refits"]},
        "rules_sleeve_choice_by_year": {r["test_year"]: r["rules_sleeve_choice"] for r in wf["refits"]},
        "overfit_diagnostic": {
            "mean_train_sharpe_hmm_switch": float(is_oos_df["train_sharpe"].mean()),
            "mean_oos_year_sharpe_hmm_switch": float(is_oos_df["oos_year_sharpe"].astype(float).mean()),
            "full_oos_sharpe_hmm_switch": strat_metrics["hmm_switch"]["sharpe"],
            "sleeve_choice_changes_across_refits": int(changes),
            "possible_choice_slots": int((len(is_oos) - 1) * len(rg.HMM_BUCKETS)),
        },
        "per_regime_tables": {k: f"results/{root}_per_regime_{k}.csv" for k in tables},
        "_tables": tables, "_net": net, "_pos": pos,
    }


# ---------------------------------------------------------------------------
# No-lookahead check: truncate the data at date T and recompute what was known at T.
# ---------------------------------------------------------------------------
def lookahead_check(root: str, close: pd.Series, wf: dict, n_dates: int = LOOKAHEAD_CHECK_DATES) -> dict:
    rng = np.random.default_rng(11)
    oos_dates = wf["hmm_label"].dropna().index
    picks = sorted(rng.choice(oos_dates[5:-5], size=n_dates, replace=False))
    res = []
    for T in picks:
        T = pd.Timestamp(T)
        c_tr = close[close.index <= T]
        f_tr = rg.compute_features(c_tr)
        ok_tr = rg.valid_mask(f_tr)
        feat_same = np.allclose(f_tr.loc[T].to_numpy(dtype=float), wf["feat"].loc[T].to_numpy(dtype=float),
                                equal_nan=True)
        m = wf["models"][T.year]
        upto = f_tr.index[ok_tr]
        X = ((f_tr.loc[upto, rg.HMM_FEATURES] - m["mu"]) / m["sd"]).to_numpy()
        P = rg.forward_filter(m["params"], X)
        k = m["params"]["k"]
        p_full = wf["probs"].loc[T, [f"p_state{j}" for j in range(k)]].to_numpy(dtype=float)
        prob_same = np.allclose(P[-1], p_full, atol=1e-10)
        sig_same = all(float(st.SLEEVES[s](f_tr).loc[T]) == float(wf["sig"][s].loc[T]) for s in st.SLEEVES)
        res.append({"date": str(T.date()), "features_identical": bool(feat_same),
                    "filtered_probs_identical": bool(prob_same), "signals_identical": bool(sig_same)})
    return {"dates_checked": res, "all_pass": all(r["features_identical"] and r["filtered_probs_identical"]
                                                  and r["signals_identical"] for r in res)}


# ---------------------------------------------------------------------------
def clean_json(o):
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def verdict(b: dict) -> str:
    if b["ci95_low"] > 0:
        return "better than buy-and-hold on Sharpe (95% bootstrap CI excludes 0)"
    if b["ci95_high"] < 0:
        return "worse than buy-and-hold on Sharpe (95% bootstrap CI excludes 0)"
    return "not statistically distinguishable from buy-and-hold on Sharpe (95% bootstrap CI spans 0)"


def main() -> int:
    for d in (DATA, RES, PARAMS):
        d.mkdir(parents=True, exist_ok=True)
    run_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out = {"generated_utc": run_at, "instruments": {}, "summary": {}}
    import hmmlearn, sklearn, scipy  # noqa: E401
    out["environment"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                          "hmmlearn": hmmlearn.__version__, "scipy": scipy.__version__,
                          "scikit_learn": sklearn.__version__}
    out["design"] = {
        "walk_forward": f"expanding window, yearly refit, first OOS year {FIRST_OOS_YEAR}",
        "hmm": "GaussianHMM full covariance on standardized [r1, log_rv20, r20]; K in {2,3,4} by BIC on training "
               "window; best of 5 seeds per K; states sorted by return variance; regimes = argmax of forward-filtered "
               "probabilities; buckets low_vol / mid_vol / high_vol",
        "rules_baseline": "20d realized vol tercile (training thresholds) x sign of 60d return, 6 regimes",
        "sleeves": "trend: sign(MA50-MA200) long/short; meanrev: vol-scaled 5d return z-score, long z<-1, short z>1",
        "allocation": "per regime, sleeve with best TRAINING Sharpe (>0, >= 63 training days) else flat",
        "execution": f"signal at close t, filled at close t+1 (lag {bt.EXEC_LAG} day), first return t+1->t+2",
        "costs": {"slippage_ticks_per_side": bt.SLIPPAGE_TICKS_PER_SIDE,
                  "commission_usd_per_side": bt.COMMISSION_PER_SIDE_USD, "contracts": bt.CONTRACTS,
                  "conversion": "(ticks x tick_value + commission) / (previous close x multiplier), per unit turnover"},
        "sharpe_definition": "mean/std x sqrt(252) of daily net returns, rf = 0 (futures returns are already "
                             "approximately excess of cash)",
        "bootstrap": "paired stationary block bootstrap (Politis and Romano 1994), mean block 20 days, 2000 reps",
    }
    for root in INSTRUMENTS:
        print(f"[{root}] downloading ...", flush=True)
        close, info = download(root)
        print(f"[{root}] {info['ticker']} rows={info['clean_rows']} {info['clean_start']} -> {info['clean_end']}"
              f"{' (PROXY)' if info['is_proxy'] else ''}", flush=True)
        wf = walk_forward(root, close)
        print(f"[{root}] walk-forward done: {len(wf['refits'])} refits, K by year "
              f"{[r['k_selected'] for r in wf['refits']]}", flush=True)
        ev = evaluate(root, close, wf)
        la = lookahead_check(root, close, wf)
        print(f"[{root}] lookahead check all_pass={la['all_pass']}", flush=True)
        ev_json = clean_json(ev)
        ev_json["data"] = info
        ev_json["lookahead_check"] = la
        out["instruments"][root] = ev_json

        sm = ev["strategy_metrics_net"]
        b = ev["bootstrap_sharpe_differences"]["hmm_switch_minus_buy_and_hold"]
        flags = []
        hs = ev["hmm_regime_stats_oos"]["per_regime"]
        thin = [k for k, v in hs.items() if v["n_episodes"] and v["n_episodes"] < 30]
        if thin:
            flags.append(f"HMM regimes with fewer than 30 OOS episodes: {thin}; per-regime Sharpe is noisy")
        od = ev["overfit_diagnostic"]
        if od["mean_train_sharpe_hmm_switch"] - od["full_oos_sharpe_hmm_switch"] > 0.3:
            flags.append("HMM switch training Sharpe exceeds OOS Sharpe by > 0.3: in-sample selection is optimistic")
        if od["sleeve_choice_changes_across_refits"] > 0.25 * od["possible_choice_slots"]:
            flags.append("Sleeve choice per regime changes often across refits: unstable mapping")
        ks = set(ev["k_selected_by_year"].values())
        if ks == {4}:
            flags.append("BIC picked K=4 (the upper bound of the search) at every refit: with ~1.6k-4.7k daily "
                         "observations BIC keeps rewarding extra states, so K is boundary-limited, not a clear optimum")
        if b["ci95_low"] <= 0 <= b["ci95_high"]:
            flags.append("Sharpe difference vs buy-and-hold not significant at 95%")
        out["summary"][root] = clean_json({
            "ticker": info["ticker"], "is_proxy": info["is_proxy"],
            "data_range": f"{info['clean_start']} to {info['clean_end']}", "data_rows": info["clean_rows"],
            "oos_range": f"{ev['oos_start']} to {ev['oos_end']}", "oos_days": ev["oos_days"],
            "sharpe_net": {k: v["sharpe"] for k, v in sm.items()},
            "cagr_net": {k: v["cagr"] for k, v in sm.items()},
            "max_drawdown": {k: v["max_drawdown"] for k, v in sm.items()},
            "ann_vol": {k: v["ann_vol"] for k, v in sm.items()},
            "pct_time_in_market": {k: v.get("pct_time_in_market") for k, v in sm.items()},
            "hmm_switch_vs_buy_and_hold_sharpe_diff": b,
            "verdict_hmm_switch": verdict(b),
            "verdict_rules_switch": verdict(ev["bootstrap_sharpe_differences"]["rules_switch_minus_buy_and_hold"]),
            "k_selected_by_year": ev["k_selected_by_year"],
            "lookahead_check_all_pass": la["all_pass"],
            "flags": flags,
        })
    out["summary"]["strategies_tested_per_instrument"] = [
        "buy_and_hold", "trend", "meanrev", "hmm_switch", "rules_switch", "hmm_switch_incl_long"]
    out["summary"]["multiple_testing_note"] = (
        "6 strategy variants x 2 instruments reported; no parameter was tuned on OOS data, but the "
        "variants were designed by hand, so the best OOS figure carries selection optimism.")
    (RES / "results.json").write_text(json.dumps(clean_json(out), indent=1))
    print(json.dumps(clean_json(out["summary"]), indent=1))
    print(f"wrote {RES / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
