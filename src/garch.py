"""
Fits four GARCH variants on SPY log returns using Maximum Likelihood
Estimation via the arch library. Produces 1-step-ahead conditional
variance forecasts on both a fixed split and six expanding walk-forward
windows.

Models fitted:
    GARCH(1,1)      — baseline, captures volatility clustering
    EGARCH(1,1)     — asymmetric, captures leverage effect
    GJR-GARCH(1,1)  — simpler asymmetric model
    GARCH(1,1)-t    — fat-tailed innovations (Student-t)

Outputs (saved to results/garch/):
    forecasts_fixed_{model}.csv   — fixed split forecasts
    forecasts_wf_{model}.csv      — walk-forward forecasts
    params_{model}.csv            — fitted parameter estimates
    P4_garch_vs_rv_SPY.png        — conditional variance vs actual r²

Walk-forward folds:
    Fold 1: train 2013-2016, test 2017
    Fold 2: train 2013-2017, test 2018
    Fold 3: train 2013-2018, test 2019
    Fold 4: train 2013-2019, test 2020  (COVID stress fold)
    Fold 5: train 2013-2020, test 2021
    Fold 6: train 2013-2021, test 2022-2023
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from arch import arch_model

warnings.filterwarnings("ignore")

# ── Paths 

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
GARCH_DIR = ROOT / "results" / "garch"
GARCH_DIR.mkdir(parents=True, exist_ok=True)

# ── Config 

TICKER = "SPY"  

TRAIN_END   = "2019-12-31"
VAL_END     = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2025-12-31"

# Walk-forward fold definitions (train_end, test_start, test_end, fold_label)
FOLDS = [
    ("2016-12-31", "2017-01-01", "2017-12-31", "Fold1_2017"),
    ("2017-12-31", "2018-01-01", "2018-12-31", "Fold2_2018"),
    ("2018-12-31", "2019-01-01", "2019-12-31", "Fold3_2019"),
    ("2019-12-31", "2020-01-01", "2020-12-31", "Fold4_2020_COVID"),
    ("2020-12-31", "2021-01-01", "2021-12-31", "Fold5_2021"),
    ("2021-12-31", "2022-01-01", "2023-12-31", "Fold6_2022_2023"),
]

# ── Helper 
def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def check_stationarity(params, model_name):
    """
    For GARCH and GJR-GARCH: verify the stationarity condition alpha + beta < 1.
    For EGARCH: verify beta < 1 (different condition due to log specification).

    If this condition is violated, the model implies variance explodes over time
    rather than mean-reverting — which is financially nonsensical.
    """
    p = params

    if model_name == "EGARCH":
        beta = p.get("beta[1]", None)
        if beta is not None:
            condition = abs(beta) < 1
            print(f"    Stationarity: |β| = {abs(beta):.4f} < 1 → "
                  f"{'✓ STATIONARY' if condition else '✗ WARNING: NON-STATIONARY'}")
    else:
        alpha = p.get("alpha[1]", 0)
        beta  = p.get("beta[1]",  0)

        # For GJR-GARCH, persistence also includes half the gamma term
        gamma = p.get("gamma[1]", 0)
        if model_name == "GJR-GARCH":
            persistence = alpha + 0.5 * gamma + beta
        else:
            persistence = alpha + beta

        condition = persistence < 1
        print(f"    Stationarity: α + β = {persistence:.4f} < 1 → "
              f"{'✓ STATIONARY' if condition else '✗ WARNING: NON-STATIONARY'}")
    return condition


def build_model(model_name, returns_scaled):
    """
    Constructs the arch model object for a given model name.
    """

    if model_name == "GARCH":
        # Standard GARCH(1,1) with Normal innovations
        # Mean model: Constant (just fits a mean return, typically ~0)
        # Vol model:  GARCH(p=1, q=1)
        # Dist:       Normal
        am = arch_model(
            returns_scaled,
            mean="Constant",
            vol="GARCH",
            p=1, q=1,
            dist="Normal",
        )

    elif model_name == "EGARCH":
        # EGARCH(1,1) — exponential GARCH
        # Models log(σ²) rather than σ² directly, which:
        # (a) guarantees variance is always positive (no constraint needed)
        # (b) allows asymmetric response via the gamma parameter
        # Negative gamma → bad news increases vol more than good news
        am = arch_model(
            returns_scaled,
            mean="Constant",
            vol="EGARCH",
            p=1, q=1,
            dist="Normal",
        )

    elif model_name == "GJR-GARCH":
        # GJR-GARCH(1,1) — Glosten, Jagannathan & Runkle (1993)
        # Adds an indicator variable: extra variance impact when return < 0
        # Simpler leverage model than EGARCH
        # In arch: use vol="GARCH" with o=1 (one asymmetric term)
        am = arch_model(
            returns_scaled,
            mean="Constant",
            vol="GARCH",
            p=1, o=1, q=1,   # o=1 adds the GJR asymmetric term
            dist="Normal",
        )

    elif model_name == "GARCH-t":
        # GARCH(1,1) with Student-t distributed innovations
        # The t-distribution has heavier tails than Normal,
        # matching the fat tails confirmed by Jarque-Bera in step1.
        # The 'nu' parameter controls tail thickness:
        # lower nu = fatter tails; nu → infinity recovers Normal.
        am = arch_model(
            returns_scaled,
            mean="Constant",
            vol="GARCH",
            p=1, q=1,
            dist="t",        # Student-t innovations
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return am


# ── Fit and forecast — single window 

def fit_and_forecast(model_name, train_returns, n_ahead=1):
    """
    Fits a GARCH model on train_returns and returns one 1-step-ahead
    conditional variance forecast.

    """
    returns_scaled = train_returns * 100   # arch convention

    am  = build_model(model_name, returns_scaled)
    res = am.fit(
        disp="off",         
        show_warning=False,
    )
    
    # Generate 1-step-ahead forecast
    fc = res.forecast(horizon=n_ahead, reindex=False)
    forecast_var = fc.variance.values[0, 0] / (100 ** 2)

    params = dict(res.params)
    return forecast_var, params


# ── Fixed split evaluation 

def run_fixed_split(returns, model_names):
    """
    Fits each model on the training window (2013–2019) and generates
    1-step-ahead forecasts for every day in the test window (2023–2025).

    """
    section("Fixed split evaluation (2023–2025 test set)")

    train_mask = returns.index <= TRAIN_END
    test_mask  = (returns.index >= TEST_START) & (returns.index <= TEST_END)

    test_dates   = returns[test_mask].index
    train_base   = returns[train_mask]

    results = {}

    for model_name in model_names:
        print(f"\n  Fitting {model_name} — {len(test_dates)} test-day forecasts...")
        forecasts = []
        params_list = []

        # Expanding window: start with 2013-2019 training data, add one observation at a time as we move through the test period
        current_train = train_base.copy()

        for i, test_date in enumerate(test_dates):
            if i % 100 == 0:
                print(f"    Progress: {i}/{len(test_dates)} "
                      f"({100*i/len(test_dates):.0f}%)")

            try:
                fc_var, params = fit_and_forecast(model_name, current_train)
                forecasts.append({"date": test_date, "forecast_var": fc_var})
                if i == 0:
                    params_list.append(params)

            except Exception as e:
                print(f"    Warning: {model_name} failed on {test_date}: {e}")
                prev = forecasts[-1]["forecast_var"] if forecasts else np.nan
                forecasts.append({"date": test_date, "forecast_var": prev})

            # Add this test day's actual return to the training window
            current_train = pd.concat([
                current_train,
                returns[returns.index == test_date]
            ])

        fc_df = pd.DataFrame(forecasts).set_index("date")

        # Add actual squared return for comparison
        fc_df["actual_var"] = (returns[test_mask] ** 2).values
        fc_df["target"] = np.log(fc_df["actual_var"].replace(0, np.nan))
        fc_df["log_forecast"] = np.log(fc_df["forecast_var"].replace(0, np.nan))

        out_path = GARCH_DIR / f"forecasts_fixed_{model_name}.csv"
        fc_df.to_csv(out_path)
        print(f"    ✓ Saved → results/garch/forecasts_fixed_{model_name}.csv")

        # Print and check stationarity for the last fitted parameters
        print(f"\n  {model_name} parameters (final training window):")
        if params_list:
            for k, v in params_list[0].items():
                print(f"    {k:<20s}: {v:.6f}")
            check_stationarity(params_list[0], model_name)

        results[model_name] = fc_df

    return results


# ── Walk-forward evaluation

def run_walk_forward(returns, model_names):
    """
    Runs the 6-fold expanding-window walk-forward evaluation.

    For each fold, the model is refit from scratch on the full
    expanding training window, then used to forecast the test period.

    """
    section("Walk-forward evaluation (6 folds)")

    all_wf_forecasts = {m: [] for m in model_names}

    for fold_idx, (train_end, test_start, test_end, fold_label) in enumerate(FOLDS):
        print(f"\n  ── {fold_label} ──")
        print(f"  Train: 2013-01-01 → {train_end}")
        print(f"  Test:  {test_start} → {test_end}")

        train_mask = returns.index <= train_end
        test_mask  = (returns.index >= test_start) & (returns.index <= test_end)

        train_returns = returns[train_mask]
        test_returns  = returns[test_mask]
        test_dates    = test_returns.index

        print(f"  Train size: {len(train_returns)} days | "
              f"Test size: {len(test_returns)} days")

        for model_name in model_names:
            forecasts = []
            current_train = train_returns.copy()

            for i, test_date in enumerate(test_dates):
                try:
                    fc_var, params = fit_and_forecast(model_name, current_train)
                    forecasts.append({
                        "date":        test_date,
                        "forecast_var": fc_var,
                        "fold":        fold_label,
                    })
                except Exception as e:
                    prev = forecasts[-1]["forecast_var"] if forecasts else np.nan
                    forecasts.append({
                        "date":        test_date,
                        "forecast_var": prev,
                        "fold":        fold_label,
                    })

                current_train = pd.concat([
                    current_train,
                    returns[returns.index == test_date]
                ])

            all_wf_forecasts[model_name].extend(forecasts)
            print(f"    {model_name}: {len(forecasts)} forecasts generated")

    wf_results = {}
    for model_name in model_names:
        fc_df = pd.DataFrame(all_wf_forecasts[model_name]).set_index("date")
        fc_df["actual_var"]   = (returns.reindex(fc_df.index) ** 2).values
        fc_df["target"]       = np.log(fc_df["actual_var"].replace(0, np.nan))
        fc_df["log_forecast"] = np.log(fc_df["forecast_var"].replace(0, np.nan))

        out_path = GARCH_DIR / f"forecasts_wf_{model_name}.csv"
        fc_df.to_csv(out_path)
        print(f"\n  ✓ {model_name} walk-forward saved → "
              f"results/garch/forecasts_wf_{model_name}.csv")
        wf_results[model_name] = fc_df

    return wf_results


# ── GARCH conditional variance vs actual r²
def plot_garch_vs_rv(returns, model_name="GARCH"):
    """
    Fits GARCH on the full training window and plots its in-sample
    conditional variance against actual squared returns.

    """
    section(f"Generating Plot P4 — GARCH conditional variance vs actual r²")

    train_mask = returns.index <= TRAIN_END
    train_returns = returns[train_mask] * 100

    am  = build_model(model_name, train_returns)
    res = am.fit(disp="off", show_warning=False)

    # res.conditional_volatility gives σ_t (not σ²_t) in scaled units, Square it and divide by 100² to get variance in return units
    cond_var = (res.conditional_volatility ** 2) / (100 ** 2)

    actual_r2 = returns[train_mask] ** 2

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top panel: full series
    ax = axes[0]
    ax.plot(actual_r2.index, actual_r2.values * 10000,
            color="#AAAAAA", alpha=0.6, linewidth=0.5, label="Actual r² (×10⁴)")
    ax.plot(cond_var.index, cond_var.values * 10000,
            color="#185FA5", linewidth=1.2, label="GARCH(1,1) σ²")
    ax.set_ylabel("Variance (×10⁴)")
    ax.set_title("Plot P4 — GARCH(1,1) Conditional Variance vs Actual Squared Returns (SPY)",
                 fontsize=11)
    ax.legend(fontsize=9)

    # Shade COVID period
    covid_start = pd.Timestamp("2020-02-01")
    covid_end   = pd.Timestamp("2020-06-30")
    ax.axvspan(covid_start, covid_end, alpha=0.15, color="#E24B4A", label="COVID-19")

    # Bottom panel: zoom into 2019–2021 to show COVID spike clearly
    ax2 = axes[1]
    zoom_mask = (actual_r2.index >= "2019-01-01") & (actual_r2.index <= "2021-12-31")
    ax2.plot(actual_r2.index[zoom_mask], actual_r2.values[zoom_mask] * 10000,
             color="#AAAAAA", alpha=0.7, linewidth=0.7, label="Actual r²")
    ax2.plot(cond_var.index[zoom_mask], cond_var.values[zoom_mask] * 10000,
             color="#185FA5", linewidth=1.5, label="GARCH(1,1) σ²")
    ax2.axvspan(covid_start, covid_end, alpha=0.2, color="#E24B4A", label="COVID-19")
    ax2.set_ylabel("Variance (×10⁴)")
    ax2.set_xlabel("Date")
    ax2.set_title("Zoom: 2019–2021 (COVID-19 stress period)", fontsize=10)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = GARCH_DIR / "P4_garch_vs_rv_SPY.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → results/garch/P4_garch_vs_rv_SPY.png")


# ── Quick evaluation summary 

def print_evaluation_summary(wf_results):
    """
    Prints RMSE on log-variance forecasts for each model across all folds.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from metrics import rmse, qlike

    section("Quick evaluation summary (walk-forward, log-variance RMSE)")
    print(f"\n  {'Model':<15} {'Overall RMSE':>14} {'Overall QLIKE':>15}")
    print(f"  {'-'*44}")

    for model_name, fc_df in wf_results.items():
        valid = fc_df[["target", "log_forecast"]].dropna()
        if len(valid) == 0:
            continue

        r = rmse(valid["target"].values, valid["log_forecast"].values)

        # QLIKE requires positive values — use raw variance (not log)
        valid2 = fc_df[["actual_var", "forecast_var"]].dropna()
        valid2 = valid2[(valid2["actual_var"] > 0) & (valid2["forecast_var"] > 0)]
        q = qlike(valid2["actual_var"].values, valid2["forecast_var"].values)

        print(f"  {model_name:<15} {r:>14.6f} {q:>15.6f}")


# ── Main

def main():
    section("GARCH family models")

    returns = pd.read_parquet(RAW_DIR / "returns.parquet")["SPY"]
    print(f"  SPY returns loaded: {len(returns)} observations")
    print(f"  Date range: {returns.index[0].date()} → {returns.index[-1].date()}")

    model_names = ["GARCH", "EGARCH", "GJR-GARCH", "GARCH-t"]

    plot_garch_vs_rv(returns, model_name="GARCH")

    # Walk-forward evaluation across 6 folds
    # Run this before fixed split — it covers more of the sample
    wf_results = run_walk_forward(returns, model_names)

    # Fixed split evaluation on 2023–2025 test set
    fixed_results = run_fixed_split(returns, model_names)

    print_evaluation_summary(wf_results)

    section("Summary")
    print(f"  Models fitted: {model_names}")
    print(f"  Walk-forward folds: {len(FOLDS)}")
    print(f"  Output directory: results/garch/")
    print(f"\n  Files saved:")
    for f in sorted(GARCH_DIR.glob("*.csv")):
        print(f"    {f.name}")
    for f in sorted(GARCH_DIR.glob("*.png")):
        print(f"    {f.name}")

    print(f"\n✓ garch.py complete.\n")


if __name__ == "__main__":
    main()