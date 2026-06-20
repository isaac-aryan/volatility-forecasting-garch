import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap

from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import rmse, qlike, diebold_mariano

# ── Paths 

ROOT      = Path(__file__).resolve().parent.parent
GARCH_DIR = ROOT / "results" / "garch"
ML_DIR    = ROOT / "results" / "ml"
RES_DIR   = ROOT / "results"
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"

# ── Config

# VIX thresholds
VIX_THRESHOLDS = {"Calm": 15, "Normal": 25, "Elevated": 35}

COLOURS = {
    "GARCH":        "#185FA5",
    "EGARCH":       "#0F6E56",
    "GJR-GARCH":    "#854F0B",
    "GARCH-t":      "#534AB7",
    "Naive":        "#AAAAAA",
    "RollingStd":   "#888888",
    "LinearReg":    "#CCAA00",
    "RandomForest": "#E24B4A",
    "XGBoost":      "#C4531C",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        10,
})

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ── Load all forecasts

def load_all_forecasts():
    
    section("Loading all walk-forward forecasts")
    forecasts = {}

    # GARCH models
    garch_models = ["GARCH", "EGARCH", "GJR-GARCH", "GARCH-t"]
    for name in garch_models:
        path = GARCH_DIR / f"forecasts_wf_{name}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            forecasts[name] = df
            print(f"  Loaded {name}: {len(df)} rows")
        else:
            print(f"  WARNING: {path} not found")

    # ML models
    ml_models = ["Naive", "RollingStd", "LinearReg", "RandomForest", "XGBoost"]
    for name in ml_models:
        path = ML_DIR / f"forecasts_wf_{name}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            forecasts[name] = df
            print(f"  Loaded {name}: {len(df)} rows")
        else:
            print(f"  WARNING: {path} not found")

    return forecasts


# ── VIX regime assignment 

def assign_vix_regime(forecasts):
    """
    Loads VIX data and assigns a regime label to each date.

    Regime labels:
        Calm     — VIX < 15
        Normal   — 15 ≤ VIX < 25
        Elevated — 25 ≤ VIX < 35
        Crisis   — VIX ≥ 35
    """
    section("Assigning VIX regimes")
    macro = pd.read_parquet(RAW_DIR / "macro.parquet")
    vix   = macro["VIX"]

    def bucket(v):
        if pd.isna(v):    return "Unknown"
        if v < 15:        return "Calm"
        elif v < 25:      return "Normal"
        elif v < 35:      return "Elevated"
        else:             return "Crisis"

    for name, df in forecasts.items():
        vix_aligned = vix.reindex(df.index, method="ffill")
        forecasts[name]["vix"]    = vix_aligned.values
        forecasts[name]["regime"] = vix_aligned.apply(bucket).values

    # Print regime distribution using first model as reference
    ref = forecasts[list(forecasts.keys())[0]]
    counts = ref["regime"].value_counts()
    print(f"\n  Regime distribution across walk-forward period:")
    for regime in ["Calm", "Normal", "Elevated", "Crisis"]:
        n = counts.get(regime, 0)
        pct = 100 * n / len(ref)
        print(f"    {regime:<10}: {n:>4} days ({pct:.1f}%)")

    return forecasts


# ── Master comparison table 

def build_comparison_table(forecasts):
    """
    Computes RMSE and QLIKE for every model × regime combination.
    """
    section("Building master comparison table")

    regimes = ["All", "Calm", "Normal", "Elevated", "Crisis"]
    rows = []

    for name, df in forecasts.items():
        row = {"Model": name}

        for regime in regimes:
            if regime == "All":
                sub = df
            else:
                sub = df[df["regime"] == regime]

            if len(sub) < 5:
                row[f"RMSE_{regime}"]  = np.nan
                row[f"QLIKE_{regime}"] = np.nan
                continue

            # RMSE on log-variance
            valid_rmse = sub[["target", "log_forecast"]].dropna()
            if len(valid_rmse) > 0:
                row[f"RMSE_{regime}"] = rmse(
                    valid_rmse["target"].values,
                    valid_rmse["log_forecast"].values
                )

            # QLIKE on raw variance
            valid_q = sub[["actual_var", "forecast_var"]].dropna()
            valid_q = valid_q[
                (valid_q["actual_var"] > 0) & (valid_q["forecast_var"] > 0)
            ]
            if len(valid_q) > 0:
                row[f"QLIKE_{regime}"] = qlike(
                    valid_q["actual_var"].values,
                    valid_q["forecast_var"].values
                )

        rows.append(row)

    comparison_df = pd.DataFrame(rows).set_index("Model")

    # Print formatted table
    print(f"\n  RMSE by regime:")
    rmse_cols = [f"RMSE_{r}" for r in regimes]
    print(comparison_df[rmse_cols].round(4).to_string())

    print(f"\n  QLIKE by regime:")
    qlike_cols = [f"QLIKE_{r}" for r in regimes]
    print(comparison_df[qlike_cols].round(4).to_string())

    # Save
    out = RES_DIR / "model_comparison.csv"
    comparison_df.to_csv(out)
    print(f"\n  ✓ Saved → results/model_comparison.csv")

    return comparison_df


# ── Diebold-Mariano test ───────────────────────────────────────────────────────

def run_dm_test(forecasts):
    """
    Runs the Diebold-Mariano test between best GARCH and best ML model.
    """
    section("Diebold-Mariano test: GJR-GARCH vs RandomForest")

    gjr  = forecasts.get("GJR-GARCH")
    rf   = forecasts.get("RandomForest")

    if gjr is None or rf is None:
        print("  Cannot run DM test — missing forecasts")
        return

    # Align on common dates
    common = gjr.index.intersection(rf.index)
    gjr_sub = gjr.loc[common]
    rf_sub  = rf.loc[common]

    # Compute errors (actual - forecast) in log-variance space
    gjr_errors = (gjr_sub["target"] - gjr_sub["log_forecast"]).dropna()
    rf_errors  = (rf_sub["target"]  - rf_sub["log_forecast"]).dropna()

    # Align again after dropna
    common2 = gjr_errors.index.intersection(rf_errors.index)
    e1 = gjr_errors.loc[common2].values   # GJR-GARCH errors
    e2 = rf_errors.loc[common2].values    # RF errors

    dm_stat, p_val = diebold_mariano(e1, e2, h=1, criterion="mse")

    print(f"\n  GJR-GARCH vs Random Forest (MSE criterion)")
    print(f"  Common observations: {len(e1)}")
    print(f"  DM statistic: {dm_stat:.4f}")
    print(f"  p-value:      {p_val:.4f}")

    if p_val < 0.05:
        winner = "Random Forest" if dm_stat > 0 else "GJR-GARCH"
        print(f"  Result: SIGNIFICANT (p<0.05) → {winner} is statistically better")
    else:
        print(f"  Result: NOT significant (p={p_val:.3f}) → "
              f"difference could be random noise")

    print(f"\n  Interpretation: A positive DM statistic means GJR-GARCH's "
          f"errors² > RF's errors² on average → RF is better on MSE.")

    # Save results
    dm_results = pd.DataFrame([{
        "model_1": "GJR-GARCH",
        "model_2": "RandomForest",
        "dm_statistic": dm_stat,
        "p_value": p_val,
        "significant_5pct": p_val < 0.05,
        "better_model": "RandomForest" if dm_stat > 0 else "GJR-GARCH",
        "n_observations": len(e1),
    }])
    out = RES_DIR / "diebold_mariano_results.csv"
    dm_results.to_csv(out, index=False)
    print(f"  ✓ Saved → results/diebold_mariano_results.csv")

    return dm_stat, p_val


# ── Plot Predicted vs Actual 

def plot_predicted_vs_actual(forecasts):

    section("Generating — Predicted vs Actual")

    models_to_plot = ["GJR-GARCH", "GARCH-t", "RandomForest", "XGBoost"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.flatten()

    # Focus on 2019–2022 to capture COVID and recovery
    zoom_start = "2019-01-01"
    zoom_end   = "2022-12-31"

    for i, name in enumerate(models_to_plot):
        ax = axes[i]
        df = forecasts.get(name)
        if df is None:
            continue

        mask = (df.index >= zoom_start) & (df.index <= zoom_end)
        sub  = df[mask]

        # Plot actual 
        ax.plot(sub.index, sub["target"],
                color="#CCCCCC", linewidth=0.8,
                alpha=0.9, label="Actual log-variance", zorder=1)

        # Plot forecast
        valid = sub[["log_forecast"]].dropna()
        ax.plot(valid.index, valid["log_forecast"],
                color=COLOURS.get(name, "#333333"),
                linewidth=1.2, label=f"{name} forecast", zorder=2)

        # Shade COVID period
        ax.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"),
                   alpha=0.12, color="#E24B4A", label="COVID-19")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel("Log-variance")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Predicted vs Actual Log-Variance (2019–2022)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = RES_DIR / "predicted_vs_actual.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → results/P5_predicted_vs_actual.png")


# ── Plot P6: Rolling RMSE over time ───────────────────────────────────────────

def plot_rolling_rmse(forecasts):
    """
    63-day (quarterly) rolling RMSE for all primary models on one chart.

    This shows WHEN each model struggles relative to others. All models spike during COVID (Fold 4) — the question
    is which spike is tallest and which returns to baseline fastest.

    Window = 63 trading days ≈ one calendar quarter.
    """
    section("Generating — Rolling RMSE over time")

    models_to_plot = ["GJR-GARCH", "GARCH-t", "RandomForest", "XGBoost", "RollingStd"]
    window = 63

    fig, ax = plt.subplots(figsize=(14, 5))

    for name in models_to_plot:
        df = forecasts.get(name)
        if df is None:
            continue

        valid = df[["target", "log_forecast"]].dropna()
        if len(valid) == 0:
            continue

        # Compute squared error at each time step
        sq_err = (valid["target"] - valid["log_forecast"]) ** 2

        # Rolling mean of squared error, then sqrt = rolling RMSE
        rolling_rmse = sq_err.rolling(window=window).apply(
            lambda x: np.sqrt(np.mean(x))
        )

        ax.plot(rolling_rmse.index, rolling_rmse.values,
                color=COLOURS.get(name, "#333333"),
                linewidth=1.2, label=name, alpha=0.85)

    # Shade key events
    ax.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"),
               alpha=0.1, color="#E24B4A", label="COVID-19")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               alpha=0.08, color="#854F0B", label="Rate hike cycle")

    ax.set_title("P6 — Rolling 63-day RMSE: GARCH vs ML (Walk-Forward)",
                 fontsize=11)
    ax.set_ylabel("Rolling RMSE (log-variance units)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = RES_DIR / "P6_rolling_rmse.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → results/P6_rolling_rmse.png")


# ── Regime-segmented bar chart 

def plot_regime_bar_chart(comparison_df):

    section("Generating — Regime-segmented bar chart")

    models = ["GJR-GARCH", "GARCH-t", "RandomForest", "XGBoost", "RollingStd"]
    regimes = ["Calm", "Normal", "Elevated", "Crisis"]
    colours_list = [COLOURS.get(m, "#888888") for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, metric in enumerate(["RMSE", "QLIKE"]):
        ax = axes[ax_idx]
        x  = np.arange(len(regimes))
        n  = len(models)
        width = 0.15
        offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

        for i, (model, colour) in enumerate(zip(models, colours_list)):
            col = [f"{metric}_{r}" for r in regimes]
            vals = []
            for c in col:
                v = comparison_df.loc[model, c] if model in comparison_df.index and c in comparison_df.columns else np.nan
                vals.append(v)

            ax.bar(x + offsets[i], vals, width=width,
                   label=model, color=colour, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.set_title(f"P7 — {metric} by VIX Regime", fontsize=11)
        ax.set_ylabel(metric)
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Model Performance by Market Regime — GARCH vs ML",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = RES_DIR / "regime_bar_chart.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → results/P7_regime_bar_chart.png")


# ── Plot P9: SHAP beeswarm ────────────────────────────────────────────────────

def plot_shap(forecasts):
    """
    SHAP beeswarm and bar chart for XGBoost on the full training data.


    The beeswarm plot shows:
    - Y-axis: features ranked by importance (most important at top)
    - X-axis: SHAP value (positive = pushed forecast higher = more vol)
    - Colour: feature value (red = high, blue = low)

    """
    section("Generating — SHAP feature importance (XGBoost)")

    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(PROC_DIR / "features_SPY.parquet")

    train_df = df[df.index <= "2019-12-31"]
    test_df  = df[(df.index >= "2023-01-01") & (df.index <= "2025-12-31")]

    feature_cols = [c for c in df.columns if c != "target"]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test  = test_df[feature_cols]

    # Scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train),
                              columns=feature_cols, index=X_train.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),
                              columns=feature_cols, index=X_test.index)

    
    import json
    params_path = ML_DIR / "best_params_XGBoost.json"
    with open(params_path) as f:
        best_params = json.load(f)

    # Refit XGBoost
    model = xgb.XGBRegressor(random_state=42, verbosity=0, **best_params)
    model.fit(X_train_s, y_train)

    # SHAP TreeExplainer is the fast, exact SHAP implementation for tree models
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)


    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Bar chart: mean |SHAP| per feature (global importance)
    ax = axes[0]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1][:15]  # top 15

    ax.barh(
        [feature_cols[i] for i in sorted_idx[::-1]],
        mean_abs_shap[sorted_idx[::-1]],
        color="#185FA5", alpha=0.8
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (XGBoost)", fontsize=11)

    # ── Beeswarm: SHAP value vs feature value ────────────────────────────
    ax2 = axes[1]
    top_features = [feature_cols[i] for i in sorted_idx[:10]]  # top 10
    top_idx      = sorted_idx[:10]

    for j, (feat_idx, feat_name) in enumerate(
            zip(top_idx[::-1], top_features[::-1])):
        sv   = shap_values[:, feat_idx]
        fv   = X_test_s.iloc[:, feat_idx].values

        # Normalise feature values for colouring (0 = low, 1 = high)
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
        colours_bees = plt.cm.RdBu_r(fv_norm)

        # Add jitter on y-axis so points don't overlap
        y_jitter = j + np.random.uniform(-0.25, 0.25, size=len(sv))
        ax2.scatter(sv, y_jitter, c=colours_bees, alpha=0.4,
                    s=8, linewidths=0)

    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features[::-1], fontsize=9)
    ax2.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("SHAP value (positive = higher vol forecast)")
    ax2.set_title("SHAP Beeswarm (top 10 features)", fontsize=11)

    # Colourbar
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.6, pad=0.02)
    cbar.set_label("Feature value (low → high)", fontsize=8)

    plt.suptitle("SHAP Feature Attribution: XGBoost Volatility Forecasts",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = RES_DIR / "P9_shap_beeswarm.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → results/shap_beeswarm.png")


# ── Main 

def main():
    section("Evaluation, regime analysis, and interpretation")

    # Load all forecasts
    forecasts = load_all_forecasts()

    # Assign VIX regimes
    forecasts = assign_vix_regime(forecasts)

    # Master comparison table
    comparison_df = build_comparison_table(forecasts)

    # Diebold-Mariano test
    dm_stat, p_val = run_dm_test(forecasts)

    # Generate plots
    plot_predicted_vs_actual(forecasts)
    plot_rolling_rmse(forecasts)
    plot_regime_bar_chart(comparison_df)
    plot_shap(forecasts)

    section("Final summary")
    print(f"\n  Best RMSE overall:  "
          f"{comparison_df['RMSE_All'].idxmin()} "
          f"({comparison_df['RMSE_All'].min():.4f})")
    print(f"  Best QLIKE overall: "
          f"{comparison_df['QLIKE_All'].idxmin()} "
          f"({comparison_df['QLIKE_All'].min():.4f})")

    crisis_rmse = comparison_df["RMSE_Crisis"]
    print(f"\n  Best RMSE in Crisis regime: "
          f"{crisis_rmse.idxmin()} ({crisis_rmse.min():.4f})")

    crisis_qlike = comparison_df["QLIKE_Crisis"]
    print(f"  Best QLIKE in Crisis regime: "
          f"{crisis_qlike.idxmin()} ({crisis_qlike.min():.4f})")

    # Compute regime degradation — the CV bullet placeholder
    print(f"\n  Regime degradation (RMSE: Crisis vs Calm):")
    for model in ["GJR-GARCH", "GARCH-t", "RandomForest", "XGBoost"]:
        if model not in comparison_df.index:
            continue
        calm   = comparison_df.loc[model, "RMSE_Calm"]
        crisis = comparison_df.loc[model, "RMSE_Crisis"]
        if pd.notna(calm) and pd.notna(crisis) and calm > 0:
            pct = 100 * (crisis - calm) / calm
            print(f"    {model:<15}: Calm={calm:.3f} → "
                  f"Crisis={crisis:.3f} (+{pct:.1f}%)")

    print(f"\n  DM test: GJR-GARCH vs RandomForest")
    print(f"    Statistic={dm_stat:.4f}, p={p_val:.4f}")

    print(f"\n  Plots saved:")
    for p in ["P5_predicted_vs_actual.png", "P6_rolling_rmse.png",
              "P7_regime_bar_chart.png", "P9_shap_beeswarm.png"]:
        print(f"    results/{p}")

    print(f"\n✓ evaluate.py complete.\n")


if __name__ == "__main__":
    main()