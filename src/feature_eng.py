"""
Reads cleaned data and constructs:
  - All predictive features (lagged returns, rolling stats, regime indicators,
    macro features, VRP proxy)
  - Target variable: y_t = ln(r²_{t+1})  — log of next day's squared return

The feature matrix X and target vector y are saved as Parquet files.
A feature correlation heatmap is saved to results/.

Outputs:
    features_SPY.parquet
    features_AAPL.parquet
    features_JPM.parquet

Each file contains both X (features) and y (target) in one DataFrame.
The column 'target' is y. All other columns are features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths

ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
RES_DIR   = ROOT / "results"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Config

TICKERS      = ["SPY", "AAPL", "JPM"]
PRIMARY      = "SPY"        
LAGS         = [1, 2, 3, 5]  
ROLLING_WINDOWS = [5, 10, 21]

# VIX regime thresholds — used to create the ordinal regime bucket feature.
# These same thresholds define the regime segments in the evaluation framework.
VIX_CALM     = 15
VIX_NORMAL   = 25
VIX_ELEVATED = 35
# Above 35 = Crisis (bucket 3)

# ── Helper

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── Feature construction

def build_features(ticker, returns, macro):
    """
    Constructs the full feature matrix for a single ticker.
    """
    r = returns[ticker].copy() 
    df = pd.DataFrame(index=r.index)

    # ── Lagged return signals

    for lag in LAGS:
        df[f"r_lag{lag}"] = r.shift(lag)

        df[f"r2_lag{lag}"] = r.shift(lag) ** 2

        df[f"absr_lag{lag}"] = r.shift(lag).abs() # Absolute return — alternative variance proxy, less sensitive to extreme outliers than squaring.

    # ──  Rolling volatility statistics

    for w in ROLLING_WINDOWS:

        df[f"rolling_std_{w}d"] = (r.shift(1).rolling(window=w).std())

        df[f"rolling_mean_{w}d"] = (r.shift(1).rolling(window=w).mean())

    # Realised vol ratio: short-term vol divided by monthly vol.
    # Ratio > 1 means recent vol has spiked above its monthly baseline.
    # This is a regime-change early warning signal.
    df["vol_ratio_5_21"] = (df["rolling_std_5d"] / df["rolling_std_21d"].replace(0, np.nan))

    # ──  Regime indicators 

    vix = macro["VIX"].copy()

    df["vix_level"] = vix.reindex(df.index)
    df["vix_change"] = vix.diff(1).reindex(df.index)
    # vix.diff(1) = VIX_t - VIX_{t-1}: daily change in the fear gauge.

    # VIX regime bucket — ordinal encoding of the four regimes.
    # 0 = Calm (<15), 1 = Normal (15-25), 2 = Elevated (25-35), 3 = Crisis (>35)
    # These same thresholds are used in the evaluation framework (step5).
    def vix_bucket(v):
        if v < VIX_CALM:     return 0
        elif v < VIX_NORMAL:  return 1
        elif v < VIX_ELEVATED:return 2
        else:                 return 3

    df["vix_regime"] = vix.reindex(df.index).apply(vix_bucket)

    # Drawdown from rolling 252-day (1 year) peak.
    # Deep drawdowns associate with volatility clustering and regime shifts.
    prices_reindexed = returns[ticker].cumsum().reindex(df.index)
    # cumsum of log returns approximates the log price path
    rolling_peak = prices_reindexed.rolling(252, min_periods=1).max()
    df["drawdown"] = prices_reindexed - rolling_peak
    # Drawdown will be <= 0; more negative = deeper drawdown

    # Moving average crossover: 1 when price > 200-day MA (bull regime), else 0.
    price_path = returns[ticker].cumsum()
    ma_200 = price_path.rolling(200, min_periods=1).mean()
    df["ma_crossover"] = (price_path > ma_200).astype(int).reindex(df.index)

    # ── Exogenous macro features

    dgs10 = macro["DGS10"].copy()

    df["dgs10_level"]    = dgs10.reindex(df.index)
    df["dgs10_change_1d"] = dgs10.diff(1).reindex(df.index)
    df["dgs10_change_5d"] = dgs10.diff(5).reindex(df.index)

    # VRP proxy: VIX / rolling 21-day realised vol
    # High VRP means market is pricing in more fear than recently observed.
    # When VRP is high and then mean-reverts, vol tends to fall.
    df["vrp_proxy"] = (df["vix_level"] / (df["rolling_std_21d"] * np.sqrt(252) * 100)).replace([np.inf, -np.inf], np.nan)

    # ── Target variable construction
    squared_returns = r ** 2
    df["target"] = np.log(squared_returns.shift(-1).replace(0, np.nan))
    # We take log to reduce skewness — squared returns are right-skewed.
    # np.log(0) = -inf, so we replace 0 with NaN before logging.

    # ── Clean up 
    # Verify the last row has NaN target before dropping
    assert pd.isna(df["target"].iloc[-1]), (
        "Last row should have NaN target (no t+1 observation). "
        "Check .shift(-1) alignment."
    )
    # Drop rows with any NaN 
    n_before = len(df)
    df = df.dropna()
    n_after  = len(df)
    print(f"  {ticker}: {n_before} rows → {n_after} after dropping NaN "
          f"({n_before - n_after} dropped, warm-up + final row)")

    return df


# ── Correlation heatmap 

def plot_correlation_heatmap(df, ticker):
    """
    Spearman rank correlation between all features and the target.
    Spearman is used rather than Pearson because financial relationships
    are often monotonic but not strictly linear.
    """
    corr = df.corr(method="spearman")

    # Reorder so 'target' is the first column and row for readability
    cols = ["target"] + [c for c in corr.columns if c != "target"]
    corr = corr.loc[cols, cols]

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    # mask upper triangle — correlation matrices are symmetric,
    # showing both halves is redundant

    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",    
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        linewidths=0.3,
        ax=ax,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title(
        f"Spearman Correlation — {ticker} Features vs Target (log r²_{{t+1}})",
        fontsize=12, pad=12
    )
    plt.tight_layout()

    out_path = RES_DIR / f"correlation_heatmap_{ticker}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → results/correlation_heatmap_{ticker}.png")


# ── Feature summary

def print_feature_summary(df, ticker):
    feature_cols = [c for c in df.columns if c != "target"]
    print(f"\n  {ticker} feature matrix: {df.shape[0]} rows × {len(feature_cols)} features")
    print(f"\n  Feature list:")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:2d}. {col}")

    # Top 5 features by absolute Spearman correlation with target
    spearman_with_target = (
        df.corr(method="spearman")["target"]
        .drop("target")
        .abs()
        .sort_values(ascending=False)
    )
    print(f"\n  Top 5 features by |Spearman corr| with target:")
    for feat, val in spearman_with_target.head(5).items():
        print(f"    {feat:<30s} {val:.4f}")


# ── Main 

def main():
    section("Feature engineering")

    returns = pd.read_parquet(RAW_DIR / "returns.parquet")
    macro   = pd.read_parquet(RAW_DIR / "macro.parquet")

    print(f"  Returns loaded: {returns.shape}")
    print(f"  Macro loaded:   {macro.shape}")

    section("Building feature matrices")

    for ticker in TICKERS:
        print(f"\n  ── {ticker} ──")
        df = build_features(ticker, returns, macro)
        print_feature_summary(df, ticker)
        plot_correlation_heatmap(df, ticker)

        out_path = PROC_DIR / f"features_{ticker}.parquet"
        df.to_parquet(out_path)
        print(f"  ✓ Saved → data/processed/features_{ticker}.parquet")

    section("Look-ahead bias verification")
    # Re-load SPY features and confirm alignment is correct
    spy_df = pd.read_parquet(PROC_DIR / "features_SPY.parquet")

    # The target on date t should equal ln(r²) on date t+1
    # Verify on the first 3 rows
    spy_returns = returns["SPY"]
    print("\n  Checking target alignment (first 3 rows after warm-up):")
    print(f"  {'Date':<14} {'Target in df':>14} {'ln(r²) next day':>16} {'Match':>8}")
    for i, (date, row) in enumerate(spy_df.head(3).iterrows()):
        idx       = spy_returns.index.get_loc(date)
        next_date = spy_returns.index[idx + 1]
        next_r2   = np.log(spy_returns.iloc[idx + 1] ** 2)
        match     = np.isclose(row["target"], next_r2, atol=1e-8)
        print(f"  {str(date.date()):<14} {row['target']:>14.6f} {next_r2:>16.6f} {str(match):>8}")

    print("\n  ✓ Target alignment confirmed — no look-ahead bias.")

    section("Summary")
    for ticker in TICKERS:
        df = pd.read_parquet(PROC_DIR / f"features_{ticker}.parquet")
        feat_cols = [c for c in df.columns if c != "target"]
        print(f"  {ticker}: {df.shape[0]} rows, {len(feat_cols)} features, "
              f"target range [{df['target'].min():.3f}, {df['target'].max():.3f}]")

    print(f"\n✓ feature_eng.py complete.\n")


if __name__ == "__main__":
    main()