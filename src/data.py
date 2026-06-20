import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import jarque_bera

warnings.filterwarnings("ignore")

# ── Paths 

ROOT    = Path(__file__).resolve().parent.parent   # project root
RAW_DIR = ROOT / "data" / "raw"
RES_DIR = ROOT / "results"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── Config 

TICKERS    = ["SPY", "AAPL", "JPM"]
START_DATE = "2013-01-01"
END_DATE   = "2025-12-31"
FRED_SERIES = {
    "VIX":   "VIXCLS",    # CBOE Volatility Index
    "DGS10": "DGS10",     # 10-Year Treasury Yield
}
WINSOR_LOW  = 0.001   # 0.1th percentile
WINSOR_HIGH = 0.999   # 99.9th percentile

# ── Helper 

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── 1. Download price data 

def download_prices():
    section("1. Downloading price data via yfinance")

    raw = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   
        progress=False,
    )["Close"]             

    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"
    raw.columns.name = "ticker"

    print(f"  Tickers:    {list(raw.columns)}")
    print(f"  Date range: {raw.index[0].date()} → {raw.index[-1].date()}")
    print(f"  Shape:      {raw.shape}  (trading days × tickers)")
    print(f"  Missing values:\n{raw.isnull().sum()}")

    raw.to_parquet(RAW_DIR / "prices_raw.parquet")
    print(f"\n  Saved → data/raw/prices_raw.parquet")
    return raw


# ── 2. Download macro data from FRED

def download_macro(spy_index):
    """
    spy_index: the DatetimeIndex from SPY prices.
    We use this to align macro data to trading days only.
    """
    section("2. Downloading macro data via FRED")

    load_dotenv()              
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not found. Check your .env file."
        )

    fred = Fred(api_key=api_key)

    frames = {}
    for col_name, series_id in FRED_SERIES.items():
        raw_series = fred.get_series(
            series_id,
            observation_start=START_DATE,
            observation_end=END_DATE,
        )
        raw_series.index = pd.to_datetime(raw_series.index)
        raw_series.name  = col_name
        frames[col_name] = raw_series
        print(f"  {col_name} ({series_id}): {len(raw_series)} observations")

    macro = pd.DataFrame(frames)

    # FRED series include weekends and holidays where markets are closed.
    # We reindex to the SPY trading calendar, then forward-fill gaps
    # (e.g. weekends, public holidays) up to a maximum of 5 business days.
    # Forward-fill means: if Monday's value is missing, use Friday's.
    macro = (
        macro
        .reindex(spy_index)
        .ffill(limit=5)
    )

    missing_after = macro.isnull().sum()
    print(f"\n  After alignment to SPY calendar:")
    print(f"  Shape: {macro.shape}")
    print(f"  Remaining missing values:\n{missing_after}")

    macro.to_parquet(RAW_DIR / "macro.parquet")
    print(f"\n  Saved → data/raw/macro.parquet")
    return macro


# ── 3. Compute log returns

def compute_returns(prices):
    """
    Log return at time t: r_t = ln(P_t / P_{t-1})

    We use log returns rather than simple percentage returns because:
    1. They are additive over time: a 2-day return = sum of two daily log returns
    2. They are symmetric: +10% and -10% log returns cancel exactly
    3. They are approximately stationary — necessary for GARCH
    """
    section("3. Computing log returns")

    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Winsorise: clip extreme outliers at the 0.1th and 99.9th percentiles.
    # This prevents a single data error or truly extreme outlier from dominating the model. We document the percentile thresholds above.
    for col in log_returns.columns:
        low  = log_returns[col].quantile(WINSOR_LOW)
        high = log_returns[col].quantile(WINSOR_HIGH)
        n_clipped = ((log_returns[col] < low) | (log_returns[col] > high)).sum()
        log_returns[col] = log_returns[col].clip(lower=low, upper=high)
        print(f"  {col}: winsorised {n_clipped} observations "
              f"(bounds: [{low:.4f}, {high:.4f}])")

    print(f"\n  Return shape: {log_returns.shape}")
    print(f"\n  Summary statistics:")
    print(log_returns.describe().round(5).to_string())

    log_returns.to_parquet(RAW_DIR / "returns.parquet")
    print(f"\n  Saved → data/raw/returns.parquet")
    return log_returns


# ── 4. Pre-modelling diagnostics

def run_diagnostics(returns, macro):

    section("4. Pre-modelling diagnostics")

    spy = returns["SPY"]

    # ── Test 1: Augmented Dickey-Fuller 

    print("\n  [Test 1] Augmented Dickey-Fuller — stationarity of SPY log returns")
    adf_result = adfuller(spy, autolag="AIC")
    adf_stat, adf_p = adf_result[0], adf_result[1]
    adf_conclusion = "STATIONARY" if adf_p < 0.05 else "NON-STATIONARY"
    print(f"  ADF statistic : {adf_stat:.4f}")
    print(f"  p-value       : {adf_p:.6f}")
    print(f"  Conclusion    : {adf_conclusion}")
    print(f"  (Report this: p={adf_p:.4f}, reject H0 of unit root → returns are stationary)")

    # ── Test 2: Ljung-Box on squared returns

    print("\n  [Test 2] Ljung-Box — autocorrelation in squared returns (volatility clustering)")
    lb_result = acorr_ljungbox(spy**2, lags=[5, 10, 20], return_df=True)
    print(lb_result.round(6).to_string())
    lb_p_lag10 = lb_result.loc[10, "lb_pvalue"]
    lb_conclusion = "CLUSTERING CONFIRMED ✓" if lb_p_lag10 < 0.05 else "NO CLUSTERING DETECTED ✗"
    print(f"\n  Conclusion: {lb_conclusion}")
    print(f"  (Report: Ljung-Box Q(10) p={lb_p_lag10:.4f} → significant autocorrelation in r² → GARCH justified)")

    # ── Test 3: Jarque-Bera normality test
    # Tests whether the return distribution is Gaussian (normal).

    print("\n  [Test 3] Jarque-Bera — normality of SPY log returns")
    jb_stat, jb_p = jarque_bera(spy)
    jb_conclusion = "NON-NORMAL ✓ (fat tails confirmed)" if jb_p < 0.05 else "NORMAL ✗ (unexpected)"
    print(f"  JB statistic : {jb_stat:.4f}")
    print(f"  p-value      : {jb_p:.6f}")
    print(f"  Skewness     : {spy.skew():.4f}")
    print(f"  Kurtosis     : {spy.kurtosis():.4f}  (excess; Normal = 0)")
    print(f"  Conclusion   : {jb_conclusion}")
    print(f"  (Report: reject normality → Student-t GARCH variant justified)")

    # ── Test 4: ACF/PACF plots 
    # Visual complement to Ljung-Box.
    # ACF of r_t:  should show near-zero autocorrelation (returns are unpredictable)
    # ACF of r_t²: should show slow decay (volatility has memory — clustering)

    print("\n  [Test 4] ACF/PACF plots — saving to results/diagnostic_acf.png")
    _plot_acf_pacf(spy)

    print("\n  All diagnostics complete.")


def _plot_acf_pacf(spy):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pre-modelling Diagnostics — SPY Log Returns", fontsize=13, y=1.01)

    plot_acf(spy,    lags=40, ax=axes[0, 0], title="ACF of returns $r_t$")
    plot_acf(spy**2, lags=40, ax=axes[0, 1], title="ACF of squared returns $r_t^2$")
    plot_pacf(spy**2,lags=40, ax=axes[1, 0], title="PACF of squared returns $r_t^2$")

    # Return distribution vs Normal overlay
    ax = axes[1, 1]
    spy.plot.hist(bins=80, density=True, alpha=0.6, color="#185FA5", ax=ax, label="SPY returns")
    from scipy.stats import norm
    x = np.linspace(spy.min(), spy.max(), 300)
    ax.plot(x, norm.pdf(x, spy.mean(), spy.std()), "r-", lw=1.5, label="Normal fit")
    ax.set_title("Return distribution vs Normal")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = RES_DIR / "diagnostic_acf.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → results/diagnostic_acf.png")


# ── Main

def main():
    print("\nSTEP 1 — Data download and diagnostics")
    print("Starting:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))

    prices = download_prices()
    macro  = download_macro(spy_index=prices.index)
    returns = compute_returns(prices)
    run_diagnostics(returns, macro)

    section("Summary")
    print(f"  Returns shape : {returns.shape}")
    print(f"  Macro shape   : {macro.shape}")
    print(f"  Date range    : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"\n  Files saved to data/raw/:")
    for f in sorted(RAW_DIR.glob("*.parquet")):
        print(f"    {f.name}")
    print(f"\n  Plots saved to results/")
    print(f"\n✓ step1_data.py complete.\n")


if __name__ == "__main__":
    main()