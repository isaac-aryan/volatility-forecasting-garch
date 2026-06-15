"""
step4_ml.py
===========
Trains baseline and ML models on the SPY feature matrix from step2.
Produces 1-step-ahead log-variance forecasts on both a fixed test split
and six walk-forward folds — identical structure to step3_garch.py.

Models trained:
    Naive baseline     — forecast = yesterday's squared return (log)
    Rolling std baseline — forecast = rolling 21-day variance (log)
    Linear Regression  — OLS on the full feature set
    Random Forest      — 300 trees, tuned via TimeSeriesSplit CV
    XGBoost            — gradient boosted trees, tuned via TimeSeriesSplit CV

Outputs (saved to results/ml/):
    forecasts_fixed_{model}.csv   — fixed split forecasts (2023–2025)
    forecasts_wf_{model}.csv      — walk-forward forecasts (2017–2023)
    best_params_{model}.json      — best hyperparameters from CV
"""

import json
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR  = ROOT / "data" / "raw"
ML_DIR   = ROOT / "results" / "ml"
ML_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────

TICKER = "SPY"

# Fixed split dates — identical to step3 for fair comparison
TRAIN_END  = "2019-12-31"
VAL_END    = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END   = "2025-12-31"

# Walk-forward fold definitions — identical to step3
FOLDS = [
    ("2016-12-31", "2017-01-01", "2017-12-31", "Fold1_2017"),
    ("2017-12-31", "2018-01-01", "2018-12-31", "Fold2_2018"),
    ("2018-12-31", "2019-01-01", "2019-12-31", "Fold3_2019"),
    ("2019-12-31", "2020-01-01", "2020-12-31", "Fold4_2020_COVID"),
    ("2020-12-31", "2021-01-01", "2021-12-31", "Fold5_2021"),
    ("2021-12-31", "2022-01-01", "2023-12-31", "Fold6_2022_2023"),
]

# Random seed for reproducibility — same seed everywhere means
# anyone who clones your repo gets the same results
SEED = 42

# ── Helper ─────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def get_Xy(df):
    """
    Splits the feature DataFrame into X (features) and y (target).

    Parameters
    ----------
    df : DataFrame with a 'target' column and all other columns as features

    Returns
    -------
    X : DataFrame of features (everything except 'target')
    y : Series — the target column (log of next day's squared return)
    """
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]
    return X, y


# ── Baseline models ────────────────────────────────────────────────────────────

def naive_forecast(df):
    """
    Naive baseline: forecast = yesterday's log squared return.

    This is the simplest possible forecast. It says:
    "tomorrow's volatility will be exactly the same as today's."
    If your real models can't beat this, they're worthless.

    We use the 'target' column shifted forward by 1 — meaning
    the forecast for day t is the actual value from day t-1.
    """
    return df["target"].shift(1)


def rolling_std_forecast(returns, index, window=21):
    """
    Rolling std baseline: forecast = log of rolling 21-day variance.

    This is the standard industry heuristic — "take the variance of
    the last 21 days and use that as your forecast for tomorrow."
    It's simple but surprisingly hard to beat.

    We compute rolling variance on the returns series (not the target)
    and take the log to match our log-variance target.

    The .shift(1) ensures we only use data up to yesterday —
    today's return is not included in the rolling window.
    """
    rolling_var = (
        returns
        .shift(1)                    # only use data up to yesterday
        .rolling(window=window)      # 21-day window
        .var()                       # variance (std squared)
    )
    # Take log to match our target scale (log-variance)
    log_rolling_var = np.log(rolling_var.replace(0, np.nan))

    # Align to the index of our feature matrix
    return log_rolling_var.reindex(index)


# ── Train and predict — single window ─────────────────────────────────────────

def train_predict(model, X_train, y_train, X_test, scaler=None):
    """
    Fits a model on training data, predicts on test data.

    Parameters
    ----------
    model    : sklearn estimator (LinearRegression, RF, or XGBoost)
    X_train  : training features (DataFrame)
    y_train  : training target (Series)
    X_test   : test features (DataFrame)
    scaler   : StandardScaler fitted on X_train, or None

    Returns
    -------
    predictions : numpy array of forecasts for the test period

    Why we pass the scaler separately:
    StandardScaler must be fit on training data ONLY, then applied
    (transform) to both train and test. If you fit on the full dataset,
    test-set statistics leak into training — a form of data leakage.
    """
    if scaler is not None:
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
    else:
        X_train_scaled = X_train
        X_test_scaled  = X_test

    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    return predictions


# ── Hyperparameter tuning ──────────────────────────────────────────────────────

def tune_model(model, param_grid, X_train, y_train, model_name):
    """
    Tunes hyperparameters using GridSearchCV with TimeSeriesSplit.

    TimeSeriesSplit is the temporal equivalent of k-fold cross-validation.
    Unlike random k-fold (which would leak future data into training),
    TimeSeriesSplit always trains on earlier data and validates on later data:

        Split 1: train [----]  val [--]
        Split 2: train [------]  val [--]
        Split 3: train [--------]  val [--]

    This preserves temporal ordering — the model never sees future data
    during any fold of the CV process.

    Parameters
    ----------
    model      : sklearn estimator
    param_grid : dict of hyperparameter options to try
    X_train    : training features
    y_train    : training target
    model_name : str for printing

    Returns
    -------
    best_model  : the fitted model with best hyperparameters
    best_params : dict of the winning hyperparameters
    """
    tscv = TimeSeriesSplit(n_splits=5)

    # GridSearchCV tries every combination of parameters in param_grid,
    # evaluates each on 5 temporal CV folds, and picks the best.
    # scoring='neg_mean_squared_error' means it minimises MSE.
    # (sklearn uses negative MSE because it maximises scores internally)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,      # use all CPU cores for parallelism
        verbose=0,
    )
    gs.fit(X_train, y_train)

    print(f"    {model_name} best params: {gs.best_params_}")
    print(f"    {model_name} best CV MSE: {-gs.best_score_:.6f}")

    return gs.best_estimator_, gs.best_params_


# ── Fixed split evaluation ─────────────────────────────────────────────────────

def run_fixed_split(df, returns):
    """
    Trains all models on 2013–2019, tests on 2023–2025.

    This is a simple train/test split — not walk-forward.
    Walk-forward comes next. The fixed split is used for:
    (a) hyperparameter tuning (done here)
    (b) a quick sanity check before the expensive walk-forward loop
    """
    section("Fixed split evaluation (2023–2025 test set)")

    # Split the data temporally
    train_df = df[df.index <= TRAIN_END]
    test_df  = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

    X_train, y_train = get_Xy(train_df)
    X_test,  y_test  = get_Xy(test_df)

    print(f"  Train: {len(X_train)} rows ({X_train.index[0].date()} → "
          f"{X_train.index[-1].date()})")
    print(f"  Test:  {len(X_test)} rows ({X_test.index[0].date()} → "
          f"{X_test.index[-1].date()})")

    # ── Fit scaler on training data only ──────────────────────────────────
    # StandardScaler transforms each feature to have mean=0, std=1.
    # This matters for Linear Regression (coefficients are comparable)
    # and slightly for tree models (though trees are scale-invariant).
    # Critical: fit on training set, transform both train and test.
    scaler = StandardScaler()
    scaler.fit(X_train)

    results = {}

    # ── Baseline 1: Naive ─────────────────────────────────────────────────
    print("\n  ── Naive baseline ──")
    naive_preds = naive_forecast(test_df).dropna()
    results["Naive"] = _build_result_df(
        naive_preds.index, naive_preds.values, test_df, "Naive"
    )

    # ── Baseline 2: Rolling std ───────────────────────────────────────────
    print("  ── Rolling std baseline ──")
    spy_returns = pd.read_parquet(RAW_DIR / "returns.parquet")["SPY"]
    rolling_preds = rolling_std_forecast(spy_returns, test_df.index)
    valid_mask = rolling_preds.notna()
    results["RollingStd"] = _build_result_df(
        rolling_preds[valid_mask].index,
        rolling_preds[valid_mask].values,
        test_df, "RollingStd"
    )

    # ── Linear Regression ─────────────────────────────────────────────────
    print("  ── Linear Regression ──")
    lr = LinearRegression()
    lr_preds = train_predict(lr, X_train, y_train, X_test, scaler)
    results["LinearReg"] = _build_result_df(
        X_test.index, lr_preds, test_df, "LinearReg"
    )

    # ── Random Forest (with tuning) ───────────────────────────────────────
    print("  ── Random Forest (tuning via TimeSeriesSplit) ──")

    # Scale features for tuning
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    rf_param_grid = {
        "n_estimators": [300],          # 300 trees (more is better, diminishing returns)
        "max_depth": [6, 8, None],      # None = unlimited depth
        "min_samples_leaf": [10, 20],   # minimum observations in each leaf node
    }
    rf_best, rf_params = tune_model(
        RandomForestRegressor(random_state=SEED),
        rf_param_grid, X_train_scaled, y_train, "RF"
    )
    rf_preds = rf_best.predict(
        pd.DataFrame(scaler.transform(X_test),
                     columns=X_test.columns, index=X_test.index)
    )
    results["RandomForest"] = _build_result_df(
        X_test.index, rf_preds, test_df, "RandomForest"
    )
    _save_params(rf_params, "RandomForest")

    # ── XGBoost (with tuning) ─────────────────────────────────────────────
    print("  ── XGBoost (tuning via TimeSeriesSplit) ──")
    xgb_param_grid = {
        "n_estimators": [200, 400],       # number of boosting rounds
        "max_depth": [3, 5],              # depth of each tree
        "learning_rate": [0.01, 0.05],    # step size — smaller = more regularised
        "subsample": [0.8],               # fraction of data used per tree
    }
    xgb_best, xgb_params = tune_model(
        xgb.XGBRegressor(random_state=SEED, verbosity=0),
        xgb_param_grid, X_train_scaled, y_train, "XGBoost"
    )
    xgb_preds = xgb_best.predict(
        pd.DataFrame(scaler.transform(X_test),
                     columns=X_test.columns, index=X_test.index)
    )
    results["XGBoost"] = _build_result_df(
        X_test.index, xgb_preds, test_df, "XGBoost"
    )
    _save_params(xgb_params, "XGBoost")

    # Save all fixed split results
    for name, fc_df in results.items():
        out = ML_DIR / f"forecasts_fixed_{name}.csv"
        fc_df.to_csv(out)
        print(f"  ✓ Saved → results/ml/forecasts_fixed_{name}.csv")

    return results, scaler, rf_params, xgb_params


# ── Walk-forward evaluation ────────────────────────────────────────────────────

def run_walk_forward(df, returns, rf_params, xgb_params):
    """
    Runs the 6-fold expanding-window walk-forward evaluation for all ML models.

    For each fold:
    1. Split data into train (everything up to train_end) and test
    2. Fit StandardScaler on training data only
    3. Train each model on scaled training features
    4. Predict the entire test period in one shot

    Unlike GARCH (which refits every single day), ML models are trained
    once per fold on the full training window. This is standard practice:
    tree models don't benefit from daily refitting the way MLE does.
    The expanding window still ensures no future data leaks.
    """
    section("Walk-forward evaluation (6 folds)")

    spy_returns = pd.read_parquet(RAW_DIR / "returns.parquet")["SPY"]

    # Accumulate forecasts across all folds
    all_wf = {
        "Naive": [], "RollingStd": [], "LinearReg": [],
        "RandomForest": [], "XGBoost": [],
    }

    for fold_idx, (train_end, test_start, test_end, fold_label) in enumerate(FOLDS):
        print(f"\n  ── {fold_label} ──")

        train_df = df[df.index <= train_end]
        test_df  = df[(df.index >= test_start) & (df.index <= test_end)]

        if len(test_df) == 0:
            print(f"    Skipping — no test data in range")
            continue

        X_train, y_train = get_Xy(train_df)
        X_test,  y_test  = get_Xy(test_df)

        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

        # Fit scaler on this fold's training data only
        fold_scaler = StandardScaler()
        fold_scaler.fit(X_train)

        X_train_s = pd.DataFrame(
            fold_scaler.transform(X_train),
            columns=X_train.columns, index=X_train.index,
        )
        X_test_s = pd.DataFrame(
            fold_scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )

        # ── Baselines ────────────────────────────────────────────────────
        naive_preds = naive_forecast(test_df).dropna()
        if len(naive_preds) > 0:
            _append_fold(all_wf["Naive"], naive_preds.index,
                         naive_preds.values, test_df, fold_label)

        rolling_preds = rolling_std_forecast(spy_returns, test_df.index)
        valid = rolling_preds.dropna()
        if len(valid) > 0:
            _append_fold(all_wf["RollingStd"], valid.index,
                         valid.values, test_df, fold_label)

        # ── Linear Regression ────────────────────────────────────────────
        lr = LinearRegression()
        lr.fit(X_train_s, y_train)
        lr_preds = lr.predict(X_test_s)
        _append_fold(all_wf["LinearReg"], X_test.index,
                     lr_preds, test_df, fold_label)

        # ── Random Forest (use best params from fixed split) ─────────────
        rf = RandomForestRegressor(random_state=SEED, **rf_params)
        rf.fit(X_train_s, y_train)
        rf_preds = rf.predict(X_test_s)
        _append_fold(all_wf["RandomForest"], X_test.index,
                     rf_preds, test_df, fold_label)
        print(f"    RF: {len(rf_preds)} forecasts")

        # ── XGBoost (use best params from fixed split) ───────────────────
        xgb_model = xgb.XGBRegressor(
            random_state=SEED, verbosity=0, **xgb_params
        )
        xgb_model.fit(X_train_s, y_train)
        xgb_preds = xgb_model.predict(X_test_s)
        _append_fold(all_wf["XGBoost"], X_test.index,
                     xgb_preds, test_df, fold_label)
        print(f"    XGB: {len(xgb_preds)} forecasts")

    # Consolidate and save
    wf_results = {}
    for name, records in all_wf.items():
        if len(records) == 0:
            continue
        fc_df = pd.DataFrame(records).set_index("date")
        out = ML_DIR / f"forecasts_wf_{name}.csv"
        fc_df.to_csv(out)
        print(f"\n  ✓ {name} walk-forward saved → results/ml/forecasts_wf_{name}.csv")
        wf_results[name] = fc_df

    return wf_results


# ── Helper: build result DataFrames ────────────────────────────────────────────

def _build_result_df(dates, predictions, test_df, model_name):
    """
    Creates a standardised forecast DataFrame matching the GARCH format.
    This is important — step5 loads GARCH and ML CSVs and expects
    identical column names.
    """
    fc_df = pd.DataFrame({
        "log_forecast": predictions,
        "target": test_df.loc[dates, "target"].values,
    }, index=dates)
    fc_df.index.name = "date"

    # Also store raw variance (exp of log) for QLIKE computation
    fc_df["forecast_var"] = np.exp(fc_df["log_forecast"])
    fc_df["actual_var"]   = np.exp(fc_df["target"])

    return fc_df


def _append_fold(record_list, dates, predictions, test_df, fold_label):
    """Appends fold results to the accumulator list."""
    for i, date in enumerate(dates):
        if date in test_df.index:
            record_list.append({
                "date":         date,
                "log_forecast": predictions[i],
                "target":       test_df.loc[date, "target"],
                "forecast_var": np.exp(predictions[i]),
                "actual_var":   np.exp(test_df.loc[date, "target"]),
                "fold":         fold_label,
            })


def _save_params(params, model_name):
    """Saves best hyperparameters as JSON for reproducibility."""
    # Convert numpy types to Python types for JSON serialisation
    clean = {k: int(v) if isinstance(v, (np.integer,)) else
                float(v) if isinstance(v, (np.floating,)) else v
             for k, v in params.items()}
    out = ML_DIR / f"best_params_{model_name}.json"
    with open(out, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"    ✓ Params saved → results/ml/best_params_{model_name}.json")


# ── Quick evaluation summary ──────────────────────────────────────────────────

def print_evaluation_summary(wf_results):
    """
    Prints RMSE and QLIKE for each ML model across all walk-forward folds.
    Same format as step3's summary for easy comparison.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from metrics import rmse, qlike

    section("Quick evaluation summary (walk-forward)")

    print(f"\n  {'Model':<15} {'RMSE (log-var)':>15} {'QLIKE':>15}")
    print(f"  {'-'*45}")

    for name, fc_df in wf_results.items():
        # RMSE on log-variance
        valid = fc_df[["target", "log_forecast"]].dropna()
        if len(valid) == 0:
            continue
        r = rmse(valid["target"].values, valid["log_forecast"].values)

        # QLIKE on raw variance (must be positive)
        valid2 = fc_df[["actual_var", "forecast_var"]].dropna()
        valid2 = valid2[(valid2["actual_var"] > 0) & (valid2["forecast_var"] > 0)]
        q = qlike(valid2["actual_var"].values, valid2["forecast_var"].values)

        print(f"  {name:<15} {r:>15.6f} {q:>15.6f}")

    # Also print GARCH results for immediate comparison
    print(f"\n  {'─ GARCH (from step3) ─':─<45}")
    garch_dir = ROOT / "results" / "garch"
    for garch_name in ["GARCH", "EGARCH", "GJR-GARCH", "GARCH-t"]:
        garch_path = garch_dir / f"forecasts_wf_{garch_name}.csv"
        if garch_path.exists():
            gc = pd.read_csv(garch_path, index_col="date", parse_dates=True)
            valid = gc[["target", "log_forecast"]].dropna()
            valid2 = gc[["actual_var", "forecast_var"]].dropna()
            valid2 = valid2[(valid2["actual_var"] > 0) & (valid2["forecast_var"] > 0)]
            if len(valid) > 0 and len(valid2) > 0:
                r = rmse(valid["target"].values, valid["log_forecast"].values)
                q = qlike(valid2["actual_var"].values, valid2["forecast_var"].values)
                print(f"  {garch_name:<15} {r:>15.6f} {q:>15.6f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    section("STEP 4 — Machine learning models (SPY)")

    # Load SPY feature matrix from step2
    df = pd.read_parquet(PROC_DIR / f"features_{TICKER}.parquet")
    print(f"  Loaded features_{TICKER}.parquet: {df.shape[0]} rows, "
          f"{df.shape[1] - 1} features")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")

    # Load returns for baselines
    returns = pd.read_parquet(RAW_DIR / "returns.parquet")["SPY"]

    # Fixed split — includes hyperparameter tuning
    fixed_results, scaler, rf_params, xgb_params = run_fixed_split(df, returns)

    # Walk-forward — uses best params from fixed split
    wf_results = run_walk_forward(df, returns, rf_params, xgb_params)

    # Summary — ML models + GARCH side by side
    print_evaluation_summary(wf_results)

    section("Summary")
    print(f"  Models trained: Naive, RollingStd, LinearReg, RandomForest, XGBoost")
    print(f"  Walk-forward folds: {len(FOLDS)}")
    print(f"\n  Files saved:")
    for f in sorted(ML_DIR.glob("*.csv")):
        print(f"    {f.name}")
    for f in sorted(ML_DIR.glob("*.json")):
        print(f"    {f.name}")

    print(f"\n✓ step4_ml.py complete.\n")


if __name__ == "__main__":
    main()