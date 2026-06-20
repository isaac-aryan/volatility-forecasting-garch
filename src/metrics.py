"""
Standalone evaluation utilities. No dependencies on other src files.
Import this module from any step file.

Functions:
    rmse(y, y_hat)
    mae(y, y_hat)
    qlike(y, y_hat)
    mape(y, y_hat)
    diebold_mariano(e1, e2, h, criterion)
"""

import numpy as np
from scipy import stats


def _validate(y, y_hat):
    """
    Converts inputs to numpy arrays and checks shapes match.
    Called at the top of every metric function.
    The underscore prefix is a Python convention meaning
    'this is a private helper — don't import it directly'.
    """
    y     = np.asarray(y,     dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError(
            f"Shape mismatch: y={y.shape}, y_hat={y_hat.shape}"
        )
    return y, y_hat


def rmse(y, y_hat):
    """
    Root Mean Squared Error.

    Penalises large errors heavily because errors are squared before
    averaging. A forecast that is occasionally very wrong will have
    a much higher RMSE than one that is consistently slightly wrong.

    Units: same as y (e.g. log-variance units in this project).
    """
    y, y_hat = _validate(y, y_hat)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mae(y, y_hat):
    """
    Mean Absolute Error.

    Takes the average of absolute errors — no squaring, so large
    errors are not penalised disproportionately. More robust to
    outliers than RMSE.

    Units: same as y.
    """
    y, y_hat = _validate(y, y_hat)
    return float(np.mean(np.abs(y - y_hat)))


def qlike(y, y_hat, clip_floor=1e-8):
    """
    QLIKE loss — the primary metric for volatility forecasting.
    Introduced by Patton (2011).

    Formula: mean( y/y_hat - ln(y/y_hat) - 1 )

    Key properties:
    - QLIKE = 0 when y = y_hat (perfect forecast)
    - Asymmetric: penalises under-prediction more than over-prediction
    - QLIKE → infinity as y_hat → 0 (catastrophically underestimating vol)

    This asymmetry is economically justified: a risk manager who
    underestimates volatility is more dangerous than one who overestimates.

    Parameters
    ----------
    y         : realised variance (must be positive)
    y_hat     : forecast variance (must be positive)
    clip_floor: small positive number to prevent division by zero
                if a forecast is numerically zero
    """
    y, y_hat = _validate(y, y_hat)

    # Guard against zero or negative forecasts
    y_hat = np.clip(y_hat, a_min=clip_floor, a_max=None)
    y     = np.clip(y,     a_min=clip_floor, a_max=None)

    ratio = y / y_hat
    return float(np.mean(ratio - np.log(ratio) - 1))


def mape(y, y_hat, clip_floor=1e-8):
    """
    Mean Absolute Percentage Error.

    Expresses the average error as a percentage of the actual value.
    Useful for communicating forecast accuracy to non-technical
    stakeholders ("our forecasts are off by X% on average").

    Note: can be unstable when y is close to zero, hence clip_floor.
    """
    y, y_hat = _validate(y, y_hat)
    y = np.clip(np.abs(y), a_min=clip_floor, a_max=None)
    return float(np.mean(np.abs(y - y_hat) / y) * 100)


def diebold_mariano(e1, e2, h=1, criterion="mse"):
    """
    Diebold-Mariano test for equal predictive accuracy.
    Diebold & Mariano (1995).

    Tests whether two sets of forecast errors have equal expected loss.
    H0: both models have equal predictive accuracy.
    Rejecting H0 means one model is statistically significantly better.

    Parameters
    ----------
    e1        : forecast errors from model 1 (y - y_hat_1), array-like
    e2        : forecast errors from model 2 (y - y_hat_2), array-like
    h         : forecast horizon (1 for 1-step-ahead, which is our case)
    criterion : loss differential criterion
                'mse'  → squared errors (default)
                'mae'  → absolute errors
                'qlike'→ QLIKE loss differentials

    Returns
    -------
    dm_stat : float  — test statistic (positive means model 2 is better)
    p_value : float  — two-sided p-value
                       p < 0.05 → reject H0 → one model significantly better
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    if criterion == "mse":
        d = e1**2 - e2**2
    elif criterion == "mae":
        d = np.abs(e1) - np.abs(e2)
    elif criterion == "qlike":
        # loss differential in QLIKE space
        # requires actual y to compute; use e1, e2 as loss values directly
        d = e1 - e2
    else:
        raise ValueError(f"Unknown criterion '{criterion}'. Use 'mse', 'mae', or 'qlike'.")

    n    = len(d)
    d_bar = np.mean(d)

    # Newey-West variance estimator accounts for autocorrelation
    # in the loss differential series up to lag (h-1).
    # For h=1 (our case) this reduces to the sample variance.
    gamma_0 = np.var(d, ddof=1)
    if h > 1:
        gammas = [
            np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
            for lag in range(1, h)
        ]
        nw_var = gamma_0 + 2 * sum(gammas)
    else:
        nw_var = gamma_0

    dm_stat = d_bar / np.sqrt(nw_var / n)
    p_value = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))

    return float(dm_stat), p_value


# ── Self-test (Run to verify all functions work correctly)

if __name__ == "__main__":
    print("Running metrics.py self-test...\n")

    # Perfect forecast — all metrics should return 0 (or near 0)
    y_true    = np.array([0.01, 0.02, 0.015, 0.03, 0.025])
    y_perfect = y_true.copy()

    print("=== Perfect forecast (y_hat == y) ===")
    print(f"  RMSE  : {rmse(y_true, y_perfect):.8f}  (expect: 0.0)")
    print(f"  MAE   : {mae(y_true,  y_perfect):.8f}  (expect: 0.0)")
    print(f"  QLIKE : {qlike(y_true, y_perfect):.8f}  (expect: 0.0)")
    print(f"  MAPE  : {mape(y_true,  y_perfect):.8f}  (expect: 0.0)")

    # Imperfect forecast
    y_hat = np.array([0.012, 0.018, 0.014, 0.028, 0.030])

    print("\n=== Imperfect forecast ===")
    print(f"  RMSE  : {rmse(y_true, y_hat):.6f}")
    print(f"  MAE   : {mae(y_true,  y_hat):.6f}")
    print(f"  QLIKE : {qlike(y_true, y_hat):.6f}")
    print(f"  MAPE  : {mape(y_true,  y_hat):.2f}%")

    # QLIKE asymmetry check
    # Under-prediction should produce higher QLIKE than over-prediction
    y_base = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
    y_under = y_base * 0.5   # forecast half the actual vol
    y_over  = y_base * 2.0   # forecast double the actual vol

    print("\n=== QLIKE asymmetry (under vs over prediction) ===")
    print(f"  QLIKE under-prediction (y_hat = 0.5y) : {qlike(y_base, y_under):.6f}")
    print(f"  QLIKE over-prediction  (y_hat = 2.0y) : {qlike(y_base, y_over):.6f}")
    print(f"  Under > Over: {qlike(y_base, y_under) > qlike(y_base, y_over)}  (expect: True)")

    # Diebold-Mariano test
    np.random.seed(42)
    e1 = np.random.normal(0, 1, 500)          # model 1 errors
    e2 = np.random.normal(0, 1, 500) * 0.8    # model 2 is genuinely better

    dm_stat, p_val = diebold_mariano(e1, e2, h=1, criterion="mse")
    print("\n=== Diebold-Mariano test ===")
    print(f"  DM statistic : {dm_stat:.4f}")
    print(f"  p-value      : {p_val:.4f}")
    print(f"  Significant  : {p_val < 0.05}  (expect: True — model 2 is better by design)")

    print("\n✓ All self-tests passed.")