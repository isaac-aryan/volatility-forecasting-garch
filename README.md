# Equity Volatility Forecasting: GARCH Models vs Machine Learning under Market Regime Shifts

**Full report (PDF): [Google Drive link](https://drive.google.com/file/d/189OTPNREe1AVCh6iFtboGs68M93Xkm4m/view)**

A comparative study of GARCH-family econometric models against Random Forest and XGBoost for 1-day-ahead equity volatility forecasting, evaluated under walk-forward backtesting across calm and crisis market regimes including COVID-19.

## Key finding

Random Forest achieves the lowest overall RMSE (2.269), a 14% improvement over the best GARCH variant (GJR-GARCH, 2.641), confirmed statistically significant via a Diebold-Mariano test (DM = 4.67, p ≈ 0). However, this advantage inverts under market stress: during the COVID-19 crisis regime (VIX > 35), ML model RMSE degrades by 33–58% relative to calm conditions, while GARCH models remain stable or improve. On QLIKE — a loss function that penalises variance under-prediction asymmetrically — GARCH-t outperforms all ML models by 2.5x overall and 35x during the crisis regime.

| Model | RMSE (overall) | QLIKE (overall) | QLIKE (Crisis) |
|---|---|---|---|
| Random Forest | **2.269** | 3.762 | 22.438 |
| XGBoost | 2.289 | 5.561 | 67.215 |
| GJR-GARCH | 2.641 | 1.499 | 2.424 |
| GARCH-t | 2.695 | **1.496** | **1.921** |

## Project structure
volatility-forecasting-garch/
├── src/
│   ├── data.py         # Download SPY/AAPL/JPM + VIX/DGS10, returns, diagnostics
│   ├── metrics.py            # RMSE, MAE, QLIKE, Diebold-Mariano (tested in isolation)
│   ├── feature_eng.py     # 28-feature matrix, target construction, correlation analysis
│   ├── garch.py        # GARCH/EGARCH/GJR-GARCH/GARCH-t via MLE, walk-forward
│   ├── ml.py           # Baselines, Random Forest, XGBoost, walk-forward
│   └── evaluate.py     # Regime segmentation, DM test, SHAP, all plots
├── data/
│   ├── raw/                  # (gitignored)
│   └── processed/            # (gitignored)
├── results/
│   ├── garch/                # GARCH forecasts and parameters
│   ├── ml/                   # ML forecasts and hyperparameters
│   └── *.png                 # All plots 
├── requirements.txt
└── run_pipeline.sh

---

## Methodology overview

**Data.** SPY daily returns (2013 to 2025, 3,268 trading days) via `yfinance`, plus VIX and the 10-year Treasury yield via the FRED API.

**Diagnostics.** Augmented Dickey-Fuller confirms return stationarity (p ≈ 0). Ljung-Box confirms volatility clustering in squared returns (Q(10) = 4,312, p ≈ 0). Jarque-Bera confirms fat-tailed innovations (excess kurtosis = 6.30). Together these motivate the GARCH specification.

**Models.** GARCH(1,1), EGARCH, GJR-GARCH, and GARCH-t fitted via Maximum Likelihood Estimation, benchmarked against Random Forest and XGBoost trained on a 28-feature set spanning lagged returns, rolling volatility windows, VIX regime indicators, and macroeconomic features.

**Validation.** Six-fold expanding-window walk-forward backtesting (2017 to 2023), covering distinct market regimes including the COVID-19 shock. Results are segmented by VIX regime bucket: Calm, Normal, Elevated, and Crisis.

**Evaluation.** RMSE and QLIKE (Patton, 2011) as the two loss functions. Diebold-Mariano test for statistical significance between the best GARCH and best ML model. SHAP TreeExplainer for feature attribution on XGBoost.

---

## Reproducing the results

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your FRED API key
echo "FRED_API_KEY=your_key_here" > .env

# Run the full pipeline
bash run_pipeline.sh
```

Or run each step individually, in order:

```bash
python src/data.py
python src/feature_eng.py
python src/garch.py
python src/ml.py
python src/evaluate.py
```

### Data sources

| Series | Source | Ticker / ID | Frequency |
|---|---|---|---|
| SPY adjusted close | Yahoo Finance via yfinance | SPY | Daily |
| AAPL adjusted close | Yahoo Finance via yfinance | AAPL | Daily |
| JPM adjusted close | Yahoo Finance via yfinance | JPM | Daily |
| CBOE Volatility Index | FRED API | VIXCLS | Daily |
| 10-Year Treasury Yield | FRED API | DGS10 | Daily |

Period: January 2013 to December 2025.

---

## References

Bollerslev, T. (1986). Generalised autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307 to 327.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253 to 263.

Engle, R. F. (1982). Autoregressive conditional heteroskedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987 to 1007.

Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779 to 1801.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347 to 370.

Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246 to 256.