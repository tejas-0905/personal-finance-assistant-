import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_arima(y, seasonal=True, horizon=12):

    y = y.asfreq('MS')

    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = sm.tsa.ARIMA(y, order=(p, d, q))
                    results = model.fit()

                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                        best_model = results
                except:
                    continue

    fc = best_model.get_forecast(steps=horizon)

    fc_mean = pd.Series(
        fc.predicted_mean,
        index=pd.date_range(
            y.index[-1] + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS"
        )
    )

    fc_mean = fc_mean.clip(lower=0)

    if len(fc_mean) > 1:
        growth_factor = 0.01
        fc_mean = fc_mean * (1 + growth_factor) ** np.arange(1, len(fc_mean) + 1)

    conf_int = fc.conf_int()
    conf_int.index = fc_mean.index

    return best_model, fc_mean, conf_int, f"Best order={best_order}, AIC={best_aic:.2f}"