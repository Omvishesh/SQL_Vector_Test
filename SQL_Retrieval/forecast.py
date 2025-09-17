from   typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from   statsmodels.tsa.statespace.sarimax import SARIMAX
from   statsmodels.tsa.api import VAR

def _select_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        raise Exception("No numeric columns found in CSV")
    return numeric_columns


def _coerce_numeric_series(series: pd.Series, name_for_error: str) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().all():
        raise Exception(f"Column '{name_for_error}' contains no numeric values")
    return coerced


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    # Heuristic based on column names
    priority_keywords = ["date", "datetime", "timestamp", "time", "period", "ds"]
    candidate_columns = []
    for col in df.columns:
        lower = str(col).lower()
        if any(k in lower for k in priority_keywords):
            candidate_columns.append(col)

    # Try all object/category columns as well
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        if col not in candidate_columns:
            candidate_columns.append(col)

    best_col = None
    best_coverage = 0.0
    for col in candidate_columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        coverage = 1.0 - (parsed.isna().mean())
        unique_count = parsed.nunique(dropna=True)
        if coverage >= 0.6 and unique_count >= 3:  # require reasonable coverage and variation
            if coverage > best_coverage:
                best_col = col
                best_coverage = coverage
    return best_col

def fiscal_to_datetime(x):
    if isinstance(x, str) and "-" in x:
        # Assume fiscal year starts April 1 of the first year
        start_year = int(x[:4])
        return pd.Timestamp(year=start_year, month=4, day=1)
    else:
        # Handle other valid date strings
        return pd.to_datetime(x, errors='coerce')

def _order_by_time_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    idx = pd.to_datetime(df[time_col], errors="coerce")
    if pd.isna(idx[0]):
        idx = df[time_col].apply(fiscal_to_datetime)
    ordered = df.copy()
    ordered = ordered.loc[~idx.isna()].copy()
    ordered["__time_index"] = idx[~idx.isna()].values
    ordered = ordered.sort_values("__time_index")
    # Drop duplicate timestamps keeping the last occurrence
    ordered = ordered.drop_duplicates(subset=["__time_index"], keep="last")
    return ordered

def _infer_future_timestamps(index: pd.DatetimeIndex, steps: int) -> List[str]:
    if len(index) == 0:
        return []
    last = index[-1]
    # Try to infer frequency
    freq = pd.infer_freq(index)
    if freq is not None:
        future = pd.date_range(start=last, periods=steps + 1, freq=freq)[1:]
    else:
        diffs = pd.Series(index).diff().dropna()
        delta = None
        if len(diffs) > 0:
            try:
                delta = diffs.mode().iloc[0]
            except Exception:
                delta = diffs.median()
        if delta is None or pd.isna(delta):
            # Fallback to 1 day
            delta = pd.Timedelta(days=1)
        future = [last + delta * (i + 1) for i in range(steps)]
        future = pd.to_datetime(future)
    return [ts.isoformat() for ts in future]


def run_forecast_core(
    df: pd.DataFrame,
    target_column: str,
    steps_ahead: int,
    model_type: str = "AUTO",
) -> Dict[str, Any]:
    if target_column not in df.columns:
        raise Exception(f"Target column '{target_column}' not found in CSV headers")

    time_column_used: Optional[str] = _detect_time_column(df)
    if time_column_used is not None:
        df = _order_by_time_index(df, time_column_used)

    numeric_columns = _select_numeric_columns(df)
    model_type_norm = (model_type or "AUTO").strip().upper()
    if model_type_norm not in {"ARIMAX", "VAR", "AUTO"}:
        model_type_norm = "AUTO"

    results = None
    model_info: Dict[str, Any] = {}
    exog_columns: List[str] = []
    y = None
    exog_df: Optional[pd.DataFrame] = None
    var_endog_df: Optional[pd.DataFrame] = None
    var_lag_order: int = 1

    feasible_var = len(numeric_columns) >= 2

    # Feature selection (optional: filter irrelevant exog variables)
    exog_columns = [col for col in numeric_columns if col != target_column]
    y = _coerce_numeric_series(df[target_column], target_column).ffill().bfill()
    exog_df = None
    if exog_columns:
        exog_df = df[exog_columns].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    
    if time_column_used and "__time_index" in df.columns:
        time_index = pd.DatetimeIndex(df["__time_index"].values)
        try:
            y.index = time_index
            if exog_df is not None:
                exog_df.index = time_index
        except Exception:
            time_column_used = None

    if model_type_norm == "AUTO":
        chosen_model = "VAR" if feasible_var else "ARIMAX"
    elif model_type_norm == "VAR" and not feasible_var:
        chosen_model = "ARIMAX"
    else:
        chosen_model = model_type_norm

    if chosen_model == "VAR":
        var_endog_df = df[numeric_columns].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
        if time_column_used is not None and "__time_index" in df.columns:
            time_index = pd.DatetimeIndex(df["__time_index"].values)
            try:
                var_endog_df.index = time_index
            except Exception:
                time_column_used = None
        if var_endog_df.shape[1] < 2:
            raise Exception("VAR requires at least 2 numeric columns")
        nobs = int(len(var_endog_df))
        neqs = int(var_endog_df.shape[1])
        # Conservative cap: ensure parameters << observations
        maxlags_cap = max(1, min(5, (nobs - 2) // max(1, (1 + neqs))))

        # Use best lag by info criteria (AIC, BIC)
        selector = VAR(var_endog_df).select_order(maxlags=maxlags_cap)
        lag_candidates = [getattr(selector, crit, None) for crit in ["bic", "aic", "hqic", "fpe"]]
        best_lag = next((int(val) for val in lag_candidates if isinstance(val, int) and 0 < val <= maxlags_cap), 1)
        var_lag_order = int(best_lag if best_lag else 1)
        var_model = VAR(var_endog_df)

        try:
            chosen = None
            if maxlags_cap > 1:
                selector = VAR(var_endog_df).select_order(maxlags=maxlags_cap)
                for crit in ["bic", "aic", "hqic", "fpe"]:
                    val = getattr(selector, crit, None)
                    if isinstance(val, (int, np.integer)) and val and val > 0 and val <= maxlags_cap:
                        chosen = int(val)
                        break
            var_lag_order = int(chosen if chosen else 1)
            var_model = VAR(var_endog_df)
            results = var_model.fit(var_lag_order)
        except Exception as exc:
            # Final fallback to lag=1
            try:
                var_lag_order = 1
                results = VAR(var_endog_df).fit(var_lag_order)
            except Exception:
                raise Exception(f"VAR model fitting failed for small sample: {exc}")
        model_info = {
            "type": "VAR",
            "order": [int(var_lag_order)],
            "endogenous_variables": var_endog_df.columns.tolist(),
            "aic": float(getattr(results, "aic", np.nan)),
            "bic": float(getattr(results, "bic", np.nan)),
        }
    else:
        y = _coerce_numeric_series(df[target_column], target_column)
        y = y.ffill().bfill()
        exog_columns = [col for col in numeric_columns if col != target_column]
        if len(exog_columns) > 0:
            exog_df = df[exog_columns].apply(pd.to_numeric, errors="coerce")
            exog_df = exog_df.ffill().bfill().fillna(0.0)
        if time_column_used is not None and "__time_index" in df.columns:
            time_index = pd.DatetimeIndex(df["__time_index"].values)
            try:
                y.index = time_index
                if exog_df is not None:
                    exog_df.index = time_index
            except Exception:
                time_column_used = None
        model = SARIMAX(
            endog=y.astype(float),
            exog=exog_df.astype(float) if exog_df is not None else None,
            order=(2, 2, 2),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False)
        model_info = {
            "type": "ARIMAX",
            "order": [2, 2, 2],
            "exogenous_variables": exog_columns,
            "aic": float(getattr(results, "aic", np.nan)),
            "bic": float(getattr(results, "bic", np.nan)),
            "sigma2": float(getattr(results, "sigma2", np.nan)),
        }

    # Future exogenous
    future_exog = None
    if exog_df is not None:
        last_row = exog_df.iloc[[-1]]
        future_exog = pd.concat([last_row] * steps_ahead, ignore_index=True)
        future_exog.columns = exog_df.columns

    # Forecast
    if chosen_model == "VAR":
        y_hist = var_endog_df.values[-var_lag_order:]
        if hasattr(results, "forecast_interval"):
            f_mean, f_lower, f_upper = results.forecast_interval(y_hist, steps=steps_ahead)
            f_mean_df = pd.DataFrame(f_mean, columns=var_endog_df.columns)
            f_lower_df = pd.DataFrame(f_lower, columns=var_endog_df.columns)
            f_upper_df = pd.DataFrame(f_upper, columns=var_endog_df.columns)
        else:
            f_mean = results.forecast(y_hist, steps=steps_ahead)
            f_mean_df = pd.DataFrame(f_mean, columns=var_endog_df.columns)
            f_lower_df = pd.DataFrame(np.nan, index=range(steps_ahead), columns=var_endog_df.columns)
            f_upper_df = pd.DataFrame(np.nan, index=range(steps_ahead), columns=var_endog_df.columns)
        if target_column not in f_mean_df.columns:
            raise Exception(f"Target column '{target_column}' not found among numeric columns for VAR")
        point_forecast = f_mean_df[target_column].tolist()
        lower_ci = f_lower_df[target_column].tolist()
        upper_ci = f_upper_df[target_column].tolist()
    else:
        forecast_res = results.get_forecast(steps=steps_ahead, exog=future_exog)
        point_forecast = forecast_res.predicted_mean.tolist()
        conf_int_df = forecast_res.conf_int()
        lower_ci = conf_int_df.iloc[:, 0].tolist()
        upper_ci = conf_int_df.iloc[:, 1].tolist()

    forecast_timestamps: List[str] = []
    if time_column_used is not None:
        time_index_for_forecast = var_endog_df.index if (chosen_model == "VAR" and isinstance(var_endog_df.index, pd.DatetimeIndex)) else (y.index if (y is not None and isinstance(y.index, pd.DatetimeIndex)) else None)
        if time_index_for_forecast is not None:
            forecast_timestamps = _infer_future_timestamps(time_index_for_forecast, steps_ahead)

    history_values: List[float] = []
    history_timestamps: List[str] = []
    if chosen_model == "VAR":
        if target_column in var_endog_df.columns:
            series = var_endog_df[target_column].astype(float)
            history_values = series.tolist()
            if isinstance(var_endog_df.index, pd.DatetimeIndex):
                history_timestamps = [ts.isoformat() for ts in var_endog_df.index]
    else:
        series = pd.Series(y, copy=False).astype(float)
        history_values = series.tolist()
        if isinstance(y.index, pd.DatetimeIndex):
            history_timestamps = [ts.isoformat() for ts in y.index]

    # Build rows for forecast values to append to the original dataframe
    if time_column_used and forecast_timestamps:
        # Use time column for index assignment
        forecast_rows = []
        for i, ts in enumerate(forecast_timestamps):
            row = {col: np.nan for col in df.columns}
            row[time_column_used] = pd.to_datetime(ts)
            row[target_column] = point_forecast[i]
            forecast_rows.append(row)
        df_forecast = pd.DataFrame(forecast_rows, columns=df.columns)
    else:
        # No time column: append rows with only target_column and NaN elsewhere
        forecast_rows = []
        for val in point_forecast:
            row = {col: np.nan for col in df.columns}
            row[target_column] = val
            forecast_rows.append(row)
        df_forecast = pd.DataFrame(forecast_rows, columns=df.columns)

    # Concatenate original df with forecasted rows
    df_appended = pd.concat([df, df_forecast], ignore_index=True)

    return {
        "target_column": target_column,
        "model": model_info,
        "steps_ahead": int(steps_ahead),
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "history_values": history_values,
        "history_timestamps": history_timestamps,
        "forecast_values": point_forecast,
        "forecast_timestamps": forecast_timestamps,
        "n_obs": int(len(var_endog_df) if (chosen_model == "VAR" and var_endog_df is not None) else (len(y) if y is not None else 0)),
        "df_with_forecast": df_appended
    }