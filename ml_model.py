# 📦 Required installations:
# pip install yfinance pandas scikit-learn xgboost prophet matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

def prepare_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)
    df = df[["Close"]].dropna()
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df

def predict_random_forest(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def predict_xgboost(X_train, y_train, X_test):
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def predict_prophet(df):
    df_prophet = df[['Close']].reset_index()
    df_prophet.columns = ['ds', 'y']

    m = Prophet()
    m.fit(df_prophet[:-30])
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    y_true = df_prophet['y'][-30:].values
    y_pred = forecast['yhat'][-30:].values

    return df_prophet['ds'][-30:].values, y_true, y_pred

def train_and_predict_all(df):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[["Close"]].values)
    y = df["Target"].values

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates = df.index[-len(X_test):]

    rf_preds = predict_random_forest(X_train, y_train, X_test)
    xgb_preds = predict_xgboost(X_train, y_train, X_test)
    prophet_dates, prophet_actual, prophet_preds = predict_prophet(df)

    result_df = pd.DataFrame({
        "Date": dates,
        "Actual": y_test,
        "RandomForest": rf_preds,
        "XGBoost": xgb_preds
    }).set_index("Date")

    prophet_df = pd.DataFrame({
        "Date": prophet_dates,
        "Actual_Prophet": prophet_actual,
        "Prophet": prophet_preds
    }).set_index("Date")

    return result_df.join(prophet_df, how='outer')