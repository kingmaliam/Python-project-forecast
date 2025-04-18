import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler


def prepare_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)
    df = df[["Close"]].dropna()
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def train_and_predict_traditional(df, ticker="AAPL"):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[["Close"]].values)
    y = df["Target"].values

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates = df.index[-len(X_test):]

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    result_df = pd.DataFrame({
        "Date": dates,
        "Actual": y_test,
        "RandomForest": rf_preds,
        "XGBoost": xgb_preds
    }).set_index("Date")

    result_df.to_csv(f"plots/results_traditional_{ticker}.csv")
    return result_df
