import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def prepare_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)
    df = df[["Close"]].dropna()
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df

def train_and_predict(df):
    X = df[["Close"]].values
    y = df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")

    return df.index[-len(X_test):], y_test, predictions
