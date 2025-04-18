from ml_model import prepare_data, train_and_predict

from plot import plot_predictions

def run_model_for_ticker(ticker):
    df = prepare_data(ticker)
    dates, actual, predicted = train_and_predict(df)
    plot_predictions(dates, actual, predicted, ticker)

if __name__ == "__main__":
    tickers = ["AAPL", "GOOGL", "MSFT"]  # Add any tickers here

    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")
        run_model_for_ticker(ticker)
