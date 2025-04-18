import matplotlib.pyplot as plt

def plot_predictions(dates, actual, predicted, ticker):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual Price")
    plt.plot(dates, predicted, label="Predicted Price")
    plt.title(f"Stock Price Prediction for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
