import matplotlib.pyplot as plt
import os

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

    # Create plots folder if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Save the figure to the 'plots' folder
    plt.savefig(f"plots/forecast_{ticker}.png")
    plt.show()
