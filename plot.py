import matplotlib.pyplot as plt
import os

def plot_predictions(dates, actual, predictions_dict, ticker):
    """
    Plots actual vs predicted stock prices from multiple models.

    Parameters:
    - dates: pandas.DatetimeIndex
    - actual: pandas.Series of actual prices
    - predictions_dict: dict where keys are model names and values are pandas.Series of predicted prices
    - ticker: string used in plot title and file name
    """
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual Price")

    for model_name, predicted in predictions_dict.items():
        if len(predicted) == len(dates):
            plt.plot(dates, predicted, label=f"{model_name} Prediction")

    plt.title(f"Stock Price Prediction Comparison for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure to file
    plt.savefig(f"plots/forecast_{ticker}.png")
    plt.show()
