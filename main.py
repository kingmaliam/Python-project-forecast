from ml_model import prepare_data, train_and_predict_all
from plot import plot_predictions
import pandas as pd

def main():
    for ticker in ["AAPL", "GOOGL", "MSFT"]:
        print(f"\n--- Processing {ticker} ---")
        df = prepare_data(ticker)

        if df.empty:
            print(f"No data found for {ticker}")
            continue

        # Run all models
        results = train_and_predict_all(df, ticker)

        # Extract actual and prediction columns
        actual = results["Actual"].dropna()
        predictions = {col: results[col].dropna() for col in results.columns if col != "Actual"}

        # Align all predictions to the common date range
        aligned_index = actual.index
        for col in predictions:
            aligned_index = aligned_index.intersection(predictions[col].index)

        actual = actual.loc[aligned_index]
        aligned_predictions = {name: preds.loc[aligned_index] for name, preds in predictions.items()}

        # Plot and save the results
        plot_predictions(aligned_index, actual, aligned_predictions, ticker)

        # Show and save the result table
        print("\n Prediction Comparison Table:")
        comparison_df = pd.DataFrame({"Actual": actual})
        for name, preds in aligned_predictions.items():
            comparison_df[name] = preds

        print(comparison_df.round(2))
        comparison_df.to_csv(f"plots/results_{ticker}.csv")

if __name__ == "__main__":
    main()
