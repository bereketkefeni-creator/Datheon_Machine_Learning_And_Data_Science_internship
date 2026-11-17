import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "telco_churn_xgboost.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"


def load_model():
    """Load the trained model pipeline and classification threshold."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        threshold = json.load(file)["threshold"]

    return model, threshold


def predict_customers(input_file, output_file):
    """Predict churn for customers in a CSV file."""

    model, threshold = load_model()

    customers = pd.read_csv(input_file)

    # Get the feature names used during model training
    expected_features = list(model.feature_names_in_)

    # Check that all required features are present
    missing_features = [
        feature for feature in expected_features
        if feature not in customers.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required customer features: {missing_features}"
        )

    # Keep the same features and column order used during training
    X = customers[expected_features]

    # Predict churn probabilities
    probabilities = model.predict_proba(X)[:, 1]

    # Apply the selected classification threshold
    predictions = (probabilities >= threshold).astype(int)

    # Build results
    results = customers.copy()

    results["Churn_Probability"] = probabilities
    results["Churn_Prediction"] = [
        "Yes" if prediction == 1 else "No"
        for prediction in predictions
    ]

    results.to_csv(output_file, index=False)

    print("\nPrediction completed successfully!")
    print(f"Classification threshold: {threshold:.4f}")
    print(f"Customers processed: {len(results)}")
    print(f"Predicted churn: {(predictions == 1).sum()}")
    print(f"Predicted no churn: {(predictions == 0).sum()}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telco Customer Churn Prediction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to customer input CSV file"
    )

    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to save prediction results"
    )

    args = parser.parse_args()

    predict_customers(args.input, args.output)