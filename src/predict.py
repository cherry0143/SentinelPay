"""
Inference Pipeline for Single and Batch Fraud Predictions.

Loads saved model bundle and applies identical preprocessing and thresholding.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FraudPredictor:
    """Production-ready inference wrapper for Credit Card Fraud Detection."""

    def __init__(self, model_path: str = "models/best_model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Please run 'python3 -m src.train' first to train and persist the model."
            )
        self.artifact = joblib.load(model_path)
        self.model_name = self.artifact["model_name"]
        self.model = self.artifact["model"]
        self.preprocessor = self.artifact["preprocessor"]
        self.feature_names = self.artifact["feature_names"]
        self.threshold = float(self.artifact.get("classification_threshold", 0.5))
        self.metrics = self.artifact.get("test_metrics", {})
        self.feature_importance = self.artifact.get("feature_importance", {})
        logger.info(
            "FraudPredictor initialized: %s (Threshold=%.4f, %d features)",
            self.model_name,
            self.threshold,
            len(self.feature_names),
        )

    def validate_input_dict(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if all required features exist in dictionary."""
        missing = [f for f in self.feature_names if f not in record]
        return len(missing) == 0, missing

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """Validate DataFrame schema against expected feature list."""
        missing = [f for f in self.feature_names if f not in df.columns]
        extra = [c for c in df.columns if c not in self.feature_names and c not in ["CUST_ID", "is_fraud"]]
        is_valid = len(missing) == 0
        return is_valid, missing, extra

    def calculate_risk_level(self, proba: float) -> str:
        """Assign human-readable risk category based on probability and threshold."""
        if proba < self.threshold * 0.5:
            return "Low Risk"
        elif proba < self.threshold:
            return "Moderate Risk (Below Cutoff)"
        elif proba < self.threshold + (1 - self.threshold) * 0.4:
            return "High Risk (Flagged Fraud)"
        else:
            return "Critical Risk (Severe Anomaly)"

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction and risk score for a single transaction/account record.
        """
        # Validate or handle missing features gracefully
        is_valid, missing = self.validate_input_dict(record)
        if not is_valid:
            logger.warning("Record missing features: %s. Using default 0.0 values.", missing)

        # Build single-row DataFrame in the exact expected feature order
        row_data = {feat: [record.get(feat, 0.0)] for feat in self.feature_names}
        df_row = pd.DataFrame(row_data)

        # Preprocess and score
        X_trans = self.preprocessor.transform(df_row)
        proba = float(self.model.predict_proba(X_trans)[0, 1])
        predicted_class = int(proba >= self.threshold)
        label = "Suspicious / Fraudulent" if predicted_class == 1 else "Legitimate"
        risk_level = self.calculate_risk_level(proba)

        # Extract top driving features for this prediction
        driver_insights = []
        if record.get("CASH_ADVANCE", 0) > 3000:
            driver_insights.append("Unusually high cash advance withdrawal")
        if record.get("ONEOFF_PURCHASES", 0) > 4000:
            driver_insights.append("Extreme single-purchase spike")
        if record.get("BALANCE", 0) > 4000 and record.get("PAYMENTS", 0) < 500:
            driver_insights.append("High debt accumulation with disproportionately low repayments")

        return {
            "prediction": predicted_class,
            "label": label,
            "fraud_probability": round(proba, 4),
            "classification_threshold": round(self.threshold, 4),
            "risk_level": risk_level,
            "is_anomaly": bool(predicted_class == 1),
            "model_used": self.model_name,
            "key_drivers": driver_insights if driver_insights else ["Transaction features fall within standard risk boundaries."],
            "disclaimer": "This is an algorithmic machine learning risk assessment and should be verified by a fraud investigation analyst.",
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a batch DataFrame, validating columns and appending prediction columns.
        """
        is_valid, missing, _ = self.validate_dataframe(df)
        if not is_valid:
            raise ValueError(
                f"Batch file is missing required feature columns: {missing}. "
                f"Expected features: {self.feature_names}"
            )

        # Select only required features in strictly preserved order
        X_input = df[self.feature_names].copy()
        X_trans = self.preprocessor.transform(X_input)

        probas = self.model.predict_proba(X_trans)[:, 1]
        classes = (probas >= self.threshold).astype(int)

        result_df = df.copy()
        result_df["fraud_probability"] = np.round(probas, 4)
        result_df["predicted_class"] = classes
        result_df["predicted_label"] = np.where(classes == 1, "Suspicious / Fraudulent", "Legitimate")
        result_df["risk_level"] = [self.calculate_risk_level(p) for p in probas]

        logger.info(
            "Batch prediction completed: %d total rows, %d flagged as suspicious (%.2f%%)",
            len(result_df),
            int(classes.sum()),
            float(classes.mean() * 100),
        )
        return result_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Credit Card Fraud Prediction Engine")
    parser.add_argument("--batch", type=str, help="Path to CSV file for batch prediction")
    parser.add_argument("--out", type=str, default="batch_predictions.csv", help="Output path for batch predictions")
    args = parser.parse_args()

    predictor = FraudPredictor()

    if args.batch:
        batch_df = pd.read_csv(args.batch)
        pred_df = predictor.predict_batch(batch_df)
        pred_df.to_csv(args.out, index=False)
        print(f"Predictions saved to {args.out}")
    else:
        # Demo single prediction
        sample = {
            "BALANCE": 5500.0,
            "BALANCE_FREQUENCY": 1.0,
            "PURCHASES": 6200.0,
            "ONEOFF_PURCHASES": 5000.0,
            "INSTALLMENTS_PURCHASES": 1200.0,
            "CASH_ADVANCE": 4500.0,
            "PURCHASES_FREQUENCY": 0.9,
            "ONEOFF_PURCHASES_FREQUENCY": 0.8,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.6,
            "CASH_ADVANCE_FREQUENCY": 0.5,
            "CASH_ADVANCE_TRX": 15,
            "PURCHASES_TRX": 85,
            "CREDIT_LIMIT": 12000.0,
            "PAYMENTS": 12000.0,
            "MINIMUM_PAYMENTS": 2800.0,
            "PRC_FULL_PAYMENT": 0.1,
            "TENURE": 12,
        }
        res = predictor.predict_single(sample)
        print("Sample Prediction Result:")
        print(json.dumps(res, indent=2))
