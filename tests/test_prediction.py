"""
Unit tests for model artifact persistence and inference engine.
"""

import os
import pandas as pd
import pytest
from src.predict import FraudPredictor


@pytest.fixture
def predictor():
    """Load the trained model artifact."""
    assert os.path.exists("models/best_model.joblib"), "Model artifact must exist before running tests"
    return FraudPredictor("models/best_model.joblib")


def test_predictor_initialization(predictor):
    """Verify predictor attributes loaded correctly from artifact."""
    assert predictor.model_name in ["Logistic Regression", "Random Forest", "HistGradientBoosting"]
    assert len(predictor.feature_names) == 17
    assert 0.0 < predictor.threshold < 1.0
    assert hasattr(predictor.model, "predict_proba")
    assert hasattr(predictor.preprocessor, "transform")


def test_single_prediction_valid_record(predictor):
    """Verify single prediction returns all required keys and valid types."""
    sample = {
        "BALANCE": 3200.0,
        "BALANCE_FREQUENCY": 1.0,
        "PURCHASES": 1500.0,
        "ONEOFF_PURCHASES": 1000.0,
        "INSTALLMENTS_PURCHASES": 500.0,
        "CASH_ADVANCE": 1200.0,
        "PURCHASES_FREQUENCY": 0.8,
        "ONEOFF_PURCHASES_FREQUENCY": 0.5,
        "PURCHASES_INSTALLMENTS_FREQUENCY": 0.5,
        "CASH_ADVANCE_FREQUENCY": 0.2,
        "CASH_ADVANCE_TRX": 3,
        "PURCHASES_TRX": 25,
        "CREDIT_LIMIT": 6000.0,
        "PAYMENTS": 2500.0,
        "MINIMUM_PAYMENTS": 800.0,
        "PRC_FULL_PAYMENT": 0.0,
        "TENURE": 12,
    }
    result = predictor.predict_single(sample)

    assert "prediction" in result
    assert result["prediction"] in [0, 1]
    assert "label" in result
    assert "fraud_probability" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert "risk_level" in result
    assert "classification_threshold" in result
    assert "key_drivers" in result
    assert isinstance(result["key_drivers"], list)


def test_single_prediction_missing_features(predictor):
    """Ensure missing fields are handled gracefully without exceptions."""
    partial_sample = {
        "BALANCE": 1000.0,
        "PURCHASES": 200.0,
    }
    result = predictor.predict_single(partial_sample)
    assert "prediction" in result
    assert "fraud_probability" in result


def test_batch_prediction_success(predictor):
    """Ensure batch prediction preserves rows and appends required score columns."""
    sample_path = "data/sample_batch_test.csv"
    assert os.path.exists(sample_path)
    df = pd.read_csv(sample_path)

    result_df = predictor.predict_batch(df)

    assert len(result_df) == len(df)
    assert "fraud_probability" in result_df.columns
    assert "predicted_class" in result_df.columns
    assert "predicted_label" in result_df.columns
    assert "risk_level" in result_df.columns
    assert result_df["fraud_probability"].between(0.0, 1.0).all()
    assert set(result_df["predicted_class"].unique()).issubset({0, 1})


def test_batch_prediction_incompatible_columns(predictor):
    """Ensure batch prediction rejects CSV with missing columns."""
    bad_df = pd.DataFrame({"RANDOM_COL_1": [1, 2], "RANDOM_COL_2": [3, 4]})
    with pytest.raises(ValueError, match="Batch file is missing required feature columns"):
        predictor.predict_batch(bad_df)
