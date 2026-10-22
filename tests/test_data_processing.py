"""
Unit tests for data processing, profiling, and preprocessing pipeline.
"""

import os
import numpy as np
import pandas as pd
import pytest
from src.data_processing import (
    apply_resampling,
    create_preprocessor,
    generate_proxy_fraud_labels,
    load_raw_data,
    prepare_features_and_target,
    profile_dataset,
    split_data,
)


@pytest.fixture
def raw_df():
    """Load the original CSV dataset."""
    assert os.path.exists("CC GENERAL.csv"), "Original dataset CC GENERAL.csv must exist"
    return load_raw_data("CC GENERAL.csv")


def test_load_raw_data(raw_df):
    """Verify raw dataset shape and required columns."""
    assert len(raw_df) == 8950
    assert "CUST_ID" in raw_df.columns
    assert "BALANCE" in raw_df.columns
    assert "PURCHASES" in raw_df.columns


def test_profile_dataset(raw_df):
    """Verify data profiler outputs correct structure and missing counts."""
    profile = profile_dataset(raw_df)
    assert profile["num_rows"] == 8950
    assert profile["num_columns"] == 18
    assert "CREDIT_LIMIT" in profile["missing_values"]
    assert "MINIMUM_PAYMENTS" in profile["missing_values"]
    assert profile["duplicate_rows"] == 0


def test_proxy_fraud_labeling(raw_df):
    """Verify Isolation Forest generates proxy labels with proper contamination."""
    df_labeled, iso_model, stats = generate_proxy_fraud_labels(
        raw_df, contamination=0.03, random_state=42
    )
    assert "is_fraud" in df_labeled.columns
    assert "anomaly_score" in df_labeled.columns
    assert set(df_labeled["is_fraud"].unique()).issubset({0, 1})
    assert stats["fraud_count"] > 0
    # Contamination around 3%
    assert 2.5 <= stats["fraud_percentage"] <= 3.5


def test_leakage_prevention(raw_df):
    """Ensure identifier (CUST_ID), target (is_fraud), and score (anomaly_score) are excluded from features."""
    df_labeled, _, _ = generate_proxy_fraud_labels(raw_df, contamination=0.03, random_state=42)
    X, y, feature_names = prepare_features_and_target(df_labeled, target_col="is_fraud")

    assert "CUST_ID" not in feature_names
    assert "is_fraud" not in feature_names
    assert "anomaly_score" not in feature_names
    assert len(feature_names) == 17
    assert len(X) == len(y) == len(raw_df)


def test_preprocessing_pipeline(raw_df):
    """Verify pipeline handles NaNs via SimpleImputer and scales features with RobustScaler."""
    df_labeled, _, _ = generate_proxy_fraud_labels(raw_df, contamination=0.03, random_state=42)
    X, _, feature_names = prepare_features_and_target(df_labeled)

    preprocessor = create_preprocessor(feature_names)
    X_trans = preprocessor.fit_transform(X)

    assert X_trans.shape == X.shape
    # Ensure no NaN values remain after transformation
    assert not np.isnan(X_trans).any()


def test_stratified_split(raw_df):
    """Ensure train/test split preserves class ratio across partitions."""
    df_labeled, _, _ = generate_proxy_fraud_labels(raw_df, contamination=0.03, random_state=42)
    X, y, _ = prepare_features_and_target(df_labeled)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 7160
    assert len(X_test) == 1790
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.005


def test_smote_resampling(raw_df):
    """Verify SMOTE balances classes on training set."""
    df_labeled, _, _ = generate_proxy_fraud_labels(raw_df, contamination=0.03, random_state=42)
    X, y, feature_names = prepare_features_and_target(df_labeled)
    X_train, _, y_train, _ = split_data(X, y, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor(feature_names)
    X_train_trans = preprocessor.fit_transform(X_train)

    X_res, y_res = apply_resampling(X_train_trans, y_train, method="smote", random_state=42)

    assert len(y_res) > len(y_train)
    # Balanced classes in resampled training set
    assert (y_res == 1).sum() == (y_res == 0).sum()
