"""
Data Processing, Profiling, and Feature Engineering Pipeline.

Provides functions to:
- Load and inspect the dataset
- Profile distribution, missingness, and anomalies
- Generate unsupervised proxy fraud labels using Isolation Forest
- Prepare leak-free feature pipelines with imputers and robust scalers
- Execute stratified train/test splitting
- Apply SMOTE resampling strictly on the training set
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str = "CC GENERAL.csv") -> pd.DataFrame:
    """Load the raw dataset from CSV with validation."""
    logger.info("Loading raw dataset from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive dataset profile metrics."""
    missing = df.isnull().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    profile = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": missing_dict,
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_columns": num_cols,
    }
    return profile


def generate_proxy_fraud_labels(
    df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
    id_col: str = "CUST_ID",
) -> Tuple[pd.DataFrame, IsolationForest, Dict[str, Any]]:
    """
    Generate unsupervised proxy fraud labels using Isolation Forest.

    In financial data without pre-existing labels, accounts demonstrating extreme,
    atypical multidimensional transaction dynamics (excessive cash advances,
    unusual purchase bursts, high balance-to-limit saturation) are identified
    as suspicious anomalous behavior (proxy fraud).

    Contamination rate is set to 3.0% by default, matching realistic institutional
    suspicious activity reporting (SAR) and fraud detection baselines.
    """
    logger.info(
        "Generating proxy fraud labels via Isolation Forest (contamination=%.3f, random_state=%d)",
        contamination,
        random_state,
    )

    df_labeled = df.copy()
    feature_cols = [c for c in df.columns if c != id_col]

    # Pre-imputation for anomaly detection fitting
    temp_imputer = SimpleImputer(strategy="median")
    X_imputed = temp_imputer.fit_transform(df_labeled[feature_cols])

    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = iso_forest.fit_predict(X_imputed)
    scores = iso_forest.decision_function(X_imputed)

    # In IsolationForest, -1 indicates outlier (anomaly), 1 indicates normal
    # We map -1 -> 1 (Suspicious / Proxy Fraud) and 1 -> 0 (Legitimate)
    is_fraud = (preds == -1).astype(int)

    df_labeled["is_fraud"] = is_fraud
    df_labeled["anomaly_score"] = scores

    fraud_count = int(is_fraud.sum())
    normal_count = int((is_fraud == 0).sum())
    fraud_pct = float(is_fraud.mean() * 100)

    stats = {
        "total_records": len(df_labeled),
        "legitimate_count": normal_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_pct, 2),
        "contamination": contamination,
        "random_state": random_state,
    }

    logger.info(
        "Proxy fraud labeling complete: %d normal, %d fraud (%.2f%%)",
        normal_count,
        fraud_count,
        fraud_pct,
    )

    return df_labeled, iso_forest, stats


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Split DataFrame into feature matrix X and target vector y.
    Strictly removes identifier and target leakage columns.
    """
    if exclude_cols is None:
        exclude_cols = ["CUST_ID", "anomaly_score", target_col]
    else:
        if target_col not in exclude_cols:
            exclude_cols.append(target_col)

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.info("Features selected (%d): %s", len(feature_cols), feature_cols)
    return X, y, feature_cols


def create_preprocessor(feature_names: List[str]) -> Pipeline:
    """
    Create a robust, reproducible scikit-learn preprocessing pipeline.
    - SimpleImputer with median strategy handles missing values without distortion.
    - RobustScaler scales features using interquartile range (IQR), resistant to extreme outliers.
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    return pipeline


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split to ensure equal fraud ratio in both sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        "Split data into Train (%d samples, %d fraud) and Test (%d samples, %d fraud)",
        len(X_train),
        int(y_train.sum()),
        len(X_test),
        int(y_test.sum()),
    )
    return X_train, X_test, y_train, y_test


def apply_resampling(
    X_train_transformed: np.ndarray,
    y_train: pd.Series,
    method: str = "smote",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply resampling strictly to the training partition.
    Never applied to validation or test partitions to avoid data leakage.
    """
    if method.lower() == "smote":
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)
        logger.info(
            "Applied SMOTE to training data: %d -> %d samples (Fraud count: %d)",
            len(y_train),
            len(y_resampled),
            int(y_resampled.sum()),
        )
        return X_resampled, y_resampled
    elif method.lower() == "none":
        return X_train_transformed, y_train.to_numpy()
    else:
        raise ValueError(f"Unsupported resampling method: {method}")
