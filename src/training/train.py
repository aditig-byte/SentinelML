
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"

for directory in [RAW_DIR, REFERENCE_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SAMPLES = 12_000

FEATURES = [
    "amount",
    "hour",
    "distance_from_home",
    "international",
    "merchant_risk",
    "transactions_last_24h",
]

TARGET = "is_fraud"


# --------------------------------------------------
# 2. Generate reproducible synthetic transactions
# --------------------------------------------------

def generate_transactions(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate simulated transaction features and fraud labels."""

    rng = np.random.default_rng(random_state)

    amount = np.clip(
        rng.lognormal(mean=3.7, sigma=1.0, size=n_samples),
        1,
        5000,
    )

    hour = rng.integers(0, 24, size=n_samples)

    distance = np.clip(
        rng.exponential(scale=25, size=n_samples),
        0,
        300,
    )

    international = rng.binomial(
        1, 0.08, size=n_samples
    )

    merchant_risk = rng.beta(
        a=2, b=5, size=n_samples
    )

    recent_transactions = np.clip(
        rng.poisson(lam=4, size=n_samples),
        0,
        30,
    )

    # Simulated fraud-generating process.
    # This is a synthetic rule, not a real-world fraud model.

    high_amount = (amount > 500).astype(int)
    unusual_distance = (distance > 80).astype(int)

    log_odds = (
        -5.5
        + 1.0 * high_amount
        + 1.4 * unusual_distance
        + 1.3 * international
        + 2.0 * merchant_risk
        + 0.10 * recent_transactions
        + 1.5 * (international * high_amount)
    )

    fraud_probability = 1 / (1 + np.exp(-log_odds))

    is_fraud = rng.binomial(
        1, fraud_probability
    )

    return pd.DataFrame(
        {
            "amount": amount.round(2),
            "hour": hour,
            "distance_from_home": distance.round(2),
            "international": international,
            "merchant_risk": merchant_risk.round(4),
            "transactions_last_24h": recent_transactions,
            "is_fraud": is_fraud,
        }
    )


# --------------------------------------------------
# 3. Train and evaluate the model
# --------------------------------------------------

def main() -> None:
    print("\n=== SentinelML: Model Training ===\n")

    df = generate_transactions()

    raw_path = RAW_DIR / "synthetic_transactions.csv"
    df.to_csv(raw_path, index=False)

    print(f"Dataset shape: {df.shape}")
    print(f"Dataset saved to: {raw_path}")
    print(
        "\nFraud distribution:\n",
        df[TARGET].value_counts().sort_index(),
    )
    print(
        f"\nFraud percentage: {df[TARGET].mean() * 100:.2f}%"
    )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
    }

    print("\n=== Evaluation Metrics ===")

    for name, value in metrics.items():
        print(f"{name:20s}: {value:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    print("=== Confusion Matrix ===")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    # --------------------------------------------------
    # 4. Save model artifact and metadata
    # --------------------------------------------------

    model_path = MODELS_DIR / "fraud_model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "project": "SentinelML",
        "model_type": "RandomForestClassifier",
        "dataset_type": "synthetic",
        "random_state": RANDOM_STATE,
        "n_samples": int(len(df)),
        "n_training_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics,
    }

    metadata_path = MODELS_DIR / "model_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # --------------------------------------------------
    # 5. Save training reference data for drift detection
    # --------------------------------------------------

    reference = X_train.copy()
    reference[TARGET] = y_train

    reference_path = (
        REFERENCE_DIR / "reference_data.csv"
    )

    reference.to_csv(reference_path, index=False)

    print("\n=== Artifacts Saved ===")
    print(f"Model:     {model_path}")
    print(f"Metadata:  {metadata_path}")
    print(f"Reference: {reference_path}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()