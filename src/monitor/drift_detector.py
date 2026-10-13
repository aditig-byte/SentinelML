
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    PROJECT_ROOT / "data" / "reference" / "reference_data.csv"
)
LOG_PATH = PROJECT_ROOT / "logs" / "inference.jsonl"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"

FEATURES = [
    "amount",
    "hour",
    "distance_from_home",
    "international",
    "merchant_risk",
    "transactions_last_24h",
]

KS_P_VALUE_THRESHOLD = 0.05
PSI_WARNING_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.20


# --------------------------------------------------
# 2. Population Stability Index
# --------------------------------------------------

def calculate_psi(reference, current, bins=10):
    """Calculate PSI using bins defined by reference quantiles."""

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("PSI requires non-empty numeric samples.")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    # A constant reference distribution needs one bin.
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Smooth empty bins to avoid log(0).
    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(
        np.sum(
            (cur_pct - ref_pct)
            * np.log(cur_pct / ref_pct)
        )
    )


# --------------------------------------------------
# 3. Load inference events
# --------------------------------------------------

def load_events(path):
    """Read valid JSONL inference events; skip malformed lines."""

    events = []

    if not path.exists():
        return events

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                event = json.loads(line)

                if isinstance(event.get("features"), dict):
                    events.append(event)

            except (json.JSONDecodeError, AttributeError):
                continue

    return events


# --------------------------------------------------
# 4. Shared statistical comparison
# --------------------------------------------------

def compare_distributions(reference, current):
    """Compare two numeric samples using KS and PSI."""

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) < 2 or len(current) < 2:
        return None

    ks = ks_2samp(reference, current)
    psi = calculate_psi(reference, current)

    if (
        ks.pvalue < KS_P_VALUE_THRESHOLD
        or psi >= PSI_DRIFT_THRESHOLD
    ):
        status = "DRIFT"
    elif psi >= PSI_WARNING_THRESHOLD:
        status = "WARNING"
    else:
        status = "OK"

    return {
        "ks_p_value": float(ks.pvalue),
        "psi": psi,
        "status": status,
        "reference_count": len(reference),
        "current_count": len(current),
    }


# --------------------------------------------------
# 5. Main monitoring analysis
# --------------------------------------------------

def analyze_drift(window_size=100, min_samples=30):
    """
    Analyze input data drift and prediction drift.

    Returns a dictionary suitable for publishing to monitoring
    systems. Insufficient data is explicitly represented.
    """

    if window_size < 1 or min_samples < 2:
        raise ValueError(
            "window_size must be positive and min_samples must be >= 2."
        )

    report = {
        "data_drift_status": "INSUFFICIENT_DATA",
        "prediction_drift_status": "INSUFFICIENT_DATA",
        "overall_status": "INSUFFICIENT_DATA",
        "feature_psi": {},
        "feature_status": {},
        "data_details": {},
        "prediction_details": None,
        "reference_count": 0,
        "current_count": 0,
        "analysis_timestamp": time.time(),
    }

    for path in [REFERENCE_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    reference = pd.read_csv(REFERENCE_PATH)

    missing = [f for f in FEATURES if f not in reference.columns]
    if missing:
        raise ValueError(
            f"Reference data is missing features: {missing}"
        )

    reference_features = reference[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    reference_features = reference_features.dropna()

    if reference_features.empty:
        raise ValueError("Reference data has no complete feature rows.")

    events = load_events(LOG_PATH)
    events = events[-window_size:]

    report["reference_count"] = len(reference_features)
    report["current_count"] = len(events)

    print("\n=== SentinelML Drift Report ===")
    print(f"Reference rows: {len(reference_features)}")
    print(f"Current window rows: {len(events)}")

    if len(events) < min_samples:
        print(
            f"STATUS: INSUFFICIENT_DATA "
            f"(need at least {min_samples} events)"
        )
        return report

    current = pd.DataFrame(
        [event["features"] for event in events]
    )

    missing = [f for f in FEATURES if f not in current.columns]
    if missing:
        raise ValueError(
            f"Inference events are missing features: {missing}"
        )

    current_features = current[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    # ----------------------------------------------
    # A. Data drift: compare input feature P(X)
    # ----------------------------------------------

    print("\n--- DATA DRIFT: Input Features ---")

    data_statuses = []

    for feature in FEATURES:
        ref = reference_features[feature].dropna().to_numpy()
        cur = current_features[feature].dropna().to_numpy()

        result = compare_distributions(ref, cur)

        if result is None:
            print(f"{feature:28} INSUFFICIENT_NUMERIC_DATA")
            report["feature_status"][feature] = "INSUFFICIENT_DATA"
            continue

        data_statuses.append(result["status"])

        report["feature_psi"][feature] = result["psi"]
        report["feature_status"][feature] = result["status"]
        report["data_details"][feature] = result

        print(
            f"{feature:28} "
            f"KS p-value={result['ks_p_value']:.4f} | "
            f"PSI={result['psi']:.4f} | "
            f"{result['status']}"
        )

    if "DRIFT" in data_statuses:
        data_overall = "DRIFT_DETECTED"
    elif "WARNING" in data_statuses:
        data_overall = "WARNING"
    elif data_statuses:
        data_overall = "NO_DRIFT"
    else:
        data_overall = "INSUFFICIENT_DATA"

    report["data_drift_status"] = data_overall

    print(f"DATA DRIFT STATUS: {data_overall}")

    # ----------------------------------------------
    # B. Prediction drift: compare fraud probabilities
    # ----------------------------------------------

    print("\n--- PREDICTION DRIFT: Fraud Probabilities ---")

    model = joblib.load(MODEL_PATH)

    fraud_class_indices = np.flatnonzero(model.classes_ == 1)

    if len(fraud_class_indices) != 1:
        raise ValueError(
            "The saved model must contain exactly one fraud class (1)."
        )

    fraud_class_index = int(fraud_class_indices[0])

    reference_probabilities = model.predict_proba(
        reference_features[FEATURES]
    )[:, fraud_class_index]

    current_probabilities = []

    for event in events:
        try:
            probability = float(event.get("fraud_probability"))

            if (
                np.isfinite(probability)
                and 0.0 <= probability <= 1.0
            ):
                current_probabilities.append(probability)

        except (TypeError, ValueError):
            continue

    prediction_overall = "INSUFFICIENT_DATA"

    print(
        "Reference prediction samples: "
        f"{len(reference_probabilities)}"
    )
    print(
        "Current prediction samples: "
        f"{len(current_probabilities)}"
    )

    if (
        len(reference_probabilities) >= min_samples
        and len(current_probabilities) >= min_samples
    ):
        result = compare_distributions(
            reference_probabilities,
            current_probabilities,
        )

        if result is not None:
            prediction_overall = result["status"]
            report["prediction_details"] = result

            print(
                f"KS p-value={result['ks_p_value']:.4f} | "
                f"PSI={result['psi']:.4f} | "
                f"STATUS: {prediction_overall}"
            )
    else:
        print(
            "Not enough valid probabilities to evaluate "
            "prediction drift."
        )

    report["prediction_drift_status"] = prediction_overall

    # ----------------------------------------------
    # C. Combined monitoring status
    # ----------------------------------------------

    available_statuses = [
        status
        for status in [data_overall, prediction_overall]
        if status in {"DRIFT", "DRIFT_DETECTED", "WARNING"}
    ]

    if any(
        status in {"DRIFT", "DRIFT_DETECTED"}
        for status in available_statuses
    ):
        overall = "DRIFT_DETECTED"
    elif "WARNING" in available_statuses:
        overall = "WARNING"
    elif (
        data_overall == "NO_DRIFT"
        and prediction_overall in {"OK", "NO_DRIFT"}
    ):
        overall = "NO_DRIFT"
    else:
        overall = "INSUFFICIENT_DATA"

    report["overall_status"] = overall

    print("\n=== Overall Monitoring Summary ===")
    print(f"Data drift:       {data_overall}")
    print(f"Prediction drift: {prediction_overall}")
    print(f"OVERALL STATUS:   {overall}")

    print(
        "\nInterpretation: KS p-values below 0.05 provide evidence "
        "of a distribution difference. PSI thresholds are heuristic, "
        "not universal guarantees. Prediction drift does not establish "
        "accuracy degradation or concept drift."
    )

    return report


# --------------------------------------------------
# 6. Command-line entry point
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=30)
    args = parser.parse_args()

    analyze_drift(
        window_size=args.window_size,
        min_samples=args.min_samples,
    )
