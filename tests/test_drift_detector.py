import json

import numpy as np
import pandas as pd
import pytest

from src.monitor import drift_detector as detector


def test_psi_is_near_zero_for_identical_distributions():
    values = np.arange(100, dtype=float)

    psi = detector.calculate_psi(values, values.copy())

    assert psi >= 0
    assert psi < 1e-9


def test_psi_detects_a_large_distribution_shift():
    reference = np.arange(100, dtype=float)
    current = np.arange(100, dtype=float) + 1000

    psi = detector.calculate_psi(reference, current)

    assert psi >= detector.PSI_DRIFT_THRESHOLD


def test_psi_rejects_empty_samples():
    with pytest.raises(ValueError, match="non-empty numeric samples"):
        detector.calculate_psi([], [1, 2, 3])


def test_comparison_classifies_large_shift_as_drift():
    reference = np.arange(100, dtype=float)
    current = np.arange(100, dtype=float) + 1000

    result = detector.compare_distributions(reference, current)

    assert result is not None
    assert result["status"] == "DRIFT"
    assert result["psi"] >= detector.PSI_DRIFT_THRESHOLD


def test_load_events_skips_malformed_json(tmp_path):
    log_path = tmp_path / "events.jsonl"
    log_path.write_text(
        '{"features": {"amount": 10}}\n'
        'not-valid-json\n'
        '{"message": "no features"}\n'
        '{"features": {"amount": 20}}\n',
        encoding="utf-8",
    )

    events = detector.load_events(log_path)

    assert len(events) == 2
    assert events[0]["features"]["amount"] == 10
    assert events[1]["features"]["amount"] == 20


def test_missing_log_returns_no_events(tmp_path):
    events = detector.load_events(tmp_path / "missing.jsonl")

    assert events == []


def test_analysis_reports_insufficient_data(tmp_path, monkeypatch):
    reference_path = tmp_path / "reference.csv"
    model_path = tmp_path / "model.joblib"
    log_path = tmp_path / "inference.jsonl"

    reference_data = pd.DataFrame({
        feature: np.arange(50, dtype=float)
        for feature in detector.FEATURES
    })
    reference_data.to_csv(reference_path, index=False)

    # The detector checks that the model artifact exists before
    # checking whether the inference window has enough samples.
    model_path.write_bytes(b"placeholder")

    events = [
        {"features": {feature: 1 for feature in detector.FEATURES},
         "fraud_probability": 0.1},
        {"features": {feature: 2 for feature in detector.FEATURES},
         "fraud_probability": 0.2},
    ]
    log_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    monkeypatch.setattr(detector, "REFERENCE_PATH", reference_path)
    monkeypatch.setattr(detector, "MODEL_PATH", model_path)
    monkeypatch.setattr(detector, "LOG_PATH", log_path)

    report = detector.analyze_drift(window_size=100, min_samples=3)

    assert report["current_count"] == 2
    assert report["data_drift_status"] == "INSUFFICIENT_DATA"
    assert report["prediction_drift_status"] == "INSUFFICIENT_DATA"
    assert report["overall_status"] == "INSUFFICIENT_DATA"
