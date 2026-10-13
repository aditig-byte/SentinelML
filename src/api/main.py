import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter, time

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.api.inference_logger import log_inference
from src.monitor.drift_detector import analyze_drift


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"

DRIFT_WINDOW_SIZE = 100
DRIFT_MIN_SAMPLES = 30
DRIFT_INTERVAL_SECONDS = 60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinelml.monitor")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run the Phase 1 training script first."
    )

model = joblib.load(MODEL_PATH)

FEATURES = [
    "amount",
    "hour",
    "distance_from_home",
    "international",
    "merchant_risk",
    "transactions_last_24h",
]


# --------------------------------------------------
# 2. Existing inference metrics
# --------------------------------------------------

PREDICTIONS_TOTAL = Counter(
    "sentinelml_predictions",
    "Total successful fraud predictions",
    ["label"],
)

INFERENCE_LATENCY = Histogram(
    "sentinelml_inference_latency_seconds",
    "Time spent generating model predictions, in seconds",
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
    ),
)

PREDICTION_ERRORS_TOTAL = Counter(
    "sentinelml_prediction_errors",
    "Total prediction-handler exceptions",
)


# --------------------------------------------------
# 3. Drift monitoring metrics
# --------------------------------------------------
#
# Status encoding:
# -1 = insufficient data / not yet analyzed
#  0 = no drift detected
#  1 = warning
#  2 = drift detected
#
# These metrics are initialized to -1 so the system
# never claims that drift is absent before analysis.
# --------------------------------------------------

DATA_DRIFT_STATUS = Gauge(
    "sentinelml_data_drift_status",
    "Input data drift status: -1=unknown, 0=OK, 1=warning, 2=drift",
)

PREDICTION_DRIFT_STATUS = Gauge(
    "sentinelml_prediction_drift_status",
    "Prediction distribution drift status: -1=unknown, 0=OK, 1=warning, 2=drift",
)

OVERALL_DRIFT_STATUS = Gauge(
    "sentinelml_overall_drift_status",
    "Overall monitoring status: -1=unknown, 0=OK, 1=warning, 2=drift",
)

FEATURE_PSI = Gauge(
    "sentinelml_feature_psi",
    "Population Stability Index for each monitored input feature",
    ["feature"],
)

DRIFT_ANALYSIS_TIMESTAMP = Gauge(
    "sentinelml_drift_analysis_timestamp_seconds",
    "Unix timestamp of the most recent completed drift analysis",
)

DRIFT_ANALYSIS_ERRORS_TOTAL = Counter(
    "sentinelml_drift_analysis_errors",
    "Total exceptions during drift analysis",
)

DRIFT_ANALYSIS_SUCCESS = Gauge(
    "sentinelml_drift_analysis_success",
    "Whether the most recent drift analysis completed: 1=yes, 0=error",
)

# Initialize status metrics before the first analysis.
DATA_DRIFT_STATUS.set(-1)
PREDICTION_DRIFT_STATUS.set(-1)
OVERALL_DRIFT_STATUS.set(-1)
DRIFT_ANALYSIS_SUCCESS.set(0)


# --------------------------------------------------
# 4. Status conversion and metric publishing
# --------------------------------------------------

def status_to_number(status):
    """Convert detector status strings into numeric metrics."""

    if status is None:
        return -1

    normalized = str(status).strip().upper()

    if normalized in {"OK", "NO_DRIFT", "NO_DRIFT_DETECTED"}:
        return 0

    if normalized == "WARNING":
        return 1

    if normalized in {
        "DRIFT",
        "DRIFT_DETECTED",
        "DATA_DRIFT",
        "PREDICTION_DRIFT",
    }:
        return 2

    return -1


def publish_drift_metrics(report):
    """Publish a completed drift report to Prometheus."""

    if not isinstance(report, dict):
        raise ValueError("Drift detector did not return a report dictionary.")

    data_status = report.get(
        "data_drift_status", "INSUFFICIENT_DATA"
    )
    prediction_status = report.get(
        "prediction_drift_status", "INSUFFICIENT_DATA"
    )
    overall_status = report.get(
        "overall_status", "INSUFFICIENT_DATA"
    )

    DATA_DRIFT_STATUS.set(status_to_number(data_status))
    PREDICTION_DRIFT_STATUS.set(
        status_to_number(prediction_status)
    )
    OVERALL_DRIFT_STATUS.set(status_to_number(overall_status))

    # Publish the PSI value for each feature that was analyzed.
    feature_psi = report.get("feature_psi", {})

    for feature, psi_value in feature_psi.items():
        value = float(psi_value)

        if not pd.notna(value):
            continue

        FEATURE_PSI.labels(feature=str(feature)).set(value)

    # Record the time of this completed analysis.
    analysis_timestamp = report.get("analysis_timestamp", time())
    DRIFT_ANALYSIS_TIMESTAMP.set(float(analysis_timestamp))

    DRIFT_ANALYSIS_SUCCESS.set(1)

    logger.info(
        "Drift analysis completed | data=%s | prediction=%s | overall=%s",
        data_status,
        prediction_status,
        overall_status,
    )


# --------------------------------------------------
# 5. Background drift monitoring
# --------------------------------------------------

async def drift_monitor_loop():
    """
    Run the existing detector periodically in a worker thread.

    Running it outside the FastAPI event loop prevents the
    analysis from blocking normal API requests.
    """

    while True:
        try:
            report = await asyncio.to_thread(
                analyze_drift,
                window_size=DRIFT_WINDOW_SIZE,
                min_samples=DRIFT_MIN_SAMPLES,
            )

            publish_drift_metrics(report)

        except asyncio.CancelledError:
            raise

        except Exception:
            DRIFT_ANALYSIS_ERRORS_TOTAL.inc()
            DRIFT_ANALYSIS_SUCCESS.set(0)

            logger.exception(
                "Drift analysis failed. "
                "The API will continue serving predictions."
            )

        await asyncio.sleep(DRIFT_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop the monitoring task with the API."""

    monitor_task = asyncio.create_task(drift_monitor_loop())
    logger.info(
        "Automatic drift monitoring started; interval=%s seconds",
        DRIFT_INTERVAL_SECONDS,
    )

    try:
        yield
    finally:
        monitor_task.cancel()

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        logger.info("Automatic drift monitoring stopped.")


# --------------------------------------------------
# 6. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="SentinelML API",
    description=(
        "Transaction fraud prediction and model observability "
        "with automated drift monitoring"
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# 7. Request and response schemas
# --------------------------------------------------

class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    distance_from_home: float = Field(ge=0, le=10_000)
    international: int = Field(ge=0, le=1)
    merchant_risk: float = Field(ge=0, le=1)
    transactions_last_24h: int = Field(ge=0, le=1000)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    fraud_probability: float
    inference_latency_ms: float


# --------------------------------------------------
# 8. API routes
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "SentinelML",
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expose inference and drift metrics to Prometheus."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/monitor/status")
def monitor_status():
    """Return the latest drift status and monitoring health."""

    return {
        "data_drift_status": DATA_DRIFT_STATUS._value.get(),
        "prediction_drift_status": (
            PREDICTION_DRIFT_STATUS._value.get()
        ),
        "overall_drift_status": OVERALL_DRIFT_STATUS._value.get(),
        "last_analysis_timestamp": (
            DRIFT_ANALYSIS_TIMESTAMP._value.get()
        ),
        "analysis_success": DRIFT_ANALYSIS_SUCCESS._value.get(),
        "analysis_errors": DRIFT_ANALYSIS_ERRORS_TOTAL._value.get(),
        "analysis_interval_seconds": DRIFT_INTERVAL_SECONDS,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    start = perf_counter()

    try:
        input_data = pd.DataFrame(
            [transaction.model_dump()],
            columns=FEATURES,
        )

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        fraud_class_index = list(model.classes_).index(1)
        fraud_probability = float(
            probabilities[fraud_class_index]
        )

        latency_seconds = perf_counter() - start
        latency_ms = round(latency_seconds * 1000, 3)

        label = "fraud" if prediction == 1 else "legitimate"

    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        raise

    # Record successful inference.
    PREDICTIONS_TOTAL.labels(label=label).inc()
    INFERENCE_LATENCY.observe(latency_seconds)

    # Logging is best-effort; failures must not break inference.
    try:
        log_inference(
            features=transaction.model_dump(),
            prediction=prediction,
            label=label,
            fraud_probability=fraud_probability,
            inference_latency_ms=latency_ms,
        )
    except Exception:
        logger.exception(
            "Inference logging failed; prediction remains successful."
        )

    return PredictionResponse(
        prediction=prediction,
        label=label,
        fraud_probability=fraud_probability,
        inference_latency_ms=latency_ms,
    )