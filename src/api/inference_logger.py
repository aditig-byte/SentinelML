
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "inference.jsonl"

# Bounded queue prevents unlimited memory growth if logging falls behind.
_event_queue = queue.Queue(maxsize=10_000)
_dropped_events = 0
_drop_lock = threading.Lock()


def _write_events():
    """Write queued inference events in a background thread."""
    global _dropped_events

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        while True:
            event = _event_queue.get()

            try:
                with LOG_FILE.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event) + "\n")
            except Exception:
                # A logging failure must not affect model inference.
                with _drop_lock:
                    _dropped_events += 1
            finally:
                _event_queue.task_done()

    except Exception:
        # If the logging worker cannot initialize, inference can still run.
        with _drop_lock:
            _dropped_events += 1


_worker = threading.Thread(
    target=_write_events,
    name="sentinelml-log-writer",
    daemon=True,
)
_worker.start()


def log_inference(
    features: dict,
    prediction: int,
    label: str,
    fraud_probability: float,
    inference_latency_ms: float,
) -> bool:
    """Queue an inference event without waiting for disk I/O."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": prediction,
        "label": label,
        "fraud_probability": fraud_probability,
        "inference_latency_ms": inference_latency_ms,
    }

    try:
        _event_queue.put_nowait(event)
        return True
    except queue.Full:
        global _dropped_events
        with _drop_lock:
            _dropped_events += 1
        return False


def get_logging_status() -> dict:
    """Return basic health information about the local logging queue."""
    with _drop_lock:
        dropped = _dropped_events

    return {
        "queue_size": _event_queue.qsize(),
        "dropped_events": dropped,
        "log_file": str(LOG_FILE),
    }
