
import argparse
import time

import numpy as np
import requests


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"
TOTAL_REQUESTS = 100
RANDOM_STATE = 2026

FEATURES = [
    "amount",
    "hour",
    "distance_from_home",
    "international",
    "merchant_risk",
    "transactions_last_24h",
]


# --------------------------------------------------
# 2. Transaction generation
# --------------------------------------------------

def generate_transaction(rng, mode: str) -> dict:
    """Generate baseline-matched or deliberately shifted traffic."""

    if mode == "baseline":
        amount = np.clip(
            rng.lognormal(mean=3.7, sigma=1.0), 1, 5000
        )
        distance = np.clip(
            rng.exponential(scale=25), 0, 300
        )
        merchant_risk = rng.beta(a=2, b=5)
        recent_transactions = np.clip(
            rng.poisson(lam=4), 0, 30
        )
        hour = rng.integers(0, 24)
        international = rng.binomial(1, 0.08)

    else:
        # Intentional feature drift:
        # higher transaction amounts and greater distances.
        amount = np.clip(
            rng.lognormal(mean=5.0, sigma=1.0), 1, 5000
        )
        distance = np.clip(
            rng.exponential(scale=65), 0, 300
        )
        merchant_risk = rng.beta(a=2, b=5)
        recent_transactions = np.clip(
            rng.poisson(lam=4), 0, 30
        )
        hour = rng.integers(0, 24)
        international = rng.binomial(1, 0.08)

    return {
        "amount": round(float(amount), 2),
        "hour": int(hour),
        "distance_from_home": round(float(distance), 2),
        "international": int(international),
        "merchant_risk": round(float(merchant_risk), 4),
        "transactions_last_24h": int(recent_transactions),
    }


# --------------------------------------------------
# 3. Send traffic to the inference API
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SentinelML monitoring traffic."
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "drift"],
        default="baseline",
        help="Use training-matched traffic or deliberately shifted traffic.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=TOTAL_REQUESTS,
        help="Number of requests to send.",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    rng = np.random.default_rng(RANDOM_STATE)

    print("\n=== SentinelML: Traffic Generator ===\n")
    print(f"API endpoint: {API_URL}")
    print(f"Requests: {args.count}")
    print(f"Mode: {args.mode}\n")

    try:
        health = requests.get(HEALTH_URL, timeout=5)
        health.raise_for_status()
        print("API health check: SUCCESS\n")
    except requests.RequestException as error:
        print(f"API health check failed: {error}")
        print("Start the FastAPI server and retry.")
        return

    successes = 0
    failures = 0

    for i in range(1, args.count + 1):
        transaction = generate_transaction(rng, args.mode)

        try:
            response = requests.post(
                API_URL,
                json=transaction,
                timeout=10,
            )
            response.raise_for_status()
            successes += 1
        except requests.RequestException as error:
            failures += 1
            print(f"Request {i} failed: {error}")

        if i % 25 == 0 or i == args.count:
            print(
                f"Sent {i}/{args.count} | "
                f"Successes: {successes} | Failures: {failures}"
            )

        time.sleep(0.02)

    print("\n=== Traffic Generation Finished ===")
    print(f"Successful requests: {successes}")
    print(f"Failed requests: {failures}")


if __name__ == "__main__":
    main()
