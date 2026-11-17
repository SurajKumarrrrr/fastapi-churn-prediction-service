"""Smoke + latency test against a running service.

Usage: python scripts/smoke_test.py --url http://localhost:8000 --requests 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import httpx

from churn_service.data import generate_customers
from churn_service.features import FEATURES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()

    # Fresh customers from a different seed than training.
    df = generate_customers(args.requests, seed=2024)[FEATURES]
    payloads = [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]

    with httpx.Client(base_url=args.url, timeout=10) as client:
        health = client.get("/health").json()
        print("health:", health)
        if not health["model_loaded"]:
            return 1

        latencies, flagged = [], 0
        for p in payloads:
            start = time.perf_counter()
            r = client.post("/predict", json=p)
            latencies.append((time.perf_counter() - start) * 1000)
            r.raise_for_status()
            flagged += r.json()["will_churn"]

        start = time.perf_counter()
        r = client.post("/predict/batch", json={"customers": payloads})
        batch_ms = (time.perf_counter() - start) * 1000
        r.raise_for_status()

    q = statistics.quantiles(latencies, n=100)
    print(f"single requests: n={len(latencies)} p50={q[49]:.1f} ms p95={q[94]:.1f} ms")
    print(f"batch of {len(payloads)}: {batch_ms:.1f} ms total")
    print(f"flagged as churn risk: {flagged}/{len(payloads)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
