#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify marimo server is responding by checking its HTTP response."
    )
    parser.add_argument("--url", required=True, help="URL of the marimo server.")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Timeout in milliseconds.",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=2000,
        help="Settle time in milliseconds before sending requests.",
    )
    parser.add_argument(
        "--screenshot",
        default=None,
        help="Ignored (for API compatibility).",
    )
    args = parser.parse_args()

    # Settle time
    time.sleep(args.settle_ms / 1000.0)

    timeout_s = args.timeout_ms / 1000.0
    start_time = time.monotonic()

    while True:
        try:
            req = urllib.request.Request(
                args.url, headers={"User-Agent": "marimo-ui-smoketest"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode("utf-8")
                # Confirm HTTP 200 or presence of marimo keywords
                if response.status == 200 or "marimo" in content.lower():  # noqa: PLR2004
                    print(f"SUCCESS: Connected to marimo server at {args.url}")
                    return 0
        except urllib.error.URLError as e:
            if time.monotonic() - start_time > timeout_s:
                print(
                    f"ERROR: Failed to connect to marimo server at {args.url} "
                    f"within timeout: {e}"
                )
                return 1
            time.sleep(0.5)
        except Exception as e:
            print(f"ERROR: Unexpected exception during connection test: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
