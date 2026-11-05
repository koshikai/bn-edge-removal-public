#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

HTTP_ERROR_THRESHOLD = 400

sync_playwright_factory: Any | None
try:
    from playwright.sync_api import sync_playwright as _sync_playwright_factory
except ImportError:
    sync_playwright_factory = None
else:
    sync_playwright_factory = _sync_playwright_factory


def _load_playwright() -> Any:
    if sync_playwright_factory is None:
        raise RuntimeError(
            "playwright is not installed. Install it with: "
            "uv add --dev playwright && uv run playwright install chromium"
        )
    return sync_playwright_factory


def run(url: str, timeout_ms: int, settle_ms: int, screenshot: str | None) -> int:
    sync_playwright = _load_playwright()

    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Collect JS/runtime errors surfaced in the browser.
        page.on(
            "console",
            lambda msg: (
                console_errors.append(msg.text) if msg.type == "error" else None
            ),
        )
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        status = response.status if response is not None else 0

        try:
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            # Some apps keep long-lived connections; proceed after a short settle.
            pass

        if settle_ms > 0:
            page.wait_for_timeout(settle_ms)

        if screenshot:
            page.screenshot(path=screenshot, full_page=True)

        browser.close()

    if status >= HTTP_ERROR_THRESHOLD or status == 0:
        print(f"ERROR: HTTP status {status} for {url}")
        return 1
    if page_errors:
        print("ERROR: page errors detected")
        for err in page_errors:
            print(f"- {err}")
        return 1
    if console_errors:
        print("ERROR: console errors detected")
        for err in console_errors:
            print(f"- {err}")
        return 1

    print("OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a marimo web UI.")
    parser.add_argument(
        "--url", required=True, help="Base URL, e.g. http://127.0.0.1:27182"
    )
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--settle-ms", type=int, default=2_000)
    parser.add_argument("--screenshot", default=None)
    args = parser.parse_args()
    return run(args.url, args.timeout_ms, args.settle_ms, args.screenshot)


if __name__ == "__main__":
    raise SystemExit(main())
