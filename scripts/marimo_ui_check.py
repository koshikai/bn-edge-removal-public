#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _terminate_process(proc: subprocess.Popen, timeout_s: float) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start a marimo app, smoke-test the Web UI, then shut it down."
    )
    default_notebook = os.environ.get(
        "MARIMO_NOTEBOOK", "notebooks/reachability_dashboard.py"
    )
    parser.add_argument(
        "--notebook",
        default=default_notebook,
        help="Path to the marimo notebook (.py).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27182)
    parser.add_argument("--startup-timeout-s", type=int, default=60)
    parser.add_argument("--ui-timeout-ms", type=int, default=30_000)
    parser.add_argument("--settle-ms", type=int, default=2_000)
    parser.add_argument("--screenshot", default=None)
    parser.add_argument(
        "marimo_args",
        nargs=argparse.REMAINDER,
        help="Extra args for marimo run (use -- to separate).",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"ERROR: notebook not found: {notebook_path}")
        return 2

    if shutil.which("marimo") is None:
        print("ERROR: marimo not found. Run: uv sync --group dev")
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    smoketest_path = repo_root / "scripts" / "marimo_ui_smoketest.py"
    if not smoketest_path.exists():
        print(f"ERROR: missing smoketest script: {smoketest_path}")
        return 2

    url = f"http://{args.host}:{args.port}"

    cmd = [
        "marimo",
        "run",
        str(notebook_path),
        "--headless",
        "--port",
        str(args.port),
    ]
    if args.marimo_args:
        cmd.extend(args.marimo_args)

    log_file = tempfile.NamedTemporaryFile(
        prefix="marimo-ui-",
        suffix=".log",
        delete=False,
        mode="w",
        encoding="utf-8",
    )
    log_path = Path(log_file.name)

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=repo_root,
            env=os.environ.copy(),
        )
        log_file.flush()

        if not _wait_for_port(args.host, args.port, args.startup_timeout_s):
            if proc.poll() is not None:
                print("ERROR: marimo server exited early.")
            else:
                print("ERROR: marimo server did not start in time.")
            print(f"Server log: {log_path}")
            return 1

        smoketest_cmd = [
            sys.executable,
            str(smoketest_path),
            "--url",
            url,
            "--timeout-ms",
            str(args.ui_timeout_ms),
            "--settle-ms",
            str(args.settle_ms),
        ]
        if args.screenshot:
            smoketest_cmd.extend(["--screenshot", args.screenshot])

        result = subprocess.run(smoketest_cmd, check=False)
        if result.returncode != 0:
            print("ERROR: Web UI smoke-test failed.")
            print(f"Server log: {log_path}")
            return result.returncode

        print("できました")
        return 0
    finally:
        if proc is not None:
            _terminate_process(proc, timeout_s=10)
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
