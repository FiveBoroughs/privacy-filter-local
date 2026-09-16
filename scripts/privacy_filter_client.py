#!/usr/bin/env python3
"""CLI for the local Privacy Filter service.

Submit one file or stdin stream; the service does its own windowing, overlap and
adaptive retries, so there is nothing for the caller to chunk. Failures are
reported as distinct exit codes rather than one generic error, because "the
service is not running" and "the GPU could not fit even the smallest window" need
different responses -- and neither of them is "skip the scan".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_URL = "http://127.0.0.1:8757"
# Resolved from this file rather than hardcoded, so a clone anywhere prints
# its own working path in error messages.
CONTROL_SCRIPT = str(Path(__file__).resolve().parents[1] / "privacy-filter-service")
# A megabyte of text is scanned as hundreds of sequential forward passes, which
# measured at roughly six minutes on an RTX 3080 sharing the GPU with a desktop
# session. 900s leaves headroom above that without hanging indefinitely.
DEFAULT_TIMEOUT = float(os.environ.get("PRIVACY_FILTER_TIMEOUT", "900"))
# /health answers immediately once the model is loaded, so waiting minutes for it
# only hides the fact that the service is still starting.
HEALTH_TIMEOUT = float(os.environ.get("PRIVACY_FILTER_HEALTH_TIMEOUT", "30"))

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_INVALID_INPUT = 2
EXIT_UNREACHABLE = 3
EXIT_NO_GPU = 4
EXIT_BUDGET_EXHAUSTED = 5
EXIT_SERVICE_ERROR = 6
EXIT_TIMEOUT = 7

EXIT_CODE_HELP = """exit codes:
  0  no findings
  1  findings reported (check), or redaction written (redact)
  2  invalid or undecodable input
  3  service unreachable
  4  service is not GPU-backed
  5  GPU out of memory at the minimum window size; nothing was scanned
  6  other service error
  7  the scan did not finish before the timeout; nothing was scanned
"""


class InvalidInput(Exception):
    """The caller's input could not be read as UTF-8 text."""


class ServiceUnreachable(Exception):
    """The service did not answer at all (vs. answering with an HTTP error)."""


class ScanTimeout(Exception):
    """The service is running but did not answer in time.

    Kept separate from ServiceUnreachable on purpose: a slow scan and a stopped
    service need opposite responses, and "start the service" is useless advice
    when the service is busy scanning.
    """


class ServiceError(Exception):
    """The service answered with an HTTP error."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def read_input(path: str | None) -> str:
    if path:
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise InvalidInput(f"No such file: {path}") from exc
        except IsADirectoryError as exc:
            raise InvalidInput(f"Not a file: {path}") from exc
        except UnicodeDecodeError as exc:
            raise InvalidInput(
                f"{path} is not valid UTF-8 text, so it cannot be scanned. "
                "Extract its text first; the scan fails closed rather than "
                "silently skipping bytes it cannot read."
            ) from exc
        except OSError as exc:
            raise InvalidInput(f"Could not read {path}: {exc}") from exc
    if sys.stdin.isatty():
        raise InvalidInput("No input provided. Pipe text on stdin or pass --file PATH.")
    try:
        return sys.stdin.read()
    except UnicodeDecodeError as exc:
        raise InvalidInput(
            "stdin is not valid UTF-8 text, so it cannot be scanned. "
            "Extract its text first; the scan fails closed rather than "
            "silently skipping bytes it cannot read."
        ) from exc


def request_json(
    base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # HTTPError subclasses URLError; catch it first so a reachable-but-errored
        # service (e.g. a 503 CUDA OOM) is not misreported as "not reachable".
        body = exc.read().decode("utf-8", errors="replace").strip()
        code, message = parse_error_body(body)
        raise ServiceError(exc.code, code, message or f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        if is_timeout(exc):
            raise ScanTimeout(
                f"Privacy Filter did not finish within {timeout:g}s, so nothing was scanned. "
                "The service may still be working; raise --timeout (or PRIVACY_FILTER_TIMEOUT), "
                "or submit less at once."
            ) from exc
        raise ServiceUnreachable(
            f"Privacy Filter service is not reachable at {base_url}. Start it with '{CONTROL_SCRIPT} ensure'."
        ) from exc


def is_timeout(exc: BaseException) -> bool:
    """True for read/connect timeouts, which urllib reports in two shapes."""
    if isinstance(exc, TimeoutError):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, TimeoutError)


def parse_error_body(body: str) -> tuple[str, str]:
    """Pull ``(error_code, message)`` out of a JSON or plain-text error body."""
    if not body:
        return "", ""
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return "", body
    detail = parsed.get("detail") if isinstance(parsed, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("error", "")), str(detail.get("message", "")) or body
    if isinstance(detail, str):
        return "", detail
    return "", body


def exit_code_for(error: ServiceError) -> int:
    if error.code == "no_gpu":
        return EXIT_NO_GPU
    if error.code == "budget_exhausted":
        return EXIT_BUDGET_EXHAUSTED
    return EXIT_SERVICE_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privacy-filter",
        description=(
            "Call the local GPU-required Privacy Filter service. Large inputs are "
            "windowed by the service itself -- submit the whole file."
        ),
        epilog=EXIT_CODE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["health", "check", "redact"])
    parser.add_argument("--file", help="UTF-8 text file to inspect. Defaults to stdin.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--json", action="store_true", help="For redact, return full JSON.")
    parser.add_argument(
        "--include-text", action="store_true", help="Include raw matched PII text in JSON findings."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            f"Seconds to wait for the service "
            f"(default {DEFAULT_TIMEOUT:g} for a scan, {HEALTH_TIMEOUT:g} for health)."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "health":
        timeout = args.timeout if args.timeout is not None else HEALTH_TIMEOUT
        print(json.dumps(request_json(args.url, "/health", timeout=timeout), indent=2))
        return EXIT_CLEAN

    text = read_input(args.file)
    response = request_json(
        args.url,
        f"/{args.command}",
        {"text": text, "include_text": args.include_text},
        timeout=args.timeout if args.timeout is not None else DEFAULT_TIMEOUT,
    )

    if args.command == "check":
        print(json.dumps(response, indent=2))
        return EXIT_FINDINGS if response.get("count", 0) else EXIT_CLEAN

    if args.json:
        print(json.dumps(response, indent=2))
    else:
        redacted = response["redacted"]
        print(redacted, end="" if redacted.endswith("\n") else "\n")
    return EXIT_CLEAN


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except InvalidInput as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except ServiceUnreachable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_UNREACHABLE
    except ScanTimeout as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_TIMEOUT
    except ServiceError as exc:
        print(f"ERROR: Privacy Filter service returned HTTP {exc.status}: {exc}", file=sys.stderr)
        return exit_code_for(exc)


if __name__ == "__main__":
    raise SystemExit(main())
