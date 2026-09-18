#!/usr/bin/env python3
"""Lifecycle control for the local Privacy Filter container.

One entry point for "is it up, bring it up, take it down", so callers -- the
pre-commit hook, the CLI, an agent following the skill -- stop reimplementing a
start-then-poll loop each time and stop racing each other.

The race is the reason this exists. ``start-service`` ends in
``podman run --replace``, which tears down whatever container currently holds the
name. Short scans survive that -- podman sends SIGTERM and uvicorn drains the
in-flight request first -- but anything still working after podman's stop grace
period (10s by default) is SIGKILLed, and the caller sees a bare connection
failure. Measured: a 13.7s scan interrupted at t=1s dies with "not reachable",
while the same scan interrupted at t=5s finishes because only 8.7s remained.
Large diffs, which are exactly what the pre-commit hook submits, sit well past
that line.

``ensure`` takes a lock, checks health first, and only starts the container when
it is genuinely not serving -- so a second caller reuses the running service in
about 50ms instead of killing the first caller's work.

Startup measured at roughly 6-10s on an RTX 3080 (cached image build, weights
already in the local HF cache), which is why there is no idle-reaper or keep-warm
mode here: holding ~2.8GB of VRAM to save six seconds is a bad trade on a machine
that also plays games. Start it, use it, stop it.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from privacy_filter_client import (  # noqa: E402
    ScanTimeout,
    ServiceError,
    ServiceUnreachable,
    request_json,
)

ROOT = Path(__file__).resolve().parents[1]
SERVICE_URL = os.environ.get("PRIVACY_FILTER_URL", "http://127.0.0.1:8757").rstrip("/")
START_SCRIPT = os.environ.get("PRIVACY_FILTER_START_SCRIPT", str(ROOT / "start-service"))
STOP_SCRIPT = os.environ.get("PRIVACY_FILTER_STOP_SCRIPT", str(ROOT / "stop-service"))
# What to tell a user whose service is down. Resolved from this file's location
# rather than hardcoded, so a clone anywhere prints its own working path.
CONTROL_SCRIPT = str(ROOT / "privacy-filter-service")
CONTAINER = os.environ.get("PRIVACY_FILTER_CONTAINER", "privacy-filter-local")

# Health answers instantly once the model is loaded; a long timeout here would
# only mask a service that is still starting.
HEALTH_TIMEOUT = float(os.environ.get("PRIVACY_FILTER_HEALTH_TIMEOUT", "5"))
# Model load measured at ~6-10s; 180s leaves room for a cold image build or a
# first-ever weight download without hanging forever.
READY_TIMEOUT = float(os.environ.get("PRIVACY_FILTER_STARTUP_TIMEOUT", "180"))
# How long to wait for another caller's ensure/stop to finish before giving up.
# Slightly longer than READY_TIMEOUT so a queued caller outlasts a full start.
LOCK_TIMEOUT = float(os.environ.get("PRIVACY_FILTER_LOCK_TIMEOUT", "200"))
POLL_INTERVAL = 0.5

# Exit codes. `status` reports the state; the action commands report why they
# could not reach a healthy service.
EXIT_HEALTHY = 0
EXIT_DOWN = 1
EXIT_NOT_READY = 2
EXIT_NO_GPU = 3
EXIT_START_FAILED = 4
EXIT_TIMED_OUT = 5
EXIT_BUSY = 6

STATE_HEALTHY = "healthy"
STATE_DOWN = "down"
STATE_NOT_READY = "not-ready"
STATE_NO_GPU = "no-gpu"

# Set by main() for --quiet; library callers (the hook) set it directly.
QUIET = False

STATE_EXIT = {
    STATE_HEALTHY: EXIT_HEALTHY,
    STATE_DOWN: EXIT_DOWN,
    STATE_NOT_READY: EXIT_NOT_READY,
    STATE_NO_GPU: EXIT_NO_GPU,
}


class LifecycleError(RuntimeError):
    """A lifecycle action failed; carries the exit code the caller should use."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def runtime_path(suffix: str) -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return Path(runtime_dir) / f"{CONTAINER}.{suffix}"


def lock_path() -> Path:
    return runtime_path("lifecycle.lock")


def lease_dir() -> Path:
    """One file per process that needs the service alive.

    Without this, two concurrent commits deadlock on politeness: the one that
    started the container stops it the moment its own scan finishes, killing the
    other's scan mid-flight. A lease says "I am still using this"; teardown waits
    for the last one out.
    """
    return runtime_path("leases")


def autostart_marker() -> Path:
    """Records that a leased caller started the container, so teardown may stop it.

    A service someone started by hand has no marker and is therefore never
    stopped from under them.
    """
    return runtime_path("autostarted")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def live_leases() -> list[int]:
    """Leases held by processes that still exist; prunes the rest.

    A commit killed with SIGKILL cannot drop its own lease, and a stale lease
    would pin the GPU forever, so dead holders are cleaned up on inspection.
    """
    directory = lease_dir()
    if not directory.is_dir():
        return []
    alive: list[int] = []
    for entry in directory.iterdir():
        try:
            pid = int(entry.name)
        except ValueError:
            entry.unlink(missing_ok=True)
            continue
        if pid_alive(pid):
            alive.append(pid)
        else:
            entry.unlink(missing_ok=True)
    return alive


def take_lease(pid: int) -> None:
    directory = lease_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / str(pid)).touch()


def drop_lease(pid: int) -> None:
    (lease_dir() / str(pid)).unlink(missing_ok=True)


def active_scans() -> int:
    """In-flight scans the service reports, or 0 when it cannot say."""
    body = health()
    if not body:
        return 0
    try:
        return int(body.get("active_scans", 0))
    except (TypeError, ValueError):
        return 0


class Lock:
    """Serialize lifecycle actions across processes.

    Two commits landing at once, or an agent and a commit overlapping, must not
    both decide to start (and therefore --replace) the container.
    """

    def __init__(self, timeout: float = LOCK_TIMEOUT) -> None:
        self.timeout = timeout
        self.handle = None

    def __enter__(self):
        path = lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(path, "w")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    self.handle.close()
                    self.handle = None
                    raise LifecycleError(
                        EXIT_BUSY,
                        f"another Privacy Filter lifecycle action held {path} for more "
                        f"than {self.timeout:g}s",
                    ) from exc
                time.sleep(POLL_INTERVAL)

    def __exit__(self, *_):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
        return False


def health() -> dict[str, Any] | None:
    """Return the /health body, or None when the service does not answer."""
    try:
        return request_json(SERVICE_URL, "/health", timeout=HEALTH_TIMEOUT)
    except (ServiceUnreachable, ScanTimeout):
        return None
    except ServiceError:
        # It answered, so the port is live, but it is not serving health.
        return {}


def state() -> str:
    body = health()
    if body is None:
        return STATE_DOWN
    if not body.get("ok"):
        return STATE_NOT_READY
    if not body.get("cuda_available"):
        return STATE_NO_GPU
    return STATE_HEALTHY


def container_running() -> bool:
    result = subprocess.run(
        ["podman", "ps", "--filter", f"name=^{CONTAINER}$", "--format", "{{.Names}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return bool(result.stdout.strip())


# Python's final traceback line -- "adaptive_scan.BudgetExceedsContextLimit: ..."
# or "RuntimeError: ..." -- carries the refusal the service wants read. Matching
# only that keeps this from being a blind log dump.
FATAL_LINE = re.compile(r"^(?:\w+\.)*\w*(?:Error|Exception):\s*(\S.*)$")


def container_failure() -> str | None:
    """Why a stopped container died, or None while it is still running.

    A service that refuses to start (a budget above the model's context limit,
    no CUDA device) exits immediately, and waiting out the readiness timeout
    would replace a precise, actionable message with "did not become ready".
    """
    if container_running():
        return None
    try:
        result = subprocess.run(
            ["podman", "logs", "--tail", "40", CONTAINER],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError:
        return None
    reason = None
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        match = FATAL_LINE.match(line)
        if match:
            reason = match.group(1)
    return reason


def run_script(script: str, action: str) -> None:
    # Progress chatter (cached build steps, container ids) goes to stderr so it
    # never pollutes a caller parsing stdout, and is dropped entirely in quiet
    # mode -- a hook that succeeded should say nothing. Failures still surface:
    # the non-zero exit becomes a LifecycleError with the action named.
    output = subprocess.DEVNULL if QUIET else sys.stderr
    try:
        subprocess.run([script], check=True, stdout=output, stderr=output)
    except FileNotFoundError as exc:
        raise LifecycleError(EXIT_START_FAILED, f"{action} script not found: {script}") from exc
    except subprocess.CalledProcessError as exc:
        hint = "" if QUIET else "; see output above"
        raise LifecycleError(
            EXIT_START_FAILED, f"{action} script failed (exit {exc.returncode}){hint}"
        ) from exc


def wait_until_ready(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = state()
        if current == STATE_HEALTHY:
            return
        if current == STATE_NO_GPU:
            raise LifecycleError(
                EXIT_NO_GPU, "Privacy Filter started but reports no CUDA device"
            )
        if current == STATE_DOWN:
            failure = container_failure()
            if failure is not None:
                raise LifecycleError(
                    EXIT_START_FAILED, f"Privacy Filter exited at startup: {failure}"
                )
        time.sleep(POLL_INTERVAL)
    raise LifecycleError(
        EXIT_TIMED_OUT, f"Privacy Filter did not become ready within {timeout:g}s"
    )


def ensure(timeout: float = READY_TIMEOUT, lease: bool = False) -> str:
    """Make the service healthy. Returns "reused" or "started".

    Pass ``lease=True`` when the caller will keep using the service and intends
    to ``release`` afterwards -- that is what stops a fast commit from tearing
    the service down under a slow concurrent one.
    """
    with Lock():
        current = state()
        outcome = None
        if current == STATE_NO_GPU:
            raise LifecycleError(
                EXIT_NO_GPU, "Privacy Filter is running without a CUDA device"
            )
        if current == STATE_HEALTHY:
            outcome = "reused"
        elif current == STATE_NOT_READY and container_running():
            # Container is up but not serving: it is either still loading the
            # model or wedged. Wait it out rather than --replace it, which would
            # kill a scan that is merely slow.
            wait_until_ready(timeout)
            outcome = "reused"
        else:
            run_script(START_SCRIPT, "start")
            wait_until_ready(timeout)
            outcome = "started"

        if lease:
            take_lease(os.getpid())
            if outcome == "started":
                autostart_marker().touch()
        return outcome


def release(pid: int | None = None) -> str:
    """Drop this caller's lease and stop the service if nobody else needs it.

    Returns what it did, so the caller can say so: "stopped", "held" (another
    lease is open, or a scan is still running), or "left-running" (nobody
    leased-started this service, so it is not ours to stop).
    """
    with Lock():
        drop_lease(os.getpid() if pid is None else pid)
        remaining = live_leases()
        if remaining:
            return f"held by {len(remaining)} other caller(s)"
        if not autostart_marker().exists():
            return "left-running"
        # Last lease out, but an unleased caller (an agent running a scan by
        # hand) may still be mid-request; do not pull the service from under it.
        if active_scans() > 0:
            return "held by a running scan"
        if container_running():
            run_script(STOP_SCRIPT, "stop")
        autostart_marker().unlink(missing_ok=True)
        return "stopped"


def stop(force: bool = False) -> str:
    """Stop the container, refusing by default while someone is still using it.

    Guarded because "stop when you are done" is the natural thing for an agent
    to run, and an unguarded stop would kill a concurrent commit's scan -- the
    same failure the leases exist to prevent. ``--force`` is the escape hatch
    for "free my VRAM now, I do not care".
    """
    with Lock():
        if not force:
            holders = live_leases()
            if holders:
                return f"refused: held by {len(holders)} caller(s); use --force to stop anyway"
            if active_scans() > 0:
                return "refused: a scan is still running; use --force to stop anyway"
        autostart_marker().unlink(missing_ok=True)
        for pid in live_leases():
            drop_lease(pid)
        if not container_running():
            return "already-down"
        run_script(STOP_SCRIPT, "stop")
        return "stopped"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privacy-filter-service",
        description=__doc__.split("\n", 1)[0],
        epilog=(
            "exit codes:\n"
            "  0  healthy (status) / action succeeded\n"
            "  1  service is down (status)\n"
            "  2  running but not ready (status)\n"
            "  3  running without a GPU\n"
            "  4  start or stop script failed\n"
            "  5  did not become ready in time\n"
            "  6  another lifecycle action holds the lock\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["status", "ensure", "release", "stop", "restart"],
        help=(
            "status: report state without changing it; "
            "ensure: start if needed and block until ready; "
            "release: drop this caller's lease, stopping the service if it was the last; "
            "stop: stop unconditionally and free VRAM; "
            "restart: stop then ensure"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With stop: stop even while another caller holds a lease or is scanning.",
    )
    parser.add_argument(
        "--lease",
        action="store_true",
        help=(
            "With ensure: register this process as a user of the service, so a "
            "concurrent caller's release cannot stop it while you are scanning."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=READY_TIMEOUT,
        help=f"Seconds to wait for readiness (default {READY_TIMEOUT:g}).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Report through the exit code only."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    global QUIET
    args = build_parser().parse_args(argv)
    QUIET = args.quiet

    def say(message: str) -> None:
        if not args.quiet:
            print(message)

    try:
        if args.command == "status":
            current = state()
            say(current)
            return STATE_EXIT[current]
        if args.command == "ensure":
            say(ensure(args.timeout, lease=args.lease))
            return EXIT_HEALTHY
        if args.command == "release":
            say(release())
            return EXIT_HEALTHY
        if args.command == "stop":
            outcome = stop(force=args.force)
            say(outcome)
            return EXIT_BUSY if outcome.startswith("refused") else EXIT_HEALTHY
        if args.command == "restart":
            stop(force=args.force)
            say(ensure(args.timeout))
            return EXIT_HEALTHY
    except LifecycleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.code
    return EXIT_HEALTHY


if __name__ == "__main__":
    raise SystemExit(main())
