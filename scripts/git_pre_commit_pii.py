#!/usr/bin/env python3
"""Git pre-commit hook: scan staged additions for PII before they are recorded.

The hook fails closed. Anything it cannot scan completely -- an oversized diff, a
binary blob, text that is not valid UTF-8, a service that is down or out of
memory -- blocks the commit rather than passing with a warning, because a warning
on a file nobody re-reads is indistinguishable from no scan at all. Binary files
that genuinely cannot carry scannable text are opted out explicitly through a
documented allowlist, never implicitly.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import service_control  # noqa: E402


SERVICE_URL = os.environ.get("PRIVACY_FILTER_URL", "http://127.0.0.1:8757").rstrip("/")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
EMAIL_RE = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+-])")

# Starting and stopping belong to service_control; it owns the lock that keeps
# two concurrent commits from replacing each other's container.
CONTROL_SCRIPT = service_control.CONTROL_SCRIPT
# Auto-start the service when it is not running. Off only if explicitly disabled.
AUTOSTART = os.environ.get("PRIVACY_FILTER_AUTOSTART", "1").lower() not in ("0", "false", "no")
# Stop the service after the commit, but only when this hook auto-started it --
# a service the user started by hand is left running. Frees VRAM back to the GPU
# (e.g. a game) once the one-off scan is done.
AUTOSTOP = os.environ.get("PRIVACY_FILTER_AUTOSTOP", "1").lower() not in ("0", "false", "no")
# How long to wait for the model to finish loading after a start, in seconds.
STARTUP_TIMEOUT = float(os.environ.get("PRIVACY_FILTER_STARTUP_TIMEOUT", "180"))
# A large staged file is scanned as hundreds of sequential windows inside the
# service: a 1 MiB diff measured at roughly six minutes on an RTX 3080 sharing
# the GPU with a desktop session, so the ceiling is in minutes, not seconds.
REQUEST_TIMEOUT = float(os.environ.get("PRIVACY_FILTER_TIMEOUT", "900"))
# /health answers immediately once the model is loaded, so it gets a short
# timeout of its own; the startup poll needs to iterate, not block on one call.
HEALTH_TIMEOUT = float(os.environ.get("PRIVACY_FILTER_HEALTH_TIMEOUT", "15"))

# Binary files cannot be scanned yet (no text extraction). They block the commit
# unless deliberately allowlisted, by fnmatch pattern, either in this env var
# (colon- or comma-separated) or one pattern per line in the repo-root file
# below. Both are opt-in and reviewable; neither is a silent skip.
BINARY_ALLOWLIST_ENV = "PRIVACY_FILTER_BINARY_ALLOWLIST"
BINARY_ALLOWLIST_FILE = ".privacy-filter-binary-allowlist"
REVIEWED_FINDINGS_FILE = ".privacy-filter-reviewed-findings"
# This scanner's own configuration, excluded from scanning; see scan_staged.
CONTROL_FILES = frozenset({BINARY_ALLOWLIST_FILE, REVIEWED_FINDINGS_FILE})


class ServiceUnreachable(RuntimeError):
    """The service did not answer at all (vs. answering with an HTTP error)."""


class ScanTimeout(RuntimeError):
    """The service is running but did not answer in time.

    Distinct from ServiceUnreachable: the startup poll retries this, and a scan
    that hits it blocks the commit without telling anyone to restart a service
    that is very likely still busy scanning.
    """


class UndecodableDiff(RuntimeError):
    """Git produced bytes for this path that are not valid UTF-8."""


@dataclass(frozen=True)
class OffsetLocation:
    path: str
    line: int
    column: int


@dataclass(frozen=True)
class ScanPayload:
    path: str
    text: str
    locations: list[OffsetLocation]


@dataclass(frozen=True)
class ReviewedFinding:
    """One finding a human looked at and declared harmless.

    Keyed on a digest of the matched text, not on its line number. Line numbers
    move and get reused: an entry pinned to "line 10, label secret" would go on
    silently suppressing whatever landed on line 10 later, including a real
    secret. The digest only matches the exact text that was actually reviewed.
    """

    path: str
    label: str
    digest: str


def run_git(args: list[str]) -> str:
    """Run git and decode its output strictly.

    Strict decoding matters: replacing undecodable bytes would hand the scanner a
    silently altered diff, and a mangled path would make the follow-up git call
    return an empty diff that looks like "nothing to scan".
    """
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UndecodableDiff(f"git {' '.join(args)} produced non-UTF-8 output") from exc


def request_json(
    path: str, payload: dict[str, Any] | None = None, timeout: float | None = None
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{SERVICE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    if timeout is None:
        timeout = HEALTH_TIMEOUT if payload is None else REQUEST_TIMEOUT
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # The service answered, but errored. HTTPError is a subclass of URLError,
        # so it must be caught first or a 500 gets misreported as "not reachable"
        # and sends people restarting an already-running service.
        body = exc.read().decode("utf-8", errors="replace").strip()
        detail = service_error_detail(body)
        raise RuntimeError(
            f"Privacy Filter service returned HTTP {exc.code} for {path}{detail}"
        ) from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        if is_timeout(exc):
            # Not "unreachable": the service is alive and probably still
            # scanning. Telling anyone to restart it here would be wrong advice.
            raise ScanTimeout(
                f"Privacy Filter did not answer {path} within {timeout:g}s, so the staged "
                "changes were not scanned. Raise PRIVACY_FILTER_TIMEOUT, or stage less at once."
            ) from exc
        raise ServiceUnreachable(
            f"Privacy Filter service is not reachable. Start it with {CONTROL_SCRIPT} ensure"
        ) from exc


def is_timeout(exc: BaseException) -> bool:
    """True for read/connect timeouts, which urllib reports in two shapes."""
    if isinstance(exc, TimeoutError):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, TimeoutError)


def service_error_detail(body: str) -> str:
    """Extract a human-readable message from a JSON or plain error body."""
    if not body:
        return ""
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return f": {body}"
    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("error")
            if message:
                return f": {message}"
        message = detail or parsed.get("message") or parsed.get("error")
        if isinstance(message, str) and message:
            return f": {message}"
    return f": {body}"


def check_service() -> bool:
    """Ensure the service is ready. Returns True if this commit leased it.

    The start-and-wait logic lives in service_control, which holds a lock while
    it works, so two commits landing at once cannot replace each other's
    container. The lease matters just as much: without it, whichever commit
    finishes first would stop the service while the other was still scanning.
    """
    if not AUTOSTART:
        state = service_control.state()
        if state == service_control.STATE_DOWN:
            raise ServiceUnreachable(
                f"Privacy Filter service is not reachable. Start it with {CONTROL_SCRIPT} ensure"
            )
        verify_state(state)
        return False

    if service_control.state() == service_control.STATE_DOWN:
        print("Privacy Filter service not running; starting it ...", file=sys.stderr)
    try:
        service_control.ensure(STARTUP_TIMEOUT, lease=True)
    except service_control.LifecycleError as exc:
        raise RuntimeError(f"Privacy Filter service is not usable: {exc}") from exc
    return True


def verify_state(state: str) -> None:
    if state != service_control.STATE_HEALTHY:
        raise RuntimeError(
            "Privacy Filter service is not GPU-backed; refusing to commit unscanned changes"
        )


def stop_service() -> None:
    """Drop this commit's lease, freeing VRAM if nothing else is using the service.

    Best effort -- a teardown problem must never fail a commit whose scan already
    passed. Releasing rather than stopping is what makes concurrent commits safe:
    the service only goes down when the last one is finished with it.
    """
    try:
        outcome = service_control.release()
    except (service_control.LifecycleError, OSError) as exc:
        print(f"WARNING: could not release Privacy Filter service: {exc}", file=sys.stderr)
        return
    if outcome == "stopped":
        print("Stopped auto-started Privacy Filter service.", file=sys.stderr)
    elif outcome.startswith("held"):
        print(f"Leaving Privacy Filter service up: {outcome}.", file=sys.stderr)


def decode_path(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UndecodableDiff("a staged path is not valid UTF-8") from exc


def staged_paths() -> list[str]:
    raw = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [decode_path(item) for item in raw.split(b"\0") if item]


def binary_paths() -> set[str]:
    raw = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--cached", "--numstat", "-z", "--diff-filter=ACMR"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.split(b"\0")
    binaries: set[str] = set()
    index = 0
    while index < len(raw):
        record = raw[index]
        index += 1
        if not record:
            continue
        fields = decode_path(record).split("\t")
        if len(fields) >= 3 and fields[0] == "-" and fields[1] == "-":
            binaries.add(fields[2])
    return binaries


def repo_root() -> str:
    try:
        return run_git(["rev-parse", "--show-toplevel"]).strip()
    except (subprocess.CalledProcessError, UndecodableDiff, OSError):
        return ""


def binary_allowlist() -> list[str]:
    """fnmatch patterns for binary paths that are deliberately not scanned."""
    raw = os.environ.get(BINARY_ALLOWLIST_ENV, "")
    patterns = [item.strip() for item in raw.replace(",", ":").split(":") if item.strip()]

    root = repo_root()
    if root:
        allowlist_file = os.path.join(root, BINARY_ALLOWLIST_FILE)
        try:
            with open(allowlist_file, encoding="utf-8") as handle:
                for line in handle:
                    entry = line.strip()
                    if entry and not entry.startswith("#"):
                        patterns.append(entry)
        except (FileNotFoundError, IsADirectoryError):
            pass
        except (OSError, UnicodeDecodeError) as exc:
            # An unreadable allowlist must not quietly become an empty one.
            raise RuntimeError(f"could not read {allowlist_file}: {exc}") from exc
    return patterns


def is_allowlisted(path: str, patterns: list[str]) -> bool:
    basename = os.path.basename(path)
    return any(
        fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(basename, pattern)
        for pattern in patterns
    )


DIGEST_LENGTH = 16


def finding_digest(text: str, start: int, end: int) -> str:
    """Short digest of the exact text a finding matched.

    Computed locally and never sent anywhere. What gets written to the allowlist
    is this digest, not the text -- and by construction you only allowlist text
    you have already confirmed is not sensitive.
    """
    return hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()[:DIGEST_LENGTH]


def is_digest(value: str) -> bool:
    return len(value) == DIGEST_LENGTH and all(char in "0123456789abcdef" for char in value)


def allowlist_entry(finding: ReviewedFinding) -> str:
    return f"{finding.path}\t{finding.label}\t{finding.digest}"


def reviewed_findings() -> set[ReviewedFinding]:
    root = repo_root()
    if not root:
        return set()

    path = os.path.join(root, REVIEWED_FINDINGS_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            entries = set()
            for line_number, raw in enumerate(handle, start=1):
                entry = raw.strip()
                if not entry or entry.startswith("#"):
                    continue
                parts = entry.split("\t")
                if len(parts) == 3 and parts[1].isdigit():
                    # The old path<TAB>line<TAB>label format. Honouring it would
                    # keep the line-pinned suppression alive, so fail closed and
                    # make the reader regenerate from a real scan.
                    raise RuntimeError(
                        f"{REVIEWED_FINDINGS_FILE} line {line_number} uses the old "
                        "path/line/label format, which could mask new findings on a "
                        "reused line. Delete these entries and re-add the "
                        "'to allowlist:' lines the hook prints when it blocks."
                    )
                if len(parts) != 3 or not is_digest(parts[2]):
                    raise RuntimeError(
                        f"invalid {REVIEWED_FINDINGS_FILE} entry at line {line_number}: "
                        "expected path<TAB>label<TAB>digest"
                    )
                entries.add(ReviewedFinding(parts[0], parts[1], parts[2]))
            return entries
    except (FileNotFoundError, IsADirectoryError):
        return set()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc


def build_payload(path: str) -> ScanPayload:
    diff = run_git(["diff", "--no-ext-diff", "--cached", "--unified=0", "--", path])
    text_parts: list[str] = []
    locations: list[OffsetLocation] = []
    line_number: int | None = None

    for diff_line in diff.splitlines():
        hunk = HUNK_RE.match(diff_line)
        if hunk:
            line_number = int(hunk.group(1))
            continue
        if line_number is None:
            continue
        if diff_line.startswith("+") and not diff_line.startswith("+++"):
            content = diff_line[1:]
            for column, char in enumerate(content, start=1):
                text_parts.append(char)
                locations.append(OffsetLocation(path=path, line=line_number, column=column))
            text_parts.append("\n")
            locations.append(OffsetLocation(path=path, line=line_number, column=len(content) + 1))
            line_number += 1
        elif diff_line.startswith("-") and not diff_line.startswith("---"):
            continue
        elif diff_line.startswith(" "):
            line_number += 1

    return ScanPayload(path=path, text="".join(text_parts), locations=locations)


def location_for(payload: ScanPayload, offset: int) -> OffsetLocation:
    if not payload.locations:
        return OffsetLocation(path=payload.path, line=0, column=0)
    if offset < 0:
        return payload.locations[0]
    if offset >= len(payload.locations):
        return payload.locations[-1]
    return payload.locations[offset]


def scan_payload(payload: ScanPayload) -> list[dict[str, Any]]:
    """Submit the whole added-text payload; the service windows it internally.

    No size limit here on purpose. The service is the single authority on token
    budgets, overlap and adaptive retries, so a client-side cap would only
    reintroduce the unscanned-file hole this hook exists to close.
    """
    response = request_json("/check", {"text": payload.text, "include_text": False})
    findings = list(response.get("findings", []))
    findings.extend(regex_findings(payload.text, findings))
    findings.sort(key=lambda item: (int(item.get("start", 0)), int(item.get("end", 0))))
    return findings


def regex_findings(text: str, existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    additions: list[dict[str, Any]] = []
    for match in EMAIL_RE.finditer(text):
        start = match.start()
        end = match.end()
        if overlaps_existing(start, end, existing):
            continue
        additions.append({"label": "private_email", "start": start, "end": end, "score": 1.0})
    return additions


def overlaps_existing(start: int, end: int, existing: list[dict[str, Any]]) -> bool:
    for finding in existing:
        finding_start = int(finding.get("start", -1))
        finding_end = int(finding.get("end", -1))
        if start < finding_end and end > finding_start:
            return True
    return False


def main() -> int:
    try:
        leased = check_service()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        return scan_staged()
    except RuntimeError as exc:
        # e.g. an HTTP 503 (GPU out of memory) raised mid-scan. Fail closed with
        # the real message rather than letting a traceback escape.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        # Always release a lease we took, including on failure -- otherwise a
        # blocked commit would pin the GPU until the machine rebooted.
        if leased and AUTOSTOP:
            stop_service()


def scan_staged() -> int:
    binaries = binary_paths()
    allowlist = binary_allowlist()
    reviewed = reviewed_findings()
    findings_by_file: list[tuple[ScanPayload, list[dict[str, Any]]]] = []
    unscannable: list[str] = []
    service_failure = ""
    binary_blocked = False

    for path in staged_paths():
        if path in CONTROL_FILES:
            # Scanning our own control files is self-defeating: their content is
            # fnmatch patterns and hex digests, and the model reads a digest as a
            # secret or an account number. Allowlisting those findings would add
            # more digests, which flag again. It converges, but only by teaching
            # people to paste allowlist entries without reading them -- a far
            # worse outcome than not scanning two files whose format is fixed and
            # reviewable. Announced, never silent.
            print(
                f"NOTE: Privacy Filter does not scan its own control file {path}",
                file=sys.stderr,
            )
            continue
        if path in binaries:
            if is_allowlisted(path, allowlist):
                print(f"NOTE: Privacy Filter skipped allowlisted binary {path}", file=sys.stderr)
                continue
            unscannable.append(f"{path}: binary file, and no text extraction exists yet")
            binary_blocked = True
            continue
        try:
            payload = build_payload(path)
        except UndecodableDiff as exc:
            unscannable.append(f"{path}: {exc}")
            continue
        if not payload.text.strip():
            continue
        try:
            findings = scan_payload(payload)
        except RuntimeError as exc:
            # A service-level failure will repeat for every remaining file, so
            # stop here and report it once instead of retrying it per file.
            service_failure = str(exc)
            unscannable.append(f"{path}: could not be scanned completely")
            break
        blocking_findings = []
        for finding in findings:
            location = location_for(payload, int(finding.get("start", 0)))
            label = str(finding.get("label", "unknown"))
            digest = finding_digest(
                payload.text, int(finding.get("start", 0)), int(finding.get("end", 0))
            )
            # Carried on the finding so the report can print a paste-ready entry
            # without recomputing it; the digest is not raw matched text.
            finding["digest"] = digest
            key = ReviewedFinding(location.path, label, digest)
            if key in reviewed:
                print(
                    f"NOTE: Privacy Filter reviewed false positive {location.path}:{location.line}:{label}",
                    file=sys.stderr,
                )
                continue
            blocking_findings.append(finding)
        if blocking_findings:
            findings_by_file.append((payload, blocking_findings))

    return report(findings_by_file, unscannable, service_failure, binary_blocked)


def report(
    findings_by_file: list[tuple[ScanPayload, list[dict[str, Any]]]],
    unscannable: list[str],
    service_failure: str,
    binary_blocked: bool,
) -> int:
    blocked = 0

    if unscannable:
        print("ERROR: Privacy Filter could not scan every staged change.", file=sys.stderr)
        for entry in unscannable:
            print(f"  {entry}", file=sys.stderr)
        if service_failure:
            print(f"  service error: {service_failure}", file=sys.stderr)
            print(
                "  Free VRAM (e.g. close a game) or restart the service, then commit again.",
                file=sys.stderr,
            )
        if binary_blocked:
            print(
                f"  Unstage the binary, or allowlist it in {BINARY_ALLOWLIST_FILE} "
                f"(or {BINARY_ALLOWLIST_ENV}) once you have confirmed it carries no PII.",
                file=sys.stderr,
            )
        if not service_failure and not binary_blocked:
            print(
                "  Re-encode the file as UTF-8 text or unstage it; unscannable "
                "content is never allowed through.",
                file=sys.stderr,
            )
        blocked = 1

    if findings_by_file:
        print("ERROR: Privacy Filter detected possible PII in staged changes.", file=sys.stderr)
        print("Raw matched text is intentionally omitted.", file=sys.stderr)
        entries: list[str] = []
        for payload, findings in findings_by_file:
            for finding in findings:
                location = location_for(payload, int(finding.get("start", 0)))
                label = str(finding.get("label", "unknown"))
                score = finding.get("score")
                score_text = f" score={score:.4f}" if isinstance(score, float) else ""
                print(
                    f"  {location.path}:{location.line}:{location.column}: {label}{score_text}",
                    file=sys.stderr,
                )
                digest = finding.get("digest")
                if digest:
                    entries.append(allowlist_entry(ReviewedFinding(location.path, label, digest)))
        if entries:
            # The digest cannot be produced by hand, so the only workable way to
            # allowlist a reviewed false positive is to hand the line over here.
            print(
                f"\nIf you have checked one of these and it is not PII, add its line to "
                f"{REVIEWED_FINDINGS_FILE} (tab-separated):",
                file=sys.stderr,
            )
            for entry in entries:
                print(f"  {entry}", file=sys.stderr)
        blocked = 1

    return blocked


if __name__ == "__main__":
    raise SystemExit(main())
