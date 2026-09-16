"""Pure, dependency-free adaptive scanning: budgets, retries, dedup, redaction.

A single fixed token budget cannot be right for a GPU whose free VRAM changes
minute to minute (a game starts, a game quits). This module drives inference
through a shrinking ladder of token budgets: it starts at the configured
maximum, and when one window raises an out-of-memory error it discards that
window's results, re-plans *only that span* at a smaller budget and retries the
smaller windows. Windows that already succeeded are never rescanned. If even the
minimum budget cannot fit, the scan fails closed -- no partial findings.

The model, the tokenizer and the CUDA error types are all injected, so every bit
of this logic (including the OOM ladder) is unit-testable on a plain interpreter
with no torch, no transformers and no GPU. The service wires the real ones in.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

Span = tuple[int, int]
PlanWindows = Callable[[str, int, int], Sequence[Span]]
Classify = Callable[[str], Iterable[dict[str, Any]]]


@dataclass(frozen=True)
class Finding:
    label: str
    start: int
    end: int
    text: str
    score: float | None


class ScanBudgetExhausted(RuntimeError):
    """Inference ran out of memory even at the smallest configured window."""

    def __init__(self, budget: int) -> None:
        super().__init__(f"out of memory at the minimum token budget ({budget} tokens)")
        self.budget = budget


@dataclass(frozen=True)
class BudgetPolicy:
    """How large a window to try, and how far to back off when one does not fit."""

    max_tokens: int = 4096
    min_tokens: int = 256
    shrink_factor: float = 0.5
    overlap_tokens: int = 64

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {self.min_tokens}")
        if not 0.0 < self.shrink_factor < 1.0:
            raise ValueError(f"shrink_factor must be in (0, 1), got {self.shrink_factor}")
        if self.overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be >= 0, got {self.overlap_tokens}")
        if self.min_tokens > self.max_tokens:
            # A tokenizer whose context is smaller than the configured minimum
            # must still be usable; the maximum wins.
            object.__setattr__(self, "min_tokens", self.max_tokens)

    def ladder(self) -> tuple[int, ...]:
        """Descending token budgets to try, ending at ``min_tokens``."""
        steps = [self.max_tokens]
        while steps[-1] > self.min_tokens:
            smaller = max(self.min_tokens, int(steps[-1] * self.shrink_factor))
            if smaller >= steps[-1]:
                smaller = steps[-1] - 1
            steps.append(smaller)
        return tuple(steps)


def normalize_label(raw: str) -> str:
    label = raw.removeprefix("B-").removeprefix("I-").removeprefix("E-").removeprefix("S-")
    return label.lower()


def findings_from_raw(chunk: str, base: int, raw_items: Iterable[dict[str, Any]]) -> list[Finding]:
    """Turn one window's raw model output into ``Finding``s in original coordinates.

    ``base`` is the window's start offset in the full input, so every finding is
    reported against the text the caller submitted, whatever budget the window
    was eventually scanned at.
    """
    findings: list[Finding] = []
    for item in raw_items:
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        start = int(start)
        end = int(end)
        while start < end and chunk[start].isspace():
            start += 1
        while end > start and chunk[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        label = normalize_label(str(item.get("entity_group") or item.get("entity") or "unknown"))
        score = item.get("score")
        findings.append(
            Finding(
                label=label,
                start=base + start,
                end=base + end,
                text=chunk[start:end],
                score=float(score) if score is not None else None,
            )
        )
    return findings


def is_merge_gap(gap: str) -> bool:
    return len(gap) <= 2 and all(char.isspace() or char in ".-_+@" for char in gap)


def merge_findings(findings: list[Finding], text: str) -> list[Finding]:
    """Coalesce same-label findings that overlap, touch, or nearly touch.

    Overlapping windows report the boundary region twice, so the same entity
    arrives two or three times, sometimes clipped differently by each window.
    Merging by union in original coordinates collapses those into one finding
    while leaving genuinely separate entities -- anything more than a joining
    character or two apart -- alone.
    """
    if not findings:
        return []

    merged: list[Finding] = []
    for finding in sorted(findings, key=lambda item: (item.start, item.end)):
        if merged and joins(merged[-1], finding, text):
            merged[-1] = join(merged[-1], finding, text)
            continue
        merged.append(finding)
    return merged


def joins(previous: Finding, finding: Finding, text: str) -> bool:
    if previous.label != finding.label:
        return False
    if finding.start <= previous.end:
        return True
    return is_merge_gap(text[previous.end : finding.start])


def join(previous: Finding, finding: Finding, text: str) -> Finding:
    end = max(previous.end, finding.end)
    scores = [score for score in (previous.score, finding.score) if score is not None]
    return Finding(
        label=previous.label,
        start=previous.start,
        end=end,
        text=text[previous.start : end],
        # The same entity seen from two windows keeps its strongest score;
        # averaging would dilute a confident hit with a clipped one.
        score=max(scores) if scores else None,
    )


def redact_text(text: str, findings: list[Finding]) -> str:
    """Replace each finding with a label marker, one marker per entity."""
    for finding in sorted(non_overlapping(findings), key=lambda item: item.start, reverse=True):
        marker = f"[{finding.label.upper()}]"
        text = text[: finding.start] + marker + text[finding.end :]
    return text


def non_overlapping(findings: list[Finding]) -> list[Finding]:
    """Drop findings that overlap a higher-scoring one.

    ``merge_findings`` already collapses same-label duplicates; what can still
    overlap is two different labels claiming the same characters. Redaction has
    to pick one, otherwise the markers interleave and corrupt the offsets.
    """
    kept: list[Finding] = []
    for finding in sorted(findings, key=lambda item: (-(item.score or 0.0), item.start)):
        if any(finding.start < other.end and finding.end > other.start for other in kept):
            continue
        kept.append(finding)
    return kept


class AdaptiveScanner:
    """Runs inference over planned windows, shrinking the budget on OOM.

    ``plan_windows(text, budget, overlap)`` returns character spans of ``text``;
    ``classify(chunk)`` runs the model on one window. ``oom_errors`` are the
    exception types that mean "this window did not fit in memory" -- CUDA's
    out-of-memory errors in production, a fake in the tests.
    """

    def __init__(
        self,
        *,
        plan_windows: PlanWindows,
        classify: Classify,
        policy: BudgetPolicy | None = None,
        oom_errors: tuple[type[BaseException], ...] = (),
        on_oom: Callable[[], None] | None = None,
        lock: Any = None,
    ) -> None:
        self.plan_windows = plan_windows
        self.classify = classify
        self.policy = policy or BudgetPolicy()
        self.budgets = self.policy.ladder()
        self.oom_errors = tuple(oom_errors)
        self.on_oom = on_oom
        # One lock per process: FastAPI runs sync endpoints on a thread pool, so
        # two requests would otherwise allocate transient CUDA memory at the same
        # time and OOM a GPU that either request alone would fit on.
        self.lock = lock if lock is not None else threading.Lock()

    def scan(self, text: str) -> list[Finding]:
        if not text:
            return []
        findings = self.scan_span(text, 0, len(text), 0)
        return merge_findings(findings, text)

    def scan_span(self, text: str, start: int, end: int, level: int) -> list[Finding]:
        budget = self.budgets[level]
        findings: list[Finding] = []
        for window_start, window_end in self.plan_windows(
            text[start:end], budget, self.policy.overlap_tokens
        ):
            absolute_start = start + window_start
            absolute_end = start + window_end
            try:
                findings.extend(self.classify_window(text, absolute_start, absolute_end))
            except self.oom_errors as exc:
                # Drop nothing but this window's (unusable) results, hand the
                # memory back, and re-plan just this span at a smaller budget.
                if self.on_oom is not None:
                    self.on_oom()
                if level + 1 >= len(self.budgets):
                    raise ScanBudgetExhausted(budget) from exc
                findings.extend(
                    self.scan_span(text, absolute_start, absolute_end, level + 1)
                )
        return findings

    def classify_window(self, text: str, start: int, end: int) -> list[Finding]:
        chunk = text[start:end]
        if not chunk:
            return []
        with self.lock:
            raw = list(self.classify(chunk))
        # Deliberately outside the lock: only the forward pass needs serializing.
        return findings_from_raw(chunk, start, raw)
