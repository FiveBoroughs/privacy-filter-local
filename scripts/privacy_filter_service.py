from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

from adaptive_scan import (
    AdaptiveScanner,
    BudgetExceedsContextLimit,
    BudgetPolicy,
    ContextLimitError,
    ContextLimitUnknown,
    Finding,
    IMPLAUSIBLE_CONTEXT_LIMIT,
    ScanBudgetExhausted,
    plausible_limit,
    redact_text,
    resolve_context_limit,
    window_budget,
)
from text_chunking import Span, plan_chunks


DEFAULT_MODEL = "openai/privacy-filter"

# Reserve a little headroom below the model's context limit for the special
# tokens the pipeline adds, plus slack for any tokenization drift between our
# offset pass and the pipeline's own tokenization.
TOKEN_BUDGET_MARGIN = 16
# The model's real maximum sequence length, for a model whose config does not
# declare one. There is no default: a guess here is a window the model truncates
# silently, which is a scan that reports success while missing findings.
CONTEXT_LIMIT = os.environ.get("PRIVACY_FILTER_CONTEXT_LIMIT") or None
# VRAM, not context length, is what bounds this: a single 30k-token forward pass
# allocates enough transient GPU memory to OOM a VRAM-starved card (e.g. while
# gaming), even on a model with a 128k context. Cap tokens per request well
# below the context window to bound that allocation; 4096 is the initial context
# length of the default model, a natural, well-supported window size. This is
# only a starting point: a window that still does not fit is retried smaller
# (see MIN_TOKENS). Our own default is sized for the GPU rather than for the
# loaded model, so a model with a shorter context clamps it; a value the
# operator set above what the model accepts is refused at startup instead.
MAX_TOKENS_CONFIGURED = bool(os.environ.get("PRIVACY_FILTER_MAX_TOKENS"))
MAX_TOKENS = int(os.environ.get("PRIVACY_FILTER_MAX_TOKENS") or 4096)
# Floor of the retry ladder. Below this the per-window overhead dominates and a
# GPU that cannot fit 256 tokens cannot usefully run the model at all, so this
# is where the scan gives up and fails closed instead of shrinking forever.
MIN_TOKENS = int(os.environ.get("PRIVACY_FILTER_MIN_TOKENS", "256"))
# Halving is aggressive enough to escape a transient VRAM squeeze in two or
# three retries without re-running most of the input at a barely smaller size.
SHRINK_FACTOR = float(os.environ.get("PRIVACY_FILTER_TOKEN_SHRINK_FACTOR", "0.5"))
# Tokens each window re-reads from its predecessor. 64 subword tokens is far
# longer than any single entity the model labels (a name, an address, an IBAN),
# so PII cannot hide in a cut, while costing only ~1.5% extra forward passes at
# the default 4096-token budget.
OVERLAP_TOKENS = int(os.environ.get("PRIVACY_FILTER_CHUNK_OVERLAP_TOKENS", "64"))


class FilterRequest(BaseModel):
    text: str
    include_text: bool = False


class FilterState:
    classifier: Any = None
    scanner: AdaptiveScanner | None = None
    model_name: str = os.environ.get("PRIVACY_FILTER_MODEL", DEFAULT_MODEL)
    gpu_name: str = ""
    # In-flight scans, so a caller deciding whether to stop the container can
    # see that another one is still working.
    active_scans: int = 0
    scan_counter: threading.Lock = threading.Lock()


state = FilterState()


def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")
    return torch.cuda.get_device_name(0)


def load_classifier() -> None:
    state.gpu_name = require_cuda()
    state.classifier = pipeline(
        task="token-classification",
        model=state.model_name,
        aggregation_strategy="simple",
        device=0,
        # Load in the checkpoint's native bfloat16 (~2.8GB) instead of letting
        # transformers upcast to fp32 (~5.6GB). Halving the resident footprint is
        # what lets the MoE share the GPU with a game without OOM-ing.
        model_kwargs={"dtype": "auto"},
    )
    try:
        state.scanner = build_scanner()
    except ContextLimitError as exc:
        raise RuntimeError(startup_refusal(exc)) from exc


def startup_refusal(exc: ContextLimitError) -> str:
    """Why the service will not start, and what to change. Never any input text."""
    model = state.model_name
    if isinstance(exc, BudgetExceedsContextLimit):
        return (
            f"PRIVACY_FILTER_MAX_TOKENS is {exc.requested}, but {model} accepts at most "
            f"{exc.limit} tokens per forward pass. Larger windows would be truncated "
            "silently and any finding past the cut would be lost, so nothing is "
            f"scanned. Lower PRIVACY_FILTER_MAX_TOKENS to {exc.limit} or below."
        )
    if isinstance(exc, ContextLimitUnknown):
        return (
            f"{model} declares no context length in its config, so no window size can "
            "be shown to fit it and nothing is scanned. Set PRIVACY_FILTER_CONTEXT_LIMIT "
            "to the model's real maximum sequence length."
        )
    return f"{exc}; nothing is scanned (model {model})."


def declared_context_limit() -> int | None:
    """The operator's PRIVACY_FILTER_CONTEXT_LIMIT, as an integer.

    Rejected here rather than deeper down so the refusal names the variable the
    operator actually set, whether it is unparseable or out of range.
    """
    if CONTEXT_LIMIT is None:
        return None
    try:
        limit = plausible_limit(int(CONTEXT_LIMIT))
    except ValueError:
        limit = None
    if limit is None:
        raise ContextLimitError(
            "PRIVACY_FILTER_CONTEXT_LIMIT must be a whole number of tokens between 1 "
            f"and {IMPLAUSIBLE_CONTEXT_LIMIT}, got {CONTEXT_LIMIT!r}"
        )
    return limit


def context_limit() -> int:
    """The largest window the model itself accepts, in tokens."""
    classifier = state.classifier
    return resolve_context_limit(
        getattr(classifier.model, "config", None),
        getattr(classifier.tokenizer, "model_max_length", None),
        declared=declared_context_limit(),
    )


def max_token_budget() -> int:
    """Largest window we will ever feed the model, in tokens."""
    tokenizer = state.classifier.tokenizer
    try:
        special = tokenizer.num_special_tokens_to_add(pair=False)
    except Exception:
        special = 2
    return window_budget(
        MAX_TOKENS,
        context_limit(),
        special,
        TOKEN_BUDGET_MARGIN,
        explicit=MAX_TOKENS_CONFIGURED,
    )


def budget_policy() -> BudgetPolicy:
    return BudgetPolicy(
        max_tokens=max_token_budget(),
        min_tokens=MIN_TOKENS,
        shrink_factor=SHRINK_FACTOR,
        overlap_tokens=OVERLAP_TOKENS,
    )


def token_offsets(text: str) -> list[Span]:
    tokenizer = state.classifier.tokenizer
    if getattr(tokenizer, "is_fast", False):
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        return list(encoding["offset_mapping"])
    # No fast tokenizer means no offset mapping; fall back to one offset per
    # character. Tokens span at least one char, so a budget measured in
    # characters can only undershoot the token budget -- never exceed it.
    return [(index, index + 1) for index in range(len(text))]


def plan_windows(text: str, budget: int, overlap: int) -> list[Span]:
    return plan_chunks(text, token_offsets(text), budget, overlap)


def classify_window(chunk: str) -> list[dict[str, Any]]:
    return list(state.classifier(chunk))


def oom_error_types() -> tuple[type[BaseException], ...]:
    """CUDA out-of-memory exception types this torch build raises."""
    types: list[type[BaseException]] = []
    for candidate in (
        getattr(torch.cuda, "OutOfMemoryError", None),
        getattr(torch, "OutOfMemoryError", None),
    ):
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            if candidate not in types:
                types.append(candidate)
    return tuple(types) or (MemoryError,)


def build_scanner() -> AdaptiveScanner:
    return AdaptiveScanner(
        plan_windows=plan_windows,
        classify=classify_window,
        policy=budget_policy(),
        oom_errors=oom_error_types(),
        on_oom=torch.cuda.empty_cache,
    )


def serialize_findings(findings: list[Finding], include_text: bool) -> list[dict[str, Any]]:
    serialized = []
    for finding in findings:
        data = asdict(finding)
        if not include_text:
            data.pop("text", None)
        serialized.append(data)
    return serialized


def service_error(status: int, code: str, message: str) -> HTTPException:
    """A machine-readable error body so clients can react per failure mode."""
    return HTTPException(status_code=status, detail={"error": code, "message": message})


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_classifier()
    yield


app = FastAPI(title="Local Privacy Filter", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    """Non-sensitive diagnostics only: never input text, never finding text."""
    scanner = state.scanner
    policy = scanner.policy if scanner is not None else None
    return {
        "ok": state.classifier is not None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": state.gpu_name,
        "model": state.model_name,
        "dtype": str(state.classifier.model.dtype) if state.classifier is not None else None,
        # The model's own maximum sequence length, so an operator can see what
        # max_tokens is being held below without reading the model's config.
        "context_limit": context_limit() if state.classifier is not None else None,
        "max_tokens": policy.max_tokens if policy else MAX_TOKENS,
        "min_tokens": policy.min_tokens if policy else MIN_TOKENS,
        "shrink_factor": policy.shrink_factor if policy else SHRINK_FACTOR,
        "overlap_tokens": policy.overlap_tokens if policy else OVERLAP_TOKENS,
        "token_budgets": list(scanner.budgets) if scanner is not None else [],
        "inference_serialized": True,
        # Lets a caller about to stop the container see that someone else is
        # still scanning. A count, never what is being scanned.
        "active_scans": state.active_scans,
    }


def scan(text: str) -> list[Finding]:
    """Scan with adaptive retries, mapping each failure to a distinct error.

    Callers (the CLI, the pre-commit hook) fail closed, so the failure mode has
    to be legible: "the GPU is busy, retry" is a different instruction to "this
    service has no GPU". Neither ever includes scanned text.
    """
    scanner = state.scanner
    if scanner is None or not torch.cuda.is_available():
        raise service_error(
            503,
            "no_gpu",
            "Privacy Filter is not GPU-backed; nothing was scanned.",
        )
    with state.scan_counter:
        state.active_scans += 1
    try:
        return scanner.scan(text)
    except ScanBudgetExhausted as exc:
        raise service_error(
            503,
            "budget_exhausted",
            (
                f"Ran out of GPU memory even at the minimum window of {exc.budget} tokens; "
                "nothing was scanned. Free VRAM (e.g. close a game), lower "
                "PRIVACY_FILTER_MAX_TOKENS, or retry."
            ),
        ) from exc
    finally:
        with state.scan_counter:
            state.active_scans -= 1


@app.post("/check")
def check(request: FilterRequest) -> dict[str, Any]:
    findings = scan(request.text)
    return {
        "count": len(findings),
        "findings": serialize_findings(findings, request.include_text),
    }


@app.post("/redact")
def redact(request: FilterRequest) -> dict[str, Any]:
    findings = scan(request.text)
    return {
        "count": len(findings),
        "findings": serialize_findings(findings, request.include_text),
        "redacted": redact_text(request.text, findings),
    }
