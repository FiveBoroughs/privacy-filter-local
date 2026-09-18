"""Pinned pplx-pii-masking backend without Hugging Face remote-code execution.

The checkpoint requires custom Python. Its MIT-licensed model implementation is
vendored unchanged as ``pplx_model.py`` from the immutable revision below;
only weights, configuration and tokenizer data are downloaded. The service
never executes a Python file fetched from the model repository.
"""

from __future__ import annotations

from hashlib import sha256
from math import isfinite
from pathlib import Path

from typing import Any, Iterable


DEFAULT_PPLX_MODEL = "perplexity-ai/pplx-pii-masking"
PPLX_REVISION = "d8bb208c4fd23070493c42b39fb3454e746c7607"
PPLX_MODEL_SOURCE_SHA256 = "c43724b9ac105a9b74d44aea7565e4c43ddc9d203f519324baa342be01269869"
PPLX_LABEL_MAP = {
    "private_person": "private_person",
    "private_email": "private_email",
    "private_phone": "private_phone",
    "private_address": "private_address",
    "private_url": "private_url",
    "private_date": "private_date",
    "account_number": "account_number",
    "secret": "secret",
    # The service's stable public vocabulary has no generic PII label. Treat the
    # checkpoint's catch-all as sensitive rather than dropping a detected span.
    "other_pii": "secret",
}


def verify_vendored_source() -> None:
    """Refuse to execute custom model code outside the audited source digest."""
    source = Path(__file__).with_name("pplx_model.py")
    digest = sha256(source.read_bytes()).hexdigest()
    if digest != PPLX_MODEL_SOURCE_SHA256:
        raise RuntimeError(
            "Vendored PPLX model code does not match the audited revision; "
            "nothing was scanned"
        )


def findings_from_spans(text: str, spans: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalize model spans, rejecting malformed or duplicate coordinates."""
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for span in spans:
        label = getattr(span, "label", None)
        canonical = PPLX_LABEL_MAP.get(label)
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        score = getattr(span, "score", None)
        if canonical is None:
            raise RuntimeError("PPLX returned an unmapped label; nothing was scanned")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
        ):
            raise RuntimeError("PPLX returned invalid character offsets; nothing was scanned")
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("PPLX returned an invalid score; nothing was scanned") from exc
        if not isfinite(numeric_score):
            raise RuntimeError("PPLX returned an invalid score; nothing was scanned")
        key = (canonical, start, end)
        if key in seen:
            raise RuntimeError("PPLX returned duplicate character offsets; nothing was scanned")
        seen.add(key)
        findings.append(
            {
                "entity_group": canonical,
                "start": start,
                "end": end,
                "score": numeric_score,
            }
        )
    return findings


class PplxClassifier:
    """Callable adapter matching the service's token-classifier contract."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        if model_name != DEFAULT_PPLX_MODEL:
            raise ValueError(
                "The PPLX backend only loads its audited pinned model; "
                f"expected {DEFAULT_PPLX_MODEL}"
            )
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("PRIVACY_FILTER_PPLX_DEVICE must be one of: auto, cpu, cuda")

        verify_vendored_source()

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        from pplx_model import PiiMaskingConfig, PiiMaskingModel

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "PRIVACY_FILTER_PPLX_DEVICE=cuda, but torch.cuda.is_available() is false"
            )
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"

        snapshot = snapshot_download(
            model_name,
            revision=PPLX_REVISION,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
        config = PiiMaskingConfig.from_pretrained(snapshot, local_files_only=True)
        model = PiiMaskingModel.from_pretrained(
            snapshot,
            config=config,
            local_files_only=True,
            dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model.tokenizer = tokenizer
        model.to(resolved_device)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.provider = "CUDA" if resolved_device == "cuda" else "CPU"
        self.gpu_name = torch.cuda.get_device_name(0) if resolved_device == "cuda" else ""

    def __call__(self, text: str) -> list[dict[str, Any]]:
        spans, _sensitivity = self.model.predict(
            text,
            tokenizer=self.tokenizer,
            max_length=self.model.config.max_seq_len,
        )
        return findings_from_spans(text, spans)
