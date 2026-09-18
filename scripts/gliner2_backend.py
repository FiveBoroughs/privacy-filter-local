"""GLiNER2 PII backend with canonical labels and verified character offsets.

The upstream extractor returns fine-grained labels and optional character
spans. This adapter requests all labels supported by the checkpoint, folds them
into the service's stable eight-label vocabulary, and refuses malformed spans
instead of reconstructing positions from matched text.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


DEFAULT_GLINER2_MODEL = "fastino/gliner2-privacy-filter-PII-multi"

GLINER2_LABEL_MAP = {
    "person": "private_person",
    "full_name": "private_person",
    "first_name": "private_person",
    "middle_name": "private_person",
    "last_name": "private_person",
    "email": "private_email",
    "phone_number": "private_phone",
    "address": "private_address",
    "street_address": "private_address",
    "city": "private_address",
    "state_or_region": "private_address",
    "postal_code": "private_address",
    "country": "private_address",
    "date_of_birth": "private_date",
    "sensitive_date": "private_date",
    "document_date": "private_date",
    "expiration_date": "private_date",
    "transaction_date": "private_date",
    "card_expiry": "private_date",
    "government_id": "account_number",
    "national_id_number": "account_number",
    "passport_number": "account_number",
    "drivers_license_number": "account_number",
    "license_number": "account_number",
    "tax_id": "account_number",
    "tax_number": "account_number",
    "bank_account": "account_number",
    "account_number": "account_number",
    "routing_number": "account_number",
    "iban": "account_number",
    "payment_card": "account_number",
    "card_number": "account_number",
    "username": "private_person",
    "ip_address": "private_url",
    "account_id": "secret",
    "sensitive_account_id": "secret",
    "password": "secret",
    "secret": "secret",
    "api_key": "secret",
    "access_token": "secret",
    "recovery_code": "secret",
    "card_cvv": "secret",
}
GLINER2_LABELS = tuple(GLINER2_LABEL_MAP)


class Gliner2Tokenizer:
    """Text tokenizer plus the schema tokens GLiNER2 adds to every request."""

    def __init__(self, tokenizer: Any, schema_tokens: int) -> None:
        self._tokenizer = tokenizer
        self._schema_tokens = schema_tokens
        self.is_fast = bool(getattr(tokenizer, "is_fast", False))
        self.model_max_length = getattr(tokenizer, "model_max_length", None)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        if pair:
            raise ValueError("pair tokenization is not used by the Privacy Filter")
        return self._schema_tokens

    def __call__(self, text: str, **kwargs: Any) -> Any:
        return self._tokenizer(text, **kwargs)


def schema_token_overhead(extractor: Any) -> int:
    """Exact encoded length of the fixed 42-label schema with no input text."""
    schema = {"entities": {label: "" for label in GLINER2_LABELS}}
    batch = extractor.processor.collate_fn_inference(
        [("", schema)],
        error_policy="raise",
        architecture=extractor.architecture,
    )
    return int(batch.attention_mask[0].sum().item())


def findings_from_result(text: str, result: Any) -> list[dict[str, Any]]:
    """Flatten GLiNER2 output, rejecting anything without trustworthy offsets."""
    if not isinstance(result, dict) or not isinstance(result.get("entities"), dict):
        raise RuntimeError("GLiNER2 returned an invalid entity result; nothing was scanned")

    findings: list[dict[str, Any]] = []
    for label, entities in result["entities"].items():
        canonical = GLINER2_LABEL_MAP.get(label)
        if canonical is None:
            raise RuntimeError("GLiNER2 returned an unmapped label; nothing was scanned")
        if not isinstance(entities, list):
            raise RuntimeError("GLiNER2 returned an invalid entity list; nothing was scanned")
        seen_spans: set[tuple[int, int]] = set()
        for entity in entities:
            if not isinstance(entity, dict):
                raise RuntimeError("GLiNER2 returned an entity without spans; nothing was scanned")
            start = entity.get("start")
            end = entity.get("end")
            matched = entity.get("text")
            score = entity.get("confidence")
            if (
                type(start) is not int
                or type(end) is not int
                or not isinstance(matched, str)
                or not 0 <= start < end <= len(text)
                or text[start:end] != matched
            ):
                raise RuntimeError("GLiNER2 returned invalid character offsets; nothing was scanned")
            position = (start, end)
            if position in seen_spans:
                raise RuntimeError(
                    "GLiNER2 returned duplicate character offsets; nothing was scanned"
                )
            seen_spans.add(position)
            findings.append(
                {
                    "entity_group": canonical,
                    "start": start,
                    "end": end,
                    "score": float(score) if score is not None else None,
                }
            )
    return findings


class Gliner2Classifier:
    """Callable adapter matching the service's token-classifier contract."""

    def __init__(self, model_name: str, device: str = "auto", threshold: float = 0.5) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("PRIVACY_FILTER_GLINER2_DEVICE must be one of: auto, cpu, cuda")
        if not 0 <= threshold <= 1:
            raise ValueError("PRIVACY_FILTER_GLINER2_THRESHOLD must be between 0 and 1")

        import torch
        from gliner2 import AutoExtractor

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "PRIVACY_FILTER_GLINER2_DEVICE=cuda, but torch.cuda.is_available() is false"
            )
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"

        extractor = AutoExtractor.from_pretrained(
            model_name,
            map_location=resolved_device,
            quantize=resolved_device == "cuda",
        )
        extractor.strict_extraction = True

        encoder_config = getattr(extractor.encoder, "config", None)
        dtype = next(extractor.parameters()).dtype
        self.model = SimpleNamespace(
            config={"encoder_config": encoder_config},
            dtype=dtype,
        )
        self.tokenizer = Gliner2Tokenizer(
            extractor.processor.tokenizer,
            schema_token_overhead(extractor),
        )
        self.provider = "CUDA" if resolved_device == "cuda" else "CPU"
        self.gpu_name = torch.cuda.get_device_name(0) if resolved_device == "cuda" else ""
        self.threshold = threshold
        self._extractor = extractor

    def __call__(self, text: str) -> list[dict[str, Any]]:
        result = self._extractor.extract_entities(
            text,
            list(GLINER2_LABELS),
            threshold=self.threshold,
            include_confidence=True,
            include_spans=True,
        )
        return findings_from_result(text, result)
