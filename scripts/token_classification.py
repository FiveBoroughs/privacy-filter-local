"""Pure BIO token grouping compatible with transformers' ``simple`` strategy.

The ONNX backend produces one label, score and character span per token. Keeping
this grouping independent of numpy and onnxruntime makes the subtle boundary
rules testable on a plain interpreter, like the rest of the scanner core.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

ONNX_OOM_MARKERS = (
    "failed to allocate memory",
    "out of memory",
    "cuda failure 2",
)


def cuda_safe_export(model_file: str) -> bool:
    """Whether this export has produced complete findings on CUDA EP.

    ONNX Runtime 1.30 has no valid int4 MoE GEMM configuration for q4/q4f16
    on the tested RTX 3080 and returned incomplete findings without raising.
    """
    return "q4" not in os.path.basename(model_file).lower()


def is_onnx_oom(message: str) -> bool:
    """Recognize only explicit allocator failures from broad ORT exceptions."""
    lowered = message.lower()
    return any(marker in lowered for marker in ONNX_OOM_MARKERS)


def label_tag(label: str) -> tuple[str, str]:
    """Return the B/I prefix and entity tag exactly as transformers does.

    Labels without B-/I- are continuations. That includes this model's E-/S-
    labels: transformers' ``simple`` strategy splits them into their own groups;
    ``merge_findings`` later reunites adjacent spans after normalizing the label.
    """
    if label.startswith("B-"):
        return "B", label[2:]
    if label.startswith("I-"):
        return "I", label[2:]
    return "I", label


def simple_groups(
    predictions: Iterable[dict[str, Any]],
    ignore_labels: frozenset[str] = frozenset({"O"}),
) -> list[dict[str, Any]]:
    """Group token predictions like TokenClassificationPipeline ``simple``.

    Each prediction must have ``entity``, ``score``, ``start`` and ``end``.
    Output uses the raw finding shape consumed by ``findings_from_raw``.
    """
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for prediction in predictions:
        if current:
            prefix, tag = label_tag(str(prediction["entity"]))
            _, previous_tag = label_tag(str(current[-1]["entity"]))
            if tag != previous_tag or prefix == "B":
                groups.append(current)
                current = []
        current.append(prediction)
    if current:
        groups.append(current)

    output = []
    for group in groups:
        label = str(group[0]["entity"]).split("-", 1)[-1]
        if label in ignore_labels:
            continue
        output.append(
            {
                "entity_group": label,
                "score": sum(float(item["score"]) for item in group) / len(group),
                "start": int(group[0]["start"]),
                "end": int(group[-1]["end"]),
            }
        )
    return output
