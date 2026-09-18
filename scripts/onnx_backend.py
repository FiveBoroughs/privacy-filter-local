"""Raw ONNX Runtime token-classification backend.

No torch, transformers or optimum dependency. The model graph is already
exported; Hugging Face's ``tokenizer.json`` provides the same Rust tokenizer and
character offsets the transformers pipeline uses.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from token_classification import cuda_safe_export, is_onnx_oom, simple_groups


class OnnxOutOfMemory(MemoryError):
    """ONNX Runtime's CUDA allocator could not fit the current window."""

DEFAULT_ONNX_FILE = "onnx/model_quantized.onnx"


class OnnxTokenizer:
    """The tokenizer surface the service needs for planning and diagnostics."""

    is_fast = True

    def __init__(self, snapshot: str) -> None:
        with open(os.path.join(snapshot, "tokenizer_config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.model_max_length = config.get("model_max_length")
        self._tokenizer = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
        empty = self._tokenizer.encode("", add_special_tokens=True)
        self._special_tokens = sum(empty.special_tokens_mask)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        if pair:
            raise ValueError("pair tokenization is not used by the Privacy Filter")
        return self._special_tokens

    def __call__(self, text: str, **_: Any) -> dict[str, list[tuple[int, int]]]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return {"offset_mapping": list(encoding.offsets)}

    def encode(self, text: str):
        return self._tokenizer.encode(text, add_special_tokens=True)


class OnnxClassifier:
    """Callable with the same finding output shape as transformers.pipeline."""

    def __init__(
        self,
        model_name: str,
        model_file: str = DEFAULT_ONNX_FILE,
        revision: str | None = None,
        provider: str = "auto",
    ) -> None:
        parts = model_file.replace("\\", "/").split("/")
        if (
            not model_file.endswith(".onnx")
            or os.path.isabs(model_file)
            or ".." in parts
            or "" in parts
        ):
            raise ValueError(
                "PRIVACY_FILTER_ONNX_FILE must be a relative .onnx path without '..'"
            )
        snapshot = snapshot_download(
            model_name,
            revision=revision,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                model_file,
                f"{model_file}_data*",
            ],
        )
        graph = os.path.join(snapshot, model_file)
        if not os.path.isfile(graph):
            raise FileNotFoundError(f"ONNX graph not found in model repository: {model_file}")

        with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.labels = {int(key): value for key, value in config["id2label"].items()}
        self.tokenizer = OnnxTokenizer(snapshot)
        self.model_file = model_file
        self.model = SimpleNamespace(config=config, dtype=self._dtype_name(model_file))
        requested = self._requested_providers(provider)
        self.session = ort.InferenceSession(graph, providers=requested)
        providers = self.session.get_providers()
        self.provider = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in providers
            else "CPUExecutionProvider"
        )
        if provider == "cuda" and self.provider != "CUDAExecutionProvider":
            raise RuntimeError(
                "PRIVACY_FILTER_ONNX_PROVIDER=cuda, but CUDAExecutionProvider "
                "could not be loaded"
            )
        if (
            self.provider == "CUDAExecutionProvider"
            and not cuda_safe_export(model_file)
        ):
            raise RuntimeError(
                f"{model_file} is unsafe on CUDAExecutionProvider: ONNX Runtime 1.30 "
                "has no valid int4 MoE GEMM configuration on the tested RTX 3080 and "
                "returned incomplete findings. Use model_quantized.onnx, or select "
                "PRIVACY_FILTER_ONNX_PROVIDER=cpu explicitly."
            )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name
        unsupported = self.input_names - {"input_ids", "attention_mask", "token_type_ids"}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise RuntimeError(f"ONNX graph requires unsupported inputs: {names}")

    @staticmethod
    def _dtype_name(model_file: str) -> str:
        filename = os.path.basename(model_file)
        if "q4f16" in filename:
            return "q4f16"
        if "q4" in filename:
            return "q4"
        if "quantized" in filename:
            return "int8"
        if "fp16" in filename:
            return "float16"
        return "float32"


    @staticmethod
    def _requested_providers(provider: str) -> list[str]:
        if provider not in {"auto", "cuda", "cpu"}:
            raise ValueError(
                "PRIVACY_FILTER_ONNX_PROVIDER must be one of: auto, cpu, cuda"
            )
        available = ort.get_available_providers()
        if provider == "cpu":
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "cuda":
            raise RuntimeError(
                "PRIVACY_FILTER_ONNX_PROVIDER=cuda, but this ONNX Runtime build "
                "does not contain CUDAExecutionProvider"
            )
        return ["CPUExecutionProvider"]
    def __call__(self, text: str) -> list[dict[str, Any]]:
        encoding = self.tokenizer.encode(text)
        arrays = {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask,
            "token_type_ids": encoding.type_ids,
        }
        inputs = {
            name: np.asarray([arrays[name]], dtype=np.int64)
            for name in self.input_names
        }
        try:
            logits = self.session.run([self.output_name], inputs)[0][0]
        except Exception as exc:
            if self.provider == "CUDAExecutionProvider" and is_onnx_oom(str(exc)):
                raise OnnxOutOfMemory(
                    "ONNX CUDA allocator exhausted at the current token budget"
                ) from exc
            raise

        maxima = np.max(logits, axis=-1, keepdims=True)
        shifted = np.exp(logits - maxima)
        scores = shifted / shifted.sum(axis=-1, keepdims=True)
        predictions = []
        for token_scores, (start, end), special in zip(
            scores,
            encoding.offsets,
            encoding.special_tokens_mask,
        ):
            if special:
                continue
            label_id = int(token_scores.argmax())
            predictions.append(
                {
                    "entity": self.labels[label_id],
                    "score": float(token_scores[label_id]),
                    "start": start,
                    "end": end,
                }
            )
        return simple_groups(predictions)
