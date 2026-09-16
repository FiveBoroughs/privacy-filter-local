"""Tests for the service's error taxonomy, health diagnostics and wiring.

scripts/privacy_filter_service.py imports torch, transformers and FastAPI, which
only exist inside the GPU container. Stubbing them here keeps the HTTP-facing
behaviour -- which status code, which error identifier, what /health reveals --
under test on a plain interpreter. Every fixture is synthetic.

Run: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class FakeOutOfMemory(Exception):
    """Stands in for torch.cuda.OutOfMemoryError."""


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeApp:
    def __init__(self, *_, **__):
        pass

    def get(self, *_, **__):
        return lambda handler: handler

    post = get


def install_stub_modules() -> None:
    """Put minimal torch/transformers/fastapi/pydantic stubs on sys.modules."""
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: "FakeGPU",
        empty_cache=lambda: None,
        OutOfMemoryError=FakeOutOfMemory,
    )
    torch = types.ModuleType("torch")
    torch.cuda = cuda
    torch.OutOfMemoryError = FakeOutOfMemory

    transformers = types.ModuleType("transformers")
    transformers.pipeline = lambda **kwargs: None

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeApp
    fastapi.HTTPException = FakeHTTPException

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})

    for name, module in (
        ("torch", torch),
        ("transformers", transformers),
        ("fastapi", fastapi),
        ("pydantic", pydantic),
    ):
        sys.modules.setdefault(name, module)


install_stub_modules()

import privacy_filter_service as service  # noqa: E402
from adaptive_scan import AdaptiveScanner, BudgetPolicy, ScanBudgetExhausted  # noqa: E402
from text_chunking import plan_chunks  # noqa: E402


class FakeTokenizer:
    """A fast tokenizer whose tokens are whitespace-delimited words."""

    is_fast = True
    model_max_length = 131072

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **_):
        offsets = []
        index = 0
        for word in text.split(" "):
            if word:
                offsets.append((index, index + len(word)))
            index += len(word) + 1
        return {"offset_mapping": offsets}


class FakeClassifier:
    def __init__(self, behaviour=None):
        self.tokenizer = FakeTokenizer()
        self.model = types.SimpleNamespace(dtype="torch.bfloat16")
        self.behaviour = behaviour or (lambda chunk: [])

    def __call__(self, chunk):
        return self.behaviour(chunk)


class ServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self.reset_state)

    def reset_state(self):
        service.state.classifier = None
        service.state.scanner = None
        service.state.gpu_name = ""
        service.torch.cuda.is_available = lambda: True

    def install(self, behaviour=None, **policy_kwargs):
        service.state.classifier = FakeClassifier(behaviour)
        service.state.gpu_name = "FakeGPU"
        policy = BudgetPolicy(
            **{"max_tokens": 20, "min_tokens": 5, "overlap_tokens": 4, **policy_kwargs}
        )
        service.state.scanner = AdaptiveScanner(
            plan_windows=service.plan_windows,
            classify=service.classify_window,
            policy=policy,
            oom_errors=service.oom_error_types(),
            on_oom=service.torch.cuda.empty_cache,
        )


class ScanErrorTest(ServiceTestCase):
    def test_exhausted_budget_is_a_503_that_names_the_failure(self):
        def always_oom(chunk):
            raise FakeOutOfMemory("no memory")

        self.install(always_oom)
        with self.assertRaises(FakeHTTPException) as ctx:
            service.scan("word " * 100)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail["error"], "budget_exhausted")
        self.assertIn("nothing was scanned", ctx.exception.detail["message"])

    def test_error_messages_never_leak_scanned_text(self):
        def always_oom(chunk):
            raise FakeOutOfMemory("no memory")

        self.install(always_oom)
        with self.assertRaises(FakeHTTPException) as ctx:
            service.scan("Ada Lovelace ada@example.com " * 40)
        self.assertNotIn("Ada", str(ctx.exception.detail))
        self.assertNotIn("example.com", str(ctx.exception.detail))

    def test_missing_gpu_is_a_distinct_503(self):
        self.install()
        service.torch.cuda.is_available = lambda: False
        with self.assertRaises(FakeHTTPException) as ctx:
            service.scan("hello")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail["error"], "no_gpu")

    def test_unloaded_service_is_a_distinct_503(self):
        with self.assertRaises(FakeHTTPException) as ctx:
            service.scan("hello")
        self.assertEqual(ctx.exception.detail["error"], "no_gpu")

    def test_a_window_that_fits_after_shrinking_succeeds(self):
        def oom_above_eight(chunk):
            if len(chunk.split()) > 8:
                raise FakeOutOfMemory("window too large")
            start = chunk.find("NAME1")
            if start < 0:
                return []
            return [{"entity_group": "private_person", "start": start, "end": start + 5, "score": 0.9}]

        self.install(oom_above_eight)
        text = " ".join(["word"] * 40) + " NAME1 " + " ".join(["word"] * 40)
        findings = service.scan(text)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].text, "NAME1")
        self.assertEqual(findings[0].start, text.index("NAME1"))


class EndpointTest(ServiceTestCase):
    def behaviour(self, chunk):
        start = chunk.find("NAME1")
        if start < 0:
            return []
        return [{"entity_group": "private_person", "start": start, "end": start + 5, "score": 0.9}]

    class Request:
        def __init__(self, text, include_text=False):
            self.text = text
            self.include_text = include_text

    def test_check_omits_matched_text_by_default(self):
        self.install(self.behaviour)
        response = service.check(self.Request("hello NAME1 there"))
        self.assertEqual(response["count"], 1)
        self.assertNotIn("text", response["findings"][0])

    def test_check_can_include_text_on_request(self):
        self.install(self.behaviour)
        response = service.check(self.Request("hello NAME1 there", include_text=True))
        self.assertEqual(response["findings"][0]["text"], "NAME1")

    def test_redact_emits_one_marker_per_entity(self):
        self.install(self.behaviour)
        text = " ".join(["word"] * 40) + " NAME1 " + " ".join(["word"] * 40)
        response = service.redact(self.Request(text))
        self.assertEqual(response["redacted"].count("[PRIVATE_PERSON]"), 1)
        self.assertNotIn("NAME1", response["redacted"])

    def test_empty_input_is_clean(self):
        self.install(self.behaviour)
        self.assertEqual(service.check(self.Request(""))["count"], 0)


class HealthTest(ServiceTestCase):
    def test_health_reports_budgets_without_any_input_text(self):
        self.install()
        health = service.health()
        self.assertTrue(health["ok"])
        self.assertTrue(health["cuda_available"])
        self.assertEqual(health["max_tokens"], 20)
        self.assertEqual(health["min_tokens"], 5)
        self.assertEqual(health["overlap_tokens"], 4)
        self.assertEqual(health["token_budgets"], [20, 10, 5])
        self.assertTrue(health["inference_serialized"])

    def test_health_before_load_reports_not_ok(self):
        health = service.health()
        self.assertFalse(health["ok"])
        self.assertIsNone(health["dtype"])


class BudgetWiringTest(ServiceTestCase):
    def test_max_budget_respects_the_tokenizer_context(self):
        service.state.classifier = FakeClassifier()
        service.state.classifier.tokenizer.model_max_length = 512
        # 512 - 2 special - 16 margin = 494, below the configured maximum.
        self.assertEqual(service.max_token_budget(), 494)

    def test_max_budget_respects_the_configured_cap(self):
        service.state.classifier = FakeClassifier()
        self.assertEqual(service.max_token_budget(), service.MAX_TOKENS)

    def test_nonsense_tokenizer_length_falls_back(self):
        service.state.classifier = FakeClassifier()
        service.state.classifier.tokenizer.model_max_length = 10**30
        self.assertEqual(
            service.max_token_budget(),
            min(service.FALLBACK_MAX_TOKENS - 18, service.MAX_TOKENS),
        )

    def test_slow_tokenizer_falls_back_to_character_offsets(self):
        service.state.classifier = FakeClassifier()
        service.state.classifier.tokenizer.is_fast = False
        text = "abcdef"
        self.assertEqual(service.token_offsets(text), [(i, i + 1) for i in range(6)])

    def test_plan_windows_matches_the_planner(self):
        service.state.classifier = FakeClassifier()
        text = " ".join(f"word{index}" for index in range(50))
        offsets = service.token_offsets(text)
        self.assertEqual(
            service.plan_windows(text, 10, 3), plan_chunks(text, offsets, 10, 3)
        )

    def test_oom_error_types_include_the_cuda_error(self):
        self.assertIn(FakeOutOfMemory, service.oom_error_types())

    def test_budget_policy_comes_from_the_environment_defaults(self):
        service.state.classifier = FakeClassifier()
        policy = service.budget_policy()
        self.assertEqual(policy.min_tokens, service.MIN_TOKENS)
        self.assertEqual(policy.overlap_tokens, service.OVERLAP_TOKENS)
        self.assertEqual(policy.shrink_factor, service.SHRINK_FACTOR)

    def test_scan_budget_exhausted_is_never_raised_raw_to_callers(self):
        self.assertTrue(issubclass(ScanBudgetExhausted, RuntimeError))


if __name__ == "__main__":
    unittest.main()
