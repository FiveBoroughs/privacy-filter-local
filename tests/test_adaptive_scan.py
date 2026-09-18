"""Tests for adaptive budgets, OOM retries, dedup and redaction.

scripts/adaptive_scan.py takes its planner, its model and its out-of-memory
exception types as arguments, so all of this runs on a plain interpreter: the
fake classifier below raises a fake OOM above a token threshold, standing in for
a CUDA card that ran out of VRAM mid-scan.

Every fixture here is synthetic.

Run: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from adaptive_scan import (  # noqa: E402
    AdaptiveScanner,
    BudgetExceedsContextLimit,
    BudgetPolicy,
    ContextLimitError,
    ContextLimitUnknown,
    Finding,
    ScanBudgetExhausted,
    findings_from_raw,
    merge_findings,
    redact_text,
    resolve_context_limit,
    window_budget,
)
from text_chunking import plan_chunks  # noqa: E402


def word_offsets(text: str) -> list[tuple[int, int]]:
    """One token per whitespace-delimited word."""
    offsets = []
    index = 0
    for word in text.split(" "):
        if word:
            offsets.append((index, index + len(word)))
        index += len(word) + 1
    return offsets


def plan(text: str, budget: int, overlap: int) -> list[tuple[int, int]]:
    return plan_chunks(text, word_offsets(text), budget, overlap)


class FakeOutOfMemory(Exception):
    """Stands in for torch.cuda.OutOfMemoryError."""


class FakeClassifier:
    """Finds ``NAME<digits>`` entities, and OOMs on windows above a size.

    ``oom_above`` is measured in words, mirroring the fake tokenizer used by the
    planner: any window with more words than that "does not fit in memory".
    """

    def __init__(self, oom_above: int = 10**9, entity: str = "NAME", delay: float = 0.0) -> None:
        self.oom_above = oom_above
        self.entity = entity
        self.delay = delay
        self.calls: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.lock = threading.Lock()

    def __call__(self, chunk: str) -> list[dict[str, object]]:
        with self.lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            # A real forward pass releases the GIL for milliseconds; without a
            # comparable pause here threads would never actually interleave and
            # the serialization test could not fail.
            time.sleep(self.delay)
            self.calls.append(chunk)
            if len(chunk.split()) > self.oom_above:
                raise FakeOutOfMemory(f"window of {len(chunk.split())} words does not fit")
            findings = []
            position = 0
            while True:
                start = chunk.find(self.entity, position)
                if start < 0:
                    break
                end = start + len(self.entity)
                while end < len(chunk) and chunk[end].isdigit():
                    end += 1
                findings.append(
                    {"entity_group": "private_person", "start": start, "end": end, "score": 0.9}
                )
                position = end
            return findings
        finally:
            with self.lock:
                self.concurrent -= 1


def build_scanner(classifier, **policy_kwargs) -> AdaptiveScanner:
    defaults = {"max_tokens": 20, "min_tokens": 5, "shrink_factor": 0.5, "overlap_tokens": 4}
    defaults.update(policy_kwargs)
    return AdaptiveScanner(
        plan_windows=plan,
        classify=classifier,
        policy=BudgetPolicy(**defaults),
        oom_errors=(FakeOutOfMemory,),
    )


def filler(count: int, offset: int = 0) -> str:
    return " ".join(f"word{index + offset}" for index in range(count))


class BudgetPolicyTest(unittest.TestCase):
    def test_ladder_descends_to_the_minimum(self):
        policy = BudgetPolicy(max_tokens=4096, min_tokens=256, shrink_factor=0.5)
        self.assertEqual(policy.ladder(), (4096, 2048, 1024, 512, 256))

    def test_ladder_is_a_single_step_when_min_equals_max(self):
        self.assertEqual(BudgetPolicy(max_tokens=256, min_tokens=256).ladder(), (256,))

    def test_minimum_above_maximum_is_clamped(self):
        policy = BudgetPolicy(max_tokens=128, min_tokens=4096)
        self.assertEqual(policy.min_tokens, 128)
        self.assertEqual(policy.ladder(), (128,))

    def test_ladder_terminates_with_a_lazy_shrink_factor(self):
        # A factor that rounds back to the same number must still make progress.
        policy = BudgetPolicy(max_tokens=8, min_tokens=6, shrink_factor=0.99)
        self.assertEqual(policy.ladder(), (8, 7, 6))

    def test_invalid_shrink_factor_is_rejected(self):
        with self.assertRaises(ValueError):
            BudgetPolicy(shrink_factor=1.0)


class ScanTest(unittest.TestCase):
    def test_empty_input_calls_no_model(self):
        classifier = FakeClassifier()
        self.assertEqual(build_scanner(classifier).scan(""), [])
        self.assertEqual(classifier.calls, [])

    def test_single_window_input_is_one_forward_pass(self):
        classifier = FakeClassifier()
        text = f"hello NAME42 and {filler(3)}"
        findings = build_scanner(classifier).scan(text)
        self.assertEqual(len(classifier.calls), 1)
        self.assertEqual([(f.label, f.start, f.end) for f in findings],
                         [("private_person", 6, 12)])

    def test_findings_keep_original_offsets_across_windows(self):
        text = f"{filler(60)} NAME7 {filler(60, offset=60)}"
        findings = build_scanner(FakeClassifier()).scan(text)
        self.assertEqual(len(findings), 1)
        start = text.index("NAME7")
        self.assertEqual((findings[0].start, findings[0].end), (start, start + len("NAME7")))
        self.assertEqual(findings[0].text, "NAME7")

    def test_entity_on_a_window_boundary_is_reported_once(self):
        # Place the entity at every offset in a long input: overlap must catch it
        # wherever the cut lands, and dedup must not report it twice.
        for position in range(0, 120, 7):
            text = f"{filler(position)} NAME9 {filler(120 - position, offset=position)}"
            findings = build_scanner(FakeClassifier()).scan(text)
            self.assertEqual(len(findings), 1, f"entity at word {position} reported {len(findings)}x")
            self.assertEqual(findings[0].text, "NAME9")

    def test_unrelated_nearby_entities_stay_separate(self):
        text = f"NAME1 {filler(4)} NAME2"
        findings = build_scanner(FakeClassifier()).scan(text)
        self.assertEqual([f.text for f in findings], ["NAME1", "NAME2"])


class AdaptiveRetryTest(unittest.TestCase):
    def test_oom_retries_the_failing_span_at_a_smaller_budget(self):
        # Windows of more than 8 words OOM, so the 20-token budget fails and the
        # 10-token retry (and then 5) has to carry the scan.
        classifier = FakeClassifier(oom_above=8)
        text = f"{filler(40)} NAME3 {filler(40, offset=40)}"
        findings = build_scanner(classifier).scan(text)
        self.assertEqual([f.text for f in findings], ["NAME3"])
        self.assertTrue(any(len(call.split()) <= 8 for call in classifier.calls))

    def test_successful_windows_are_not_rescanned_after_a_later_oom(self):
        # Only the tail is oversized: the first window succeeds at full budget
        # and must not be sent to the model a second time.
        text = f"{filler(20)} {'longword ' * 40}"

        class TailOnlyOom(FakeClassifier):
            def __call__(self, chunk):
                if "longword" in chunk and len(chunk.split()) > 6:
                    self.calls.append(chunk)
                    raise FakeOutOfMemory("tail window does not fit")
                return super().__call__(chunk)

        classifier = TailOnlyOom()
        build_scanner(classifier).scan(text)
        head = [call for call in classifier.calls if call.startswith("word0 ")]
        self.assertEqual(len(head), 1, "a window that succeeded was scanned again")

    def test_exhausted_budget_fails_closed_without_partial_findings(self):
        classifier = FakeClassifier(oom_above=0)
        text = f"NAME1 {filler(60)}"
        with self.assertRaises(ScanBudgetExhausted) as ctx:
            build_scanner(classifier).scan(text)
        self.assertEqual(ctx.exception.budget, 5)

    def test_oom_hook_reclaims_memory_before_retrying(self):
        classifier = FakeClassifier(oom_above=8)
        released = []
        scanner = AdaptiveScanner(
            plan_windows=plan,
            classify=classifier,
            policy=BudgetPolicy(max_tokens=20, min_tokens=5, overlap_tokens=4),
            oom_errors=(FakeOutOfMemory,),
            on_oom=lambda: released.append(True),
        )
        scanner.scan(filler(60))
        self.assertTrue(released)

    def test_failure_in_a_later_span_discards_earlier_findings(self):
        text = f"NAME1 {filler(20)} {'longword ' * 40}"

        class TailOnlyOom(FakeClassifier):
            def __call__(self, chunk):
                if "longword" in chunk:
                    raise FakeOutOfMemory("tail window never fits")
                return super().__call__(chunk)

        with self.assertRaises(ScanBudgetExhausted):
            build_scanner(TailOnlyOom()).scan(text)


class UnlockedGate:
    """A no-op stand-in for the inference lock, used as a negative control."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class SerializationTest(unittest.TestCase):
    TEXT = f"{filler(60)} NAME5 {filler(60, offset=60)}"

    def run_concurrently(self, scanner, count: int = 4) -> list[list[Finding]]:
        results: list[list[Finding]] = []
        threads = [
            threading.Thread(target=lambda: results.append(scanner.scan(self.TEXT)))
            for _ in range(count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results

    def test_concurrent_scans_never_overlap_in_the_model(self):
        classifier = FakeClassifier(delay=0.005)
        results = self.run_concurrently(build_scanner(classifier))
        self.assertEqual(classifier.max_concurrent, 1, "two requests ran inference at once")
        self.assertTrue(all(len(result) == 1 for result in results))

    def test_control_without_the_lock_does_overlap(self):
        # Guards the test above: if concurrent requests could not overlap here
        # either, passing it would prove nothing about the lock.
        classifier = FakeClassifier(delay=0.005)
        scanner = AdaptiveScanner(
            plan_windows=plan,
            classify=classifier,
            policy=BudgetPolicy(max_tokens=20, min_tokens=5, overlap_tokens=4),
            lock=UnlockedGate(),
        )
        self.run_concurrently(scanner)
        self.assertGreater(classifier.max_concurrent, 1)


class MergeTest(unittest.TestCase):
    def test_duplicate_findings_from_two_windows_collapse(self):
        text = "contact Ada Lovelace today"
        start = text.index("Ada")
        end = text.index(" today")
        duplicates = [
            Finding("private_person", start, end, text[start:end], 0.8),
            Finding("private_person", start, end, text[start:end], 0.95),
        ]
        merged = merge_findings(duplicates, text)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].score, 0.95)

    def test_partial_overlap_becomes_one_union_finding(self):
        text = "contact Ada Lovelace today"
        first = Finding("private_person", 8, 11, "Ada", 0.7)
        second = Finding("private_person", 8, 20, "Ada Lovelace", 0.9)
        merged = merge_findings([second, first], text)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (8, 20))
        self.assertEqual(merged[0].text, "Ada Lovelace")

    def test_separate_entities_are_not_merged(self):
        text = "Ada wrote and Grace compiled"
        findings = [
            Finding("private_person", 0, 3, "Ada", 0.9),
            Finding("private_person", 14, 19, "Grace", 0.9),
        ]
        self.assertEqual(len(merge_findings(findings, text)), 2)

    def test_different_labels_are_never_merged(self):
        text = "Ada ada@example.com"
        findings = [
            Finding("private_person", 0, 3, "Ada", 0.9),
            Finding("private_email", 4, 19, "ada@example.com", 0.9),
        ]
        self.assertEqual(len(merge_findings(findings, text)), 2)


class RedactTest(unittest.TestCase):
    def test_redaction_uses_deduplicated_original_offsets(self):
        classifier = FakeClassifier()
        text = f"{filler(60)} NAME8 {filler(60, offset=60)}"
        findings = build_scanner(classifier).scan(text)
        redacted = redact_text(text, findings)
        self.assertEqual(redacted.count("[PRIVATE_PERSON]"), 1)
        self.assertNotIn("NAME8", redacted)
        self.assertTrue(redacted.startswith("word0 "))
        self.assertTrue(redacted.endswith(f"word{119}"))

    def test_overlapping_labels_yield_one_marker(self):
        text = "Ada Lovelace"
        findings = [
            Finding("private_person", 0, 12, text, 0.9),
            Finding("private_url", 4, 12, "Lovelace", 0.4),
        ]
        self.assertEqual(redact_text(text, findings), "[PRIVATE_PERSON]")

    def test_multiple_entities_are_redacted_in_place(self):
        text = "Ada wrote to Grace"
        findings = [
            Finding("private_person", 0, 3, "Ada", 0.9),
            Finding("private_person", 13, 18, "Grace", 0.9),
        ]
        self.assertEqual(
            redact_text(text, findings), "[PRIVATE_PERSON] wrote to [PRIVATE_PERSON]"
        )


class FindingsFromRawTest(unittest.TestCase):
    def test_offsets_are_rebased_and_whitespace_trimmed(self):
        chunk = "  Ada  "
        findings = findings_from_raw(chunk, 100, [{"entity": "B-PRIVATE_PERSON", "start": 0, "end": 7, "score": 0.5}])
        self.assertEqual((findings[0].start, findings[0].end), (102, 105))
        self.assertEqual(findings[0].label, "private_person")
        self.assertEqual(findings[0].text, "Ada")

    def test_whitespace_only_findings_are_dropped(self):
        self.assertEqual(findings_from_raw("   ", 0, [{"entity": "X", "start": 0, "end": 3}]), [])

    def test_findings_without_offsets_are_dropped(self):
        self.assertEqual(findings_from_raw("Ada", 0, [{"entity": "X", "score": 0.9}]), [])


class Config:
    """A model config with attribute access, as transformers ships them."""

    def __init__(self, **values):
        self.__dict__.update(values)


class ContextLimitTest(unittest.TestCase):
    def test_config_limit_wins_over_a_tokenizer_advertising_more(self):
        # pplx-pii-masking: advertises 131072, truncates at 4096.
        config = Config(max_seq_len=4096, max_position_embeddings=131072)
        self.assertEqual(resolve_context_limit(config, 131072), 4096)

    def test_nested_encoder_limit_is_found(self):
        # gliner2: mdeberta encoder one level down, tokenizer says 1e30.
        config = Config(encoder_config={"max_position_embeddings": 512})
        self.assertEqual(resolve_context_limit(config, 10**30), 512)

    def test_tokenizer_may_lower_the_limit_but_never_establish_it(self):
        # openai/privacy-filter: 131072 architecture, 128000 advertised.
        config = Config(max_position_embeddings=131072)
        self.assertEqual(resolve_context_limit(config, 128000), 128000)
        with self.assertRaises(ContextLimitUnknown):
            resolve_context_limit(Config(), 128000)

    def test_a_declared_limit_overrides_an_unreadable_config(self):
        self.assertEqual(resolve_context_limit(Config(), None, declared=768), 768)
        with self.assertRaises(ContextLimitError):
            resolve_context_limit(Config(), None, declared=0)

    def test_an_explicit_budget_above_the_limit_is_refused(self):
        with self.assertRaises(BudgetExceedsContextLimit) as ctx:
            window_budget(8192, 512, explicit=True)
        self.assertEqual((ctx.exception.requested, ctx.exception.limit), (8192, 512))

    def test_a_default_budget_above_the_limit_is_clamped_to_it(self):
        self.assertEqual(window_budget(8192, 512, special_tokens=2, margin=16), 494)

    def test_a_budget_within_the_limit_keeps_headroom(self):
        self.assertEqual(window_budget(4096, 4096, special_tokens=2, margin=16), 4078)
        self.assertEqual(window_budget(1024, 4096, special_tokens=2, margin=16), 1024)


if __name__ == "__main__":
    unittest.main()
