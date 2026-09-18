"""Pure contract tests for the PPLX adapter; no model or GPU required."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from pplx_backend import (  # noqa: E402
    DEFAULT_PPLX_MODEL,
    PPLX_LABEL_MAP,
    PPLX_REVISION,
    findings_from_spans,
)


class FindingAdapterTest(unittest.TestCase):
    def span(self, label, start, end, score=0.9):
        return SimpleNamespace(label=label, start=start, end=end, score=score)

    def test_model_and_revision_are_immutable_identifiers(self):
        self.assertEqual(DEFAULT_PPLX_MODEL, "perplexity-ai/pplx-pii-masking")
        self.assertEqual(len(PPLX_REVISION), 40)

    def test_every_source_label_has_a_canonical_mapping(self):
        self.assertEqual(len(PPLX_LABEL_MAP), 9)
        self.assertEqual(PPLX_LABEL_MAP["other_pii"], "secret")

    def test_repeated_values_keep_their_distinct_original_offsets(self):
        text = "Mira Voss reviewed the form; Mira Voss signed it."
        second = text.rindex("Mira Voss")
        spans = [
            self.span("private_person", 0, 9, 0.91),
            self.span("private_person", second, second + 9, 0.89),
        ]
        findings = findings_from_spans(text, spans)
        self.assertEqual([item["start"] for item in findings], [0, second])
        self.assertEqual({item["entity_group"] for item in findings}, {"private_person"})

    def test_invalid_offsets_fail_closed_without_echoing_text(self):
        text = "Mira Voss reviewed the form."
        with self.assertRaises(RuntimeError) as caught:
            findings_from_spans(text, [self.span("private_person", -1, 9)])
        self.assertIn("invalid character offsets", str(caught.exception))
        self.assertNotIn("Mira Voss", str(caught.exception))

    def test_duplicate_offsets_fail_closed(self):
        text = "Mira Voss reviewed the form."
        span = self.span("private_person", 0, 9)
        with self.assertRaisesRegex(RuntimeError, "duplicate character offsets"):
            findings_from_spans(text, [span, span])

    def test_invalid_scores_fail_closed(self):
        for score in (None, float("nan"), "not-a-score"):
            with self.subTest(score=score):
                with self.assertRaisesRegex(RuntimeError, "invalid score"):
                    findings_from_spans(
                        "value",
                        [self.span("secret", 0, 5, score=score)],
                    )

    def test_unmapped_labels_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "unmapped label"):
            findings_from_spans("value", [self.span("unknown", 0, 5)])


if __name__ == "__main__":
    unittest.main()
