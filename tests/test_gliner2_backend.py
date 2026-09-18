"""Pure contract tests for the GLiNER2 adapter; no model or GPU required."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from gliner2_backend import (  # noqa: E402
    GLINER2_LABELS,
    Gliner2Tokenizer,
    findings_from_result,
    schema_token_overhead,
)


class FakeTokenizer:
    is_fast = True
    model_max_length = 10**30

    def __call__(self, text, **kwargs):
        return {"text": text, "kwargs": kwargs}


class TokenizerTest(unittest.TestCase):
    def test_schema_tokens_are_reserved_from_every_text_window(self):
        tokenizer = Gliner2Tokenizer(FakeTokenizer(), schema_tokens=137)
        self.assertEqual(tokenizer.num_special_tokens_to_add(), 137)
        self.assertTrue(tokenizer.is_fast)
        self.assertEqual(tokenizer("word", return_offsets_mapping=True)["text"], "word")

    def test_schema_overhead_uses_the_same_mapping_shape_as_inference(self):
        class Scalar:
            def sum(self):
                return self

            def item(self):
                return 137

        class Processor:
            def collate_fn_inference(self, records, **kwargs):
                _, schema = records[0]
                self.schema = schema
                return type("Batch", (), {"attention_mask": [Scalar()]})()

        extractor = type(
            "Extractor",
            (),
            {"processor": Processor(), "architecture": "span"},
        )()
        self.assertEqual(schema_token_overhead(extractor), 137)
        self.assertIsInstance(extractor.processor.schema["entities"], dict)
        self.assertEqual(tuple(extractor.processor.schema["entities"]), GLINER2_LABELS)


class FindingAdapterTest(unittest.TestCase):
    def test_every_requested_label_has_a_canonical_mapping(self):
        self.assertEqual(len(GLINER2_LABELS), 42)

    def test_repeated_values_keep_their_distinct_original_offsets(self):
        text = "Mira Voss reviewed the form; Mira Voss signed it."
        second = text.rindex("Mira Voss")
        result = {
            "entities": {
                "full_name": [
                    {"text": "Mira Voss", "start": 0, "end": 9, "confidence": 0.91},
                    {
                        "text": "Mira Voss",
                        "start": second,
                        "end": second + 9,
                        "confidence": 0.89,
                    },
                ]
            }
        }
        findings = findings_from_result(text, result)
        self.assertEqual([item["start"] for item in findings], [0, second])
        self.assertEqual({item["entity_group"] for item in findings}, {"private_person"})

    def test_duplicate_first_occurrence_offsets_fail_closed(self):
        text = "Mira Voss reviewed the form; Mira Voss signed it."
        result = {
            "entities": {
                "full_name": [
                    {"text": "Mira Voss", "start": 0, "end": 9},
                    {"text": "Mira Voss", "start": 0, "end": 9},
                ]
            }
        }
        with self.assertRaisesRegex(RuntimeError, "duplicate character offsets"):
            findings_from_result(text, result)

    def test_a_span_that_does_not_slice_to_the_reported_value_fails_closed(self):
        text = "Mira Voss reviewed the form."
        result = {
            "entities": {
                "full_name": [
                    {"text": "Mira Voss", "start": 1, "end": 10, "confidence": 0.9}
                ]
            }
        }
        with self.assertRaisesRegex(RuntimeError, "invalid character offsets"):
            findings_from_result(text, result)

    def test_fine_grained_secret_and_address_labels_stay_stable_for_callers(self):
        text = "token-1 at 8 Kestrel Way"
        result = {
            "entities": {
                "access_token": [
                    {"text": "token-1", "start": 0, "end": 7, "confidence": 0.8}
                ],
                "street_address": [
                    {"text": "8 Kestrel Way", "start": 11, "end": 24, "confidence": 0.7}
                ],
            }
        }
        findings = findings_from_result(text, result)
        self.assertEqual([item["entity_group"] for item in findings], ["secret", "private_address"])


if __name__ == "__main__":
    unittest.main()
