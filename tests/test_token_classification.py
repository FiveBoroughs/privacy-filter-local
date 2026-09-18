"""Behavioral tests for the dependency-free token-classification decoder."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from token_classification import (  # noqa: E402
    cuda_safe_export,
    is_onnx_oom,
    label_tag,
    simple_groups,
)


def prediction(entity, start, end, score=0.9):
    return {"entity": entity, "start": start, "end": end, "score": score}


class SimpleGroupsTest(unittest.TestCase):
    def test_b_and_i_tokens_form_one_span_with_their_mean_score(self):
        groups = simple_groups(
            [
                prediction("B-private_person", 0, 3, 0.8),
                prediction("I-private_person", 3, 7, 1.0),
            ]
        )
        self.assertEqual(
            groups,
            [
                {
                    "entity_group": "private_person",
                    "start": 0,
                    "end": 7,
                    "score": 0.9,
                }
            ],
        )

    def test_a_second_b_starts_a_new_entity_of_the_same_type(self):
        groups = simple_groups(
            [
                prediction("B-private_person", 0, 3),
                prediction("B-private_person", 4, 7),
            ]
        )
        self.assertEqual([(item["start"], item["end"]) for item in groups], [(0, 3), (4, 7)])

    def test_o_tokens_break_entities_and_are_filtered_after_grouping(self):
        groups = simple_groups(
            [
                prediction("B-private_email", 0, 4),
                prediction("O", 4, 5),
                prediction("I-private_email", 5, 9),
            ]
        )
        self.assertEqual([(item["start"], item["end"]) for item in groups], [(0, 4), (5, 9)])

    def test_e_and_s_labels_follow_transformers_simple_semantics(self):
        # TokenClassificationPipeline only knows B-/I-. E-/S- are continuation
        # labels with distinct tags, so they form their own groups. The scanner
        # later normalizes and merges adjacent spans of the same entity type.
        groups = simple_groups(
            [
                prediction("B-private_phone", 0, 2),
                prediction("I-private_phone", 2, 4),
                prediction("E-private_phone", 4, 6),
                prediction("S-private_phone", 7, 9),
            ]
        )
        self.assertEqual(
            [(item["entity_group"], item["start"], item["end"]) for item in groups],
            [
                ("private_phone", 0, 4),
                ("private_phone", 4, 6),
                ("private_phone", 7, 9),
            ],
        )
        self.assertEqual(label_tag("E-private_phone"), ("I", "E-private_phone"))


class OnnxSafetyTest(unittest.TestCase):
    def test_q4_exports_are_not_cuda_safe(self):
        self.assertFalse(cuda_safe_export("onnx/model_q4.onnx"))
        self.assertFalse(cuda_safe_export("onnx/model_q4f16.onnx"))
        self.assertTrue(cuda_safe_export("onnx/model_quantized.onnx"))
        self.assertTrue(cuda_safe_export("onnx/model_fp16.onnx"))

    def test_only_explicit_allocator_failures_are_ooms(self):
        self.assertTrue(is_onnx_oom("BFCArena Failed to allocate memory for buffer"))
        self.assertTrue(is_onnx_oom("CUDA failure 2: out of memory"))
        self.assertFalse(is_onnx_oom("Invalid graph input shape"))
        self.assertFalse(is_onnx_oom("Unsupported operator"))


if __name__ == "__main__":
    unittest.main()
