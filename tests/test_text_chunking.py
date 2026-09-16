"""Regression tests for the oversized-input chunk planner.

These exercise scripts/text_chunking.py, which is deliberately dependency-free so
the planning logic can be verified without the GPU/torch container.

Run: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from text_chunking import clamp_overlap, plan_chunks, snap_to_whitespace  # noqa: E402


def subword_offsets(text: str) -> list[tuple[int, int]]:
    """One token per non-whitespace char; whitespace is not tokenized.

    Mimics a real subword tokenizer where spaces leave gaps in the offset map,
    which is exactly the structure plan_chunks snaps against.
    """
    return [(i, i + 1) for i, char in enumerate(text) if not char.isspace()]


def word_offsets(text: str) -> list[tuple[int, int]]:
    """One token per whitespace-delimited word, for entity-crossing tests."""
    offsets = []
    index = 0
    for word in text.split(" "):
        if word:
            offsets.append((index, index + len(word)))
        index += len(word) + 1
    return offsets


def tokens_in_span(offsets: list[tuple[int, int]], start: int, end: int) -> int:
    return sum(1 for token_start, _ in offsets if start <= token_start < end)


class PlanChunksTest(unittest.TestCase):
    def assert_valid_plan(self, text: str, offsets, budget: int, overlap: int = 0):
        spans = plan_chunks(text, offsets, budget, overlap)
        if not text:
            self.assertEqual(spans, [])
            return spans
        # The windows together cover the whole string, in order, with no gaps.
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))
        for (prev_start, prev_end), (next_start, next_end) in zip(spans, spans[1:]):
            self.assertLessEqual(next_start, prev_end, "gap between windows")
            self.assertGreater(next_start, prev_start, "window did not advance")
            self.assertGreater(next_end, prev_end, "window did not advance")
        # Every window stays within the token budget, overlap included.
        for start, end in spans:
            self.assertLessEqual(tokens_in_span(offsets, start, end), budget)
        return spans

    def test_empty_text_yields_no_spans(self):
        self.assertEqual(plan_chunks("", [], 8), [])

    def test_small_input_is_a_single_span(self):
        text = "alice@example.com lives here"
        offsets = subword_offsets(text)
        spans = self.assert_valid_plan(text, offsets, budget=10_000, overlap=16)
        self.assertEqual(spans, [(0, len(text))])

    def test_large_input_is_split_into_multiple_windows(self):
        # The original bug: one oversized request. The planner must break it up.
        text = " ".join(f"word{i}" for i in range(500))
        offsets = subword_offsets(text)
        spans = self.assert_valid_plan(text, offsets, budget=50, overlap=10)
        self.assertGreater(len(spans), 1)

    def test_cuts_land_on_whitespace_boundaries(self):
        text = " ".join(f"token{i}" for i in range(60))
        offsets = subword_offsets(text)
        spans = plan_chunks(text, offsets, budget=20)
        # Each boundary (except the final end) should sit right after whitespace,
        # so no word is sliced in half.
        for _, end in spans[:-1]:
            self.assertTrue(text[end - 1].isspace(), f"cut at {end} not on whitespace")

    def test_no_whitespace_still_makes_progress(self):
        # A giant unbroken token run (e.g. minified line) must not loop forever.
        text = "x" * 1000
        offsets = subword_offsets(text)
        spans = self.assert_valid_plan(text, offsets, budget=100, overlap=20)
        self.assertGreater(len(spans), 1)

    def test_char_fallback_windows_stay_within_budget(self):
        # The non-fast-tokenizer fallback passes one offset per character.
        text = "the quick brown fox jumps over the lazy dog " * 5
        offsets = [(i, i + 1) for i in range(len(text))]
        spans = self.assert_valid_plan(text, offsets, budget=30, overlap=8)
        self.assertGreater(len(spans), 1)

    def test_zero_overlap_windows_are_contiguous(self):
        text = " ".join(f"word{i}" for i in range(200))
        offsets = subword_offsets(text)
        spans = self.assert_valid_plan(text, offsets, budget=40, overlap=0)
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            self.assertEqual(prev_end, next_start)

    def test_every_boundary_region_appears_in_both_neighbours(self):
        text = " ".join(f"word{i}" for i in range(400))
        offsets = word_offsets(text)
        overlap = 5
        spans = self.assert_valid_plan(text, offsets, budget=40, overlap=overlap)
        self.assertGreater(len(spans), 1)
        for (prev_start, prev_end), (next_start, _) in zip(spans, spans[1:]):
            self.assertLess(next_start, prev_end, "no overlap between windows")
            self.assertGreaterEqual(next_start, prev_start)
            shared = tokens_in_span(offsets, next_start, prev_end)
            self.assertGreaterEqual(shared, overlap, "boundary region too small")

    def test_entity_crossing_a_cut_is_whole_in_some_window(self):
        # A four-token entity is planted at every position; with overlap, at
        # least one window must contain it intact.
        entity = "Jean Baptiste Emmanuel Zorg"
        filler = " ".join(f"word{i}" for i in range(300))
        for position in range(0, len(filler), 137):
            text = filler[:position] + " " + entity + " " + filler[position:]
            spans = plan_chunks(text, word_offsets(text), budget=20, overlap=6)
            start = text.index(entity)
            end = start + len(entity)
            self.assertTrue(
                any(span_start <= start and end <= span_end for span_start, span_end in spans),
                f"entity at {start} was split across every window",
            )

    def test_overlap_is_capped_at_half_the_budget(self):
        # An overlap larger than the budget would stall the stride; it is clamped.
        self.assertEqual(clamp_overlap(100, 400), 50)
        self.assertEqual(clamp_overlap(100, 10), 10)
        self.assertEqual(clamp_overlap(100, -5), 0)
        text = " ".join(f"word{i}" for i in range(300))
        offsets = subword_offsets(text)
        self.assert_valid_plan(text, offsets, budget=30, overlap=1000)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            plan_chunks("abc", subword_offsets("abc"), 0)


class SnapToWhitespaceTest(unittest.TestCase):
    def test_snaps_back_to_after_last_space(self):
        text = "hello world here"
        # window [0, 14) -> "hello world he"; last space is index 11.
        self.assertEqual(snap_to_whitespace(text, 0, 14), 12)

    def test_returns_hard_boundary_without_whitespace(self):
        text = "abcdefgh"
        self.assertEqual(snap_to_whitespace(text, 0, 5), 5)


if __name__ == "__main__":
    unittest.main()
