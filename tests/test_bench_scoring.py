"""Tests for the benchmark's entity scoring.

The first version of the benchmark counted a finding as a hit whenever it
overlapped an annotation anywhere, and duly reported an identical score for
three different exports. These tests pin the distinctions that replaced it:
exact boundaries versus partial ones, a mislabelled span versus a missed one,
and -- the one that matters for a redaction tool -- an entity whose second
mention was left in the clear.

Every fixture here is synthetic.

Run: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

from scoring import (  # noqa: E402
    DIRECT_IDENTIFIERS,
    QUASI_IDENTIFIERS,
    Report,
    Span,
    canonical,
    entity_recall,
    merge_predictions,
    match,
)


def person(start, end, entity="a"):
    return Span(start, end, "private_person", entity)


class MatchTest(unittest.TestCase):
    def test_exact_span_with_the_right_label_is_correct(self):
        counts, _ = match([person(0, 10)], [person(0, 10)])
        self.assertEqual((counts.correct, counts.partial, counts.incorrect), (1, 0, 0))

    def test_right_span_with_the_wrong_label_is_incorrect_not_missed(self):
        # The text was found; only the name for it is wrong. Scoring it as a
        # miss would claim a leak that did not happen.
        counts, _ = match([person(0, 10)], [Span(0, 10, "private_address", "a")])
        self.assertEqual((counts.incorrect, counts.missed, counts.spurious), (1, 0, 0))

    def test_overlapping_boundaries_are_partial(self):
        counts, _ = match([person(4, 16)], [person(0, 16)])
        self.assertEqual((counts.partial, counts.correct), (1, 0))

    def test_partial_matches_are_half_credit_only_in_the_partial_strategy(self):
        counts, _ = match([person(4, 16)], [person(0, 16)])
        self.assertEqual(counts.scores(strict=True)["recall"], 0.0)
        self.assertEqual(counts.scores(strict=False)["recall"], 0.5)

    def test_an_annotation_with_no_prediction_is_missed(self):
        counts, _ = match([person(0, 10)], [])
        self.assertEqual((counts.missed, counts.spurious), (1, 0))

    def test_a_prediction_with_no_annotation_is_spurious(self):
        counts, _ = match([], [person(0, 10)])
        self.assertEqual((counts.spurious, counts.missed), (1, 0))

    def test_one_sprawling_prediction_cannot_satisfy_two_annotations(self):
        # Without consuming the prediction, a single span covering the whole
        # line would score as two hits and never be charged for its boundaries.
        counts, _ = match([person(0, 5, "a"), person(10, 15, "b")], [person(0, 15, "c")])
        self.assertEqual((counts.partial, counts.missed), (1, 1))

    def test_an_exact_prediction_is_preferred_over_a_sloppier_overlapping_one(self):
        counts, _ = match([person(10, 20)], [person(0, 30), person(10, 20)])
        self.assertEqual((counts.correct, counts.spurious), (1, 1))

    def test_per_label_counts_are_keyed_on_the_annotation_not_the_prediction(self):
        _, by_label = match([person(0, 10)], [Span(0, 10, "secret", "a")])
        self.assertEqual(by_label["private_person"].incorrect, 1)


class MergePredictionsTest(unittest.TestCase):
    def test_canonical_name_parts_merge_across_separators(self):
        text = "Marcus Oyelaran"
        merged = merge_predictions([person(0, 6), person(7, 15)], text)
        self.assertEqual(merged, [person(0, 15)])

    def test_exposed_alphanumeric_text_prevents_a_merge(self):
        text = ".invalid exposed .invalid"
        spans = [
            Span(0, 8, "private_email"),
            Span(17, 25, "private_email"),
        ]
        self.assertEqual(merge_predictions(spans, text), spans)


class EntityRecallTest(unittest.TestCase):
    """TAB's rule: an entity is protected only when every mention is masked."""

    def test_an_entity_whose_second_mention_survives_is_exposed(self):
        gold = [person(0, 5, "marcus"), person(40, 45, "marcus")]
        result = entity_recall(gold, [person(0, 5, "marcus")], DIRECT_IDENTIFIERS)
        self.assertEqual((result.protected, result.exposed), (0, 1))
        self.assertEqual(result.recall, 0.0)

    def test_every_mention_masked_protects_the_entity(self):
        gold = [person(0, 5, "marcus"), person(40, 45, "marcus")]
        result = entity_recall(gold, gold, DIRECT_IDENTIFIERS)
        self.assertEqual((result.protected, result.exposed), (1, 0))

    def test_a_partly_covered_mention_does_not_count_as_masked(self):
        # Redaction replaces the matched span. Half a surname left in the text
        # is still the surname.
        result = entity_recall([person(0, 10)], [person(0, 6)], DIRECT_IDENTIFIERS)
        self.assertEqual(result.exposed, 1)

    def test_masking_under_the_wrong_label_still_protects(self):
        # The text is replaced either way, so this is a labelling error, not a leak.
        result = entity_recall(
            [person(0, 10)], [Span(0, 10, "private_address", "x")], DIRECT_IDENTIFIERS
        )
        self.assertEqual(result.protected, 1)

    def test_only_separator_characters_may_remain_between_masked_name_parts(self):
        text = "Marcus Oyelaran"
        gold = [person(0, len(text), "marcus")]
        predicted = [person(0, 6), person(7, len(text))]
        result = entity_recall(gold, predicted, DIRECT_IDENTIFIERS, text)
        self.assertEqual(result.protected, 1)

    def test_direct_and_quasi_identifiers_are_never_pooled(self):
        gold = [person(0, 5, "marcus"), Span(20, 30, "private_date", "when")]
        report = Report()
        report.add_record(gold, [person(0, 5, "marcus")])
        self.assertEqual(report.direct.recall, 1.0)
        self.assertEqual(report.quasi.recall, 0.0)


class ReportTest(unittest.TestCase):
    def test_false_alarms_are_counted_only_against_clean_records(self):
        # A false positive on a record that does contain PII is already charged
        # as spurious; the separate count is about crying wolf on clean code.
        report = Report()
        report.add_record([], [person(0, 4), person(9, 12)])
        report.add_record([person(0, 5)], [person(0, 5), person(30, 33)])
        self.assertEqual(report.false_alarms, 2)
        self.assertEqual((report.clean_records, report.clean_records_flagged), (1, 1))
        self.assertEqual(report.counts.spurious, 3)

    def test_a_clean_record_left_alone_is_not_flagged(self):
        report = Report()
        report.add_record([], [])
        self.assertEqual((report.clean_records, report.clean_records_flagged), (1, 0))

    def test_summary_reports_both_strategies_and_both_identifier_classes(self):
        report = Report()
        report.add_record([person(0, 5, "a")], [person(0, 7, "a")])
        summary = report.summary()
        self.assertEqual(summary["strict"]["f1"], 0.0)
        self.assertEqual(summary["partial"]["f1"], 0.5)
        self.assertEqual(summary["entity_recall"]["direct"]["total"], 1)
        self.assertEqual(summary["entity_recall"]["quasi"]["total"], 0)


class CanonicalLabelTest(unittest.TestCase):
    def test_a_finer_vocabulary_folds_into_the_canonical_one(self):
        self.assertEqual(canonical("first_name"), "private_person")
        self.assertEqual(canonical("street_address"), "private_address")
        self.assertEqual(canonical("api_key"), "secret")

    def test_canonical_labels_map_to_themselves(self):
        self.assertEqual(canonical("private_person"), "private_person")

    def test_an_unknown_label_is_reported_rather_than_silently_dropped(self):
        self.assertIsNone(canonical("blood_type"))

    def test_every_identifier_class_uses_canonical_labels(self):
        for label in DIRECT_IDENTIFIERS | QUASI_IDENTIFIERS:
            self.assertEqual(canonical(label), label)


if __name__ == "__main__":
    unittest.main()
