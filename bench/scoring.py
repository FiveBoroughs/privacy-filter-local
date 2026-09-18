"""Entity-level scoring for the benchmark. Pure: no I/O, no model, no network.

Counting a finding as a hit because it overlaps the right text flatters every
model equally, which is how the first version of this benchmark ended up
reporting 30/32 for three different exports. Two established schemes fix that,
and this module implements both:

MUC-5 / SemEval entity categories, as used by ``nervaluate``: each annotated
entity is correct, incorrect (matched but wrong label), partial (overlapping
boundaries), or missed, and each unmatched prediction is spurious. Precision and
recall follow from those counts, with partial matches taking half credit in the
partial strategy and none in the strict one. Boundary quality stops being
invisible.

The Text Anonymization Benchmark's privacy recall (Pilan et al., Computational
Linguistics 48(4), 2022): an entity counts as protected only when *every* one of
its mentions is masked, and direct identifiers are scored separately from
quasi-identifiers. TAB's own numbers are the argument for it -- Presidio scored
0.707 mention-level recall on their corpus but 0.46 entity recall on direct
identifiers, because frequent quasi-identifiers hid the leaked names. For a hook
whose job is to stop a name reaching a commit, a name masked in two places out
of three is a leak, and mention-level recall is the wrong instrument to see it.

This deviates from ``nervaluate`` in one place, deliberately: its partial
strategy ignores entity type, while here a partial match must still carry the
right canonical label, and an overlapping span with the wrong label is counted
incorrect rather than partial. Mislabelling is not free for us -- the label is
what the hook prints and what a reviewed-findings entry is keyed against.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

# The default model's own vocabulary is the canonical one: it is the coarsest,
# and every other model in the benchmark can be folded into it. A model with a
# finer taxonomy answering `city` where this corpus says `private_address` has
# found the same text and should not be scored as a miss.
CANONICAL_LABELS = (
    "private_person",
    "private_email",
    "private_phone",
    "private_address",
    "private_date",
    "account_number",
    "private_url",
    "secret",
)

# Anything a model may emit, mapped onto the canonical set. Labels absent here
# are reported as unmapped rather than dropped: a silently ignored label would
# turn a model's finding into a phantom miss and understate its recall.
LABEL_MAP = {label: label for label in CANONICAL_LABELS}
LABEL_MAP.update(
    {
        # OpenMed/privacy-filter-nemotron's 55-way vocabulary.
        "first_name": "private_person",
        "last_name": "private_person",
        "user_name": "private_person",
        "email": "private_email",
        "phone_number": "private_phone",
        "fax_number": "private_phone",
        "street_address": "private_address",
        "city": "private_address",
        "county": "private_address",
        "state": "private_address",
        "country": "private_address",
        "postcode": "private_address",
        "coordinate": "private_address",
        "date": "private_date",
        "date_of_birth": "private_date",
        "date_time": "private_date",
        "time": "private_date",
        "bank_routing_number": "account_number",
        "credit_debit_card": "account_number",
        "swift_bic": "account_number",
        "tax_id": "account_number",
        "ssn": "account_number",
        "national_id": "account_number",
        "medical_record_number": "account_number",
        "health_plan_beneficiary_number": "account_number",
        "customer_id": "account_number",
        "employee_id": "account_number",
        "certificate_license_number": "account_number",
        "api_key": "secret",
        "password": "secret",
        "pin": "secret",
        "cvv": "secret",
        "http_cookie": "secret",
        "unique_id": "secret",
        "device_identifier": "secret",
        "url": "private_url",
        "ipv4": "private_url",
        "ipv6": "private_url",
        "mac_address": "private_url",
    }
)

# TAB's split. A direct identifier names the person on its own; a quasi-identifier
# only re-identifies them in combination with something else. Missing one of each
# is not the same failure, so they are never averaged together here.
DIRECT_IDENTIFIERS = frozenset(
    {"private_person", "private_email", "private_phone", "account_number", "secret", "private_url"}
)
QUASI_IDENTIFIERS = frozenset({"private_address", "private_date"})


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    # Mentions sharing a key are the same person or thing, which is what makes
    # "every mention masked" answerable.
    entity: str = ""

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class Counts:
    """MUC-5 categories, accumulated over records."""

    correct: int = 0
    incorrect: int = 0
    partial: int = 0
    missed: int = 0
    spurious: int = 0

    def add(self, other: Counts) -> None:
        self.correct += other.correct
        self.incorrect += other.incorrect
        self.partial += other.partial
        self.missed += other.missed
        self.spurious += other.spurious

    @property
    def possible(self) -> int:
        """Annotated entities: everything that could have been found."""
        return self.correct + self.incorrect + self.partial + self.missed

    @property
    def actual(self) -> int:
        """Predictions made: everything that could have been wrong."""
        return self.correct + self.incorrect + self.partial + self.spurious

    def scores(self, strict: bool) -> dict[str, float]:
        credit = self.correct if strict else self.correct + 0.5 * self.partial
        precision = credit / self.actual if self.actual else 0.0
        recall = credit / self.possible if self.possible else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}


def canonical(label: str) -> str | None:
    """The canonical label for a model's own, or None when it is unmapped."""
    return LABEL_MAP.get(label)

def merge_predictions(predicted: Sequence[Span], text: str) -> list[Span]:
    """Merge adjacent canonical spans when only separators lie between them.

    Models with fine taxonomies emit first and last names, or street and city,
    as separate labels. Once both labels map to one canonical type, Presidio's
    span evaluator merges adjacent same-type spans across punctuation and
    whitespace. Without this step a finer vocabulary is penalised merely for
    being finer.

    Alphanumeric text in the gap prevents a merge: predictions for `.invalid`
    on either side of an exposed email local-part must remain fragments.
    """
    merged: list[Span] = []
    for span in sorted(predicted, key=lambda item: (item.start, item.end)):
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        gap = text[previous.end : span.start]
        if previous.label == span.label and not any(character.isalnum() for character in gap):
            merged[-1] = Span(
                previous.start, max(previous.end, span.end), previous.label, previous.entity
            )
        else:
            merged.append(span)
    return merged


def match(gold: Sequence[Span], predicted: Sequence[Span]) -> tuple[Counts, dict[str, Counts]]:
    """Categorise every annotated entity and every prediction in one record.

    Each prediction is consumed by at most one annotation, exact boundaries
    first: without that, one sprawling prediction covering three annotations
    would score as three hits and never be charged for its boundaries.
    """
    overall = Counts()
    by_label: dict[str, Counts] = {}
    taken: set[int] = set()

    def bucket(label: str) -> Counts:
        return by_label.setdefault(label, Counts())

    for annotation in gold:
        candidates = [
            index
            for index, prediction in enumerate(predicted)
            if index not in taken and prediction.overlaps(annotation)
        ]
        # Exact boundaries win, then the largest overlap, so a near-miss is not
        # chosen over the prediction that actually got the span right.
        candidates.sort(
            key=lambda index: (
                predicted[index].start == annotation.start
                and predicted[index].end == annotation.end,
                min(predicted[index].end, annotation.end)
                - max(predicted[index].start, annotation.start),
            ),
            reverse=True,
        )
        if not candidates:
            overall.missed += 1
            bucket(annotation.label).missed += 1
            continue

        chosen = candidates[0]
        taken.add(chosen)
        prediction = predicted[chosen]
        exact = prediction.start == annotation.start and prediction.end == annotation.end
        same_label = prediction.label == annotation.label
        if exact and same_label:
            overall.correct += 1
            bucket(annotation.label).correct += 1
        elif same_label:
            overall.partial += 1
            bucket(annotation.label).partial += 1
        else:
            overall.incorrect += 1
            bucket(annotation.label).incorrect += 1

    for index, prediction in enumerate(predicted):
        if index not in taken:
            overall.spurious += 1
            bucket(prediction.label).spurious += 1

    return overall, by_label


def masked(spans: Iterable[Span]) -> set[int]:
    """Every character offset the predictions cover."""
    covered: set[int] = set()
    for span in spans:
        covered.update(range(span.start, span.end))
    return covered


@dataclass
class EntityRecall:
    """TAB-style protection: an entity is safe only if no mention survives."""

    protected: int = 0
    exposed: int = 0

    @property
    def total(self) -> int:
        return self.protected + self.exposed

    @property
    def recall(self) -> float:
        return self.protected / self.total if self.total else 0.0


def entity_recall(
    gold: Sequence[Span],
    predicted: Sequence[Span],
    labels: frozenset[str],
    text: str | None = None,
) -> EntityRecall:
    """Fraction of annotated entities whose every mention is fully masked.

    Label agreement is irrelevant here: redaction replaces the text either way,
    so a person found and called an address is still not leaked. What is not
    tolerated is a second mention left in the clear.
    """
    result = EntityRecall()
    covered = masked(predicted)
    mentions: dict[str, list[Span]] = {}
    for annotation in gold:
        if annotation.label in labels:
            mentions.setdefault(annotation.entity, []).append(annotation)
    for spans in mentions.values():
        required = (
            offset
            for span in spans
            for offset in range(span.start, span.end)
            if text is None or text[offset].isalnum()
        )
        if all(offset in covered for offset in required):
            result.protected += 1
        else:
            result.exposed += 1
    return result


@dataclass
class Report:
    counts: Counts = field(default_factory=Counts)
    by_label: dict[str, Counts] = field(default_factory=dict)
    direct: EntityRecall = field(default_factory=EntityRecall)
    quasi: EntityRecall = field(default_factory=EntityRecall)
    # Predictions on records annotated as containing nothing. Precision over the
    # whole corpus dilutes these; a hook that cries wolf on clean code teaches
    # people not to trust it, so false alarms are counted on their own.
    false_alarms: int = 0
    clean_records: int = 0
    clean_records_flagged: int = 0
    unmapped: dict[str, int] = field(default_factory=dict)

    def add_record(
        self, gold: Sequence[Span], predicted: Sequence[Span], text: str | None = None
    ) -> None:
        overall, by_label = match(gold, predicted)
        self.counts.add(overall)
        for label, counts in by_label.items():
            self.by_label.setdefault(label, Counts()).add(counts)
        for store, labels in ((self.direct, DIRECT_IDENTIFIERS), (self.quasi, QUASI_IDENTIFIERS)):
            recall = entity_recall(gold, predicted, labels, text)
            store.protected += recall.protected
            store.exposed += recall.exposed
        if not gold:
            self.clean_records += 1
            self.false_alarms += len(predicted)
            if predicted:
                self.clean_records_flagged += 1

    def note_unmapped(self, label: str) -> None:
        self.unmapped[label] = self.unmapped.get(label, 0) + 1

    def summary(self) -> dict[str, object]:
        return {
            "strict": self.counts.scores(strict=True),
            "partial": self.counts.scores(strict=False),
            "counts": {
                "correct": self.counts.correct,
                "incorrect": self.counts.incorrect,
                "partial": self.counts.partial,
                "missed": self.counts.missed,
                "spurious": self.counts.spurious,
            },
            "entity_recall": {
                "direct": {
                    "recall": self.direct.recall,
                    "protected": self.direct.protected,
                    "total": self.direct.total,
                },
                "quasi": {
                    "recall": self.quasi.recall,
                    "protected": self.quasi.protected,
                    "total": self.quasi.total,
                },
            },
            "false_alarms": self.false_alarms,
            "clean_records": self.clean_records,
            "clean_records_flagged": self.clean_records_flagged,
            "per_label": {
                label: {
                    **counts.scores(strict=True),
                    "support": counts.possible,
                    "missed": counts.missed,
                    "spurious": counts.spurious,
                }
                for label, counts in sorted(self.by_label.items())
            },
            "unmapped_labels": dict(sorted(self.unmapped.items())),
        }
