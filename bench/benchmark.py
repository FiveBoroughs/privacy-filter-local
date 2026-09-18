#!/usr/bin/env python3
"""Measure model/backend variants against the synthetic corpus in corpus.json.

Every number in the README's benchmark table comes from this script, so the
table can be re-derived on other hardware instead of being trusted. It restarts
the real service once per variant, scans the corpus over HTTP exactly as a
caller would, and scores the findings with bench/scoring.py: MUC-5 entity
categories for detection quality and the Text Anonymization Benchmark's
entity-level recall for whether an identifier was actually protected.

Aggregate scores over a corpus this size cannot resolve small differences
between two exports of the same weights. ``--baseline`` exists for that: it
diffs one variant's findings against another's span by span, which is the
sensitive instrument here.

Standard library only, like the rest of the host-side tooling: this runs on the
host next to the CLI, not inside the container.

Run: bench/benchmark.py                                  # every variant
     bench/benchmark.py --baseline onnx-cpu-int8         # plus a findings diff
     bench/benchmark.py --variant onnx-cuda-int8 --json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring import Report, Span, canonical, merge_predictions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(__file__).resolve().parent / "corpus.json"
BASE_URL = os.environ.get("PRIVACY_FILTER_URL", "http://127.0.0.1:8757")
CONTAINER = os.environ.get("PRIVACY_FILTER_CONTAINER", "privacy-filter-local")
DEFAULT_MODEL = "openai/privacy-filter"
NEMOTRON = "OpenMed/privacy-filter-nemotron"

# Both models declare a 131072-token context, so a variant is only ever
# distinguished by its runtime, its export and its weights -- never by a
# different window size. The window is pinned so latency stays comparable and
# the cold-start VRAM selection in start-service cannot change it underneath a
# measurement.
WINDOW_TOKENS = "4096"


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str]
    note: str = ""


VARIANTS = (
    Variant(
        "torch-cuda-bf16",
        {"PRIVACY_FILTER_BACKEND": "torch", "PRIVACY_FILTER_MODEL": DEFAULT_MODEL},
        "default model, native checkpoint",
    ),
    Variant(
        "onnx-cuda-int8",
        {
            "PRIVACY_FILTER_BACKEND": "onnx",
            "PRIVACY_FILTER_ONNX_PROVIDER": "cuda",
            "PRIVACY_FILTER_ONNX_FILE": "onnx/model_quantized.onnx",
            "PRIVACY_FILTER_MODEL": DEFAULT_MODEL,
        },
        "ONNX default",
    ),
    Variant(
        "onnx-cuda-fp16",
        {
            "PRIVACY_FILTER_BACKEND": "onnx",
            "PRIVACY_FILTER_ONNX_PROVIDER": "cuda",
            "PRIVACY_FILTER_ONNX_FILE": "onnx/model_fp16.onnx",
            "PRIVACY_FILTER_MODEL": DEFAULT_MODEL,
        },
        "reference export",
    ),
    Variant(
        "onnx-cpu-int8",
        {
            "PRIVACY_FILTER_BACKEND": "onnx",
            "PRIVACY_FILTER_ONNX_PROVIDER": "cpu",
            "PRIVACY_FILTER_ONNX_FILE": "onnx/model_quantized.onnx",
            "PRIVACY_FILTER_MODEL": DEFAULT_MODEL,
        },
        "no GPU required",
    ),
    Variant(
        "onnx-cpu-fp16",
        {
            "PRIVACY_FILTER_BACKEND": "onnx",
            "PRIVACY_FILTER_ONNX_PROVIDER": "cpu",
            "PRIVACY_FILTER_ONNX_FILE": "onnx/model_fp16.onnx",
            "PRIVACY_FILTER_MODEL": DEFAULT_MODEL,
        },
        "quantization reference",
    ),
    Variant(
        "onnx-cpu-q4f16",
        {
            "PRIVACY_FILTER_BACKEND": "onnx",
            "PRIVACY_FILTER_ONNX_PROVIDER": "cpu",
            "PRIVACY_FILTER_ONNX_FILE": "onnx/model_q4f16.onnx",
            "PRIVACY_FILTER_MODEL": DEFAULT_MODEL,
        },
        "refused on CUDA",
    ),
    Variant(
        "nemotron-torch-cuda-bf16",
        {"PRIVACY_FILTER_BACKEND": "torch", "PRIVACY_FILTER_MODEL": NEMOTRON},
        "55 entity types, no ONNX export",
    ),
)


@dataclass
class Record:
    id: str
    tier: str
    text: str
    spans: list[Span] = field(default_factory=list)


def load_corpus() -> list[Record]:
    """Build texts and their ground-truth spans from the chunk lists.

    Offsets are derived here rather than written into the fixture, so editing a
    record's wording cannot silently invalidate its expected findings.
    """
    payload = json.loads(CORPUS.read_text(encoding="utf-8"))
    records = []
    for entry in payload["records"]:
        text = ""
        spans: list[Span] = []
        for chunk in entry["chunks"]:
            if isinstance(chunk, str):
                text += chunk
                continue
            label, fragment = chunk[0], chunk[1]
            # Mentions with no explicit key are their own entity.
            entity = chunk[2] if len(chunk) > 2 else f"{entry['id']}:{len(spans)}"
            spans.append(Span(len(text), len(text) + len(fragment), label, entity))
            text += fragment
        records.append(Record(id=entry["id"], tier=entry["tier"], text=text, spans=spans))
    return records


def post(path: str, payload: dict | None = None, timeout: float = 900.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class VariantUnavailable(RuntimeError):
    """This variant could not be brought up on this machine right now.

    A VRAM-starved GPU is the ordinary reason -- the torch rows refuse to load
    while a game holds the card -- so one unavailable variant is reported in
    its row rather than losing every other measurement in the sweep.
    """


def restart(variant: Variant) -> dict:
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("PRIVACY_FILTER_"):
            del env[name]
    env.update(variant.env)
    env["PRIVACY_FILTER_MAX_TOKENS"] = WINDOW_TOKENS
    result = subprocess.run(
        [str(ROOT / "privacy-filter-service"), "restart", "--force"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        reasons = [line for line in result.stdout.splitlines() if line.startswith("ERROR:")]
        raise VariantUnavailable(reasons[0] if reasons else result.stdout.strip())
    return post("/health", timeout=60.0)


def container_pid() -> int | None:
    result = subprocess.run(
        ["podman", "inspect", CONTAINER, "--format", "{{.State.Pid}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    digits = result.stdout.strip()
    return int(digits) if digits.isdigit() else None


def gpu_mib(pid: int) -> int:
    """MiB this process has resident on the GPU, or 0 when it holds none."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
            return int(fields[1])
    return 0


class GpuSampler:
    """Poll one process's GPU residency in the background, keeping the peak.

    Peak matters more than the figure after a scan: the transient allocation a
    forward pass makes is what decides whether the next window fits.
    """

    def __init__(self, pid: int | None, interval: float = 0.05) -> None:
        self.pid = pid
        self.interval = interval
        self.peak = 0
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> GpuSampler:
        if self.pid is not None:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        return self

    def run(self) -> None:
        assert self.pid is not None
        while not self.stopping.is_set():
            self.peak = max(self.peak, gpu_mib(self.pid))
            self.stopping.wait(self.interval)

    def __exit__(self, *exc: object) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def predictions(findings: list[dict], report: Report, text: str) -> list[Span]:
    """Canonical model spans, merged where only separators split one entity."""
    spans = []
    for finding in findings:
        label = canonical(finding["label"])
        if label is None:
            # Kept, because it still masks text and still counts for protection;
            # recorded, because a label nobody mapped is a gap in this harness
            # rather than evidence about the model.
            report.note_unmapped(finding["label"])
            label = finding["label"]
        spans.append(Span(finding["start"], finding["end"], label))
    return merge_predictions(spans, text)


def measure(variant: Variant, records: list[Record]) -> dict:
    health = restart(variant)
    pid = container_pid() if health.get("cuda_available") else None

    # One discarded pass: the first forward pass on a fresh session pays for
    # kernel selection and allocator growth, which is startup cost, not scan cost.
    for record in records:
        post("/check", {"text": record.text, "include_text": False})

    raw: list[list[dict]] = []
    with GpuSampler(pid) as sampler:
        started = time.monotonic()
        for record in records:
            response = post("/check", {"text": record.text, "include_text": False})
            raw.append(response["findings"])
        elapsed = time.monotonic() - started

    overall = Report()
    by_tier: dict[str, Report] = {}
    for record, findings in zip(records, raw):
        spans = predictions(findings, overall, record.text)
        overall.add_record(record.spans, spans, record.text)
        by_tier.setdefault(record.tier, Report()).add_record(record.spans, spans, record.text)

    return {
        "variant": variant.name,
        "note": variant.note,
        "model": health.get("model"),
        "backend": health.get("backend"),
        "provider": health.get("provider"),
        "dtype": health.get("dtype"),
        "model_file": health.get("model_file"),
        "max_tokens": health.get("max_tokens"),
        "seconds": round(elapsed, 3),
        "ms_per_text": round(1000 * elapsed / len(records), 1),
        "peak_gpu_mib": sampler.peak or None,
        "scores": overall.summary(),
        "by_tier": {tier: report.summary() for tier, report in sorted(by_tier.items())},
        # Raw spans, so --baseline can diff two variants without rescanning.
        "findings": [
            [(finding["start"], finding["end"], finding["label"]) for finding in findings]
            for findings in raw
        ],
    }


def disagreements(baseline: dict, other: dict, records: list[Record]) -> dict:
    """Where two variants actually differ, span by span.

    Two exports of one checkpoint can score identically and still disagree on
    individual spans; on a corpus this size that diff is the only thing with
    enough resolution to justify switching default weights.
    """
    detail = []
    differing = 0
    for record, left, right in zip(records, baseline["findings"], other["findings"]):
        def normalized(findings: list[list[object]]) -> set[tuple[int, int, str]]:
            raw = [
                {"start": start, "end": end, "label": label}
                for start, end, label in findings
            ]
            return {
                (span.start, span.end, span.label)
                for span in predictions(raw, Report(), record.text)
            }

        ours = normalized(left)
        theirs = normalized(right)
        if ours == theirs:
            continue
        differing += 1
        detail.append(
            {
                "record": record.id,
                "tier": record.tier,
                "only_in_baseline": sorted(ours - theirs),
                "only_in_variant": sorted(theirs - ours),
            }
        )
    return {
        "baseline": baseline["variant"],
        "variant": other["variant"],
        "records_differing": differing,
        "records": len(records),
        "detail": detail,
    }


def short_reason(error: str, limit: int = 90) -> str:
    """A refusal short enough for a table cell; --json keeps the whole thing.

    ONNX Runtime's allocator failures run to several hundred characters of
    source path and signature, which turns one unmeasured row into an unreadable
    table.
    """
    reason = error.removeprefix("ERROR: ").replace("|", "/").strip()
    return reason if len(reason) <= limit else reason[: limit - 1].rstrip() + "..."


def percent(value: float) -> str:
    return f"{100 * value:.0f}%"


def markdown(results: list[dict], records: list[Record], diffs: list[dict]) -> str:
    annotated = sum(len(record.spans) for record in records)
    clean = sum(1 for record in records if not record.spans)
    hard = sum(1 for record in records if record.tier == "hard")
    lines = [
        f"{len(records)} synthetic texts ({hard} hard, {clean} clean), "
        f"{annotated} annotated entities, {WINDOW_TOKENS}-token window.",
        "",
        "| Variant | Weights | Strict F1 | Partial F1 | Direct ER | Quasi ER | Hard-tier F1 |"
        " False alarms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        if result.get("error"):
            lines.append(
                f"| `{result['variant']}` | - | - | - | - | - | - | "
                f"{short_reason(result['error'])} |"
            )
            continue
        scores = result["scores"]
        hard_scores = result["by_tier"].get("hard", {}).get("strict", {"f1": 0.0})
        lines.append(
            f"| `{result['variant']}` | {result['dtype'] or '?'} | "
            f"{scores['strict']['f1']:.2f} | "
            f"{scores['partial']['f1']:.2f} | "
            f"{percent(scores['entity_recall']['direct']['recall'])} | "
            f"{percent(scores['entity_recall']['quasi']['recall'])} | "
            f"{hard_scores['f1']:.2f} | "
            f"{scores['false_alarms']} in {scores['clean_records']} clean texts |"
        )

    measured = [result for result in results if not result.get("error")]
    if measured:
        lines += [
            "",
            "| Variant | Corpus scan | GPU resident |",
            "| --- | ---: | ---: |",
        ]
        for result in measured:
            vram = f"{result['peak_gpu_mib']} MiB" if result["peak_gpu_mib"] else "-"
            lines.append(
                f"| `{result['variant']}` | {result['seconds']:.2f}s "
                f"({result['ms_per_text']:.0f} ms/text) | {vram} |"
            )

    for diff in diffs:
        lines += [
            "",
            f"`{diff['variant']}` vs `{diff['baseline']}`: "
            f"{diff['records_differing']} of {diff['records']} texts differ.",
        ]
        for record in diff["detail"]:
            lines.append(
                f"- `{record['record']}` ({record['tier']}): "
                f"baseline only {record['only_in_baseline']}, "
                f"variant only {record['only_in_variant']}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    names = [variant.name for variant in VARIANTS]
    parser.add_argument(
        "--variant",
        action="append",
        choices=names,
        help="Measure only this variant; repeatable. Default: all of them.",
    )
    parser.add_argument(
        "--baseline",
        choices=names,
        help="Diff every other measured variant's findings against this one.",
    )
    parser.add_argument("--json", action="store_true", help="Emit raw results, not a table.")
    args = parser.parse_args(argv)

    selected = [
        variant
        for variant in VARIANTS
        if args.variant is None or variant.name in args.variant
    ]
    if args.baseline and args.baseline not in {variant.name for variant in selected}:
        selected = [variant for variant in VARIANTS if variant.name == args.baseline] + selected

    records = load_corpus()
    results = []
    for variant in selected:
        print(f"measuring {variant.name} ...", file=sys.stderr)
        try:
            results.append(measure(variant, records))
        except VariantUnavailable as exc:
            print(f"  unavailable: {exc}", file=sys.stderr)
            results.append({"variant": variant.name, "note": variant.note, "error": str(exc)})

    diffs = []
    if args.baseline:
        baseline = next(
            (
                result
                for result in results
                if result["variant"] == args.baseline and not result.get("error")
            ),
            None,
        )
        if baseline is None:
            print(f"baseline {args.baseline} was not measurable", file=sys.stderr)
        else:
            diffs = [
                disagreements(baseline, result, records)
                for result in results
                if result is not baseline and not result.get("error")
            ]

    if args.json:
        print(json.dumps({"results": results, "disagreements": diffs}, indent=2))
    else:
        print(markdown(results, records, diffs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
