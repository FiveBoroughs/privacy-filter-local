# AGENTS.md

Instructions for coding agents working in this repository. `CLAUDE.md` is a symlink to this file.

This is a local PII scanner: a containerized torch/CUDA or ONNX/CUDA-or-CPU service, a CLI, and a git pre-commit hook. It handles text people expect never to leave their machine, and its value is entirely in being trustworthy about that. Read the invariants before changing behaviour.

## Invariants

Breaking any of these is a bug even if the tests pass.

1. Fail closed. Anything that cannot be scanned completely blocks the commit or returns an error. Never add a path where unscanned content produces a success exit code. Exit codes other than 0 and 1 all mean "not scanned".
2. Never emit raw matched text. JSON findings omit `text` unless `--include-text` is passed. Hook output, logs, service errors and health responses never contain scanned text. Report offsets, labels and scores.
3. Keep the logic testable without a GPU. `text_chunking.py`, `adaptive_scan.py` and `token_classification.py` import nothing heavier than the standard library, model/runtime errors are injected, and the service module is exercised against stubs. Heavy imports are conditional on the selected backend. New decision logic belongs in the pure modules; `privacy_filter_service.py` stays thin wiring. The suite must keep running on a plain interpreter with no torch, ONNX Runtime or FastAPI.
4. All fixtures are synthetic. Never add real personal data, or anything derived from it, to tests, docs, logs or commit messages.
5. Never suggest bypassing the hook. No `--no-verify`, no "skip the scan this once". `tests/test_pre_commit_scan_policy.py` greps the source tree for this and fails.

## Layout

| Path | Notes |
| --- | --- |
| `scripts/text_chunking.py` | Window planning. Pure, no dependencies. |
| `scripts/adaptive_scan.py` | Budget ladder, OOM retries, dedup, redaction. Pure, dependencies injected. |
| `scripts/token_classification.py` | Transformers-compatible BIO grouping. Pure, no dependencies. |
| `scripts/onnx_backend.py` | Raw ONNX Runtime classifier, tokenizer and Hub download. |
| `scripts/privacy_filter_service.py` | Thin FastAPI/backend wiring. Heavy imports are conditional. |
| `scripts/privacy_filter_client.py` | CLI. Standard library only. |
| `scripts/service_control.py` | Container lifecycle, locking, leases. Standard library only. |
| `scripts/git_pre_commit_pii.py` | Pre-commit scanner. Standard library only. |
| `skill/SKILL.md` | Claude Code skill. Paths are placeholders; `./install-skill` substitutes them. |

The service files copied into an image are enumerated in `Containerfile`; add any new runtime module there. The torch and ONNX images install different pinned requirement sets.

## Working on it

Run the suite with `python3 -m unittest discover -s tests`. It needs no GPU and takes about two seconds.

Use `./privacy-filter-service ensure` to bring the service up, and `./privacy-filter-service stop` to take it down. Do not run `./start-service` directly while anything might be scanning: it ends in `podman run --replace`, and a scan that outlasts podman's ten-second stop grace is SIGKILLed, which the caller sees as a bare connection failure. `ensure` holds a lock and reuses a healthy service in about 50ms. `stop` refuses while another caller holds a lease or a scan is in flight; `--force` overrides that.

To try a different model or backend without code changes, set `PRIVACY_FILTER_MODEL` or `PRIVACY_FILTER_BACKEND` and use `privacy-filter-service restart`; `ensure` deliberately reuses an already healthy service. `start-service` forwards the budget, model, backend and ONNX-file environment variables into the container.

Changing the CLI surface, exit codes or environment variables means updating `README.md` and `skill/SKILL.md` together, then re-running `./install-skill`.

## Measured numbers

Do not re-derive these; they were measured on an RTX 3080 sharing the GPU with a desktop session.

| | |
| --- | --- |
| Cold start to ready | 6 to 10 seconds, which is why there is no keep-warm mode |
| Warm `ensure` | about 50ms |
| 61 KB scan | under 2 seconds warm |
| 1 MiB staged diff | about 6 minutes |
| Container image | torch/CUDA 6.28 GB; ONNX CUDA+CPU 1.51 GB |

## Traps

The tokenizer does not know the model's real context limit. `resolve_context_limit` in `adaptive_scan.py` takes it from the model config (`max_seq_len`, `max_position_embeddings`, `n_positions`, including nested encoder configs, smallest wins); `tokenizer.model_max_length` may only lower that, never establish it. A window bigger than the model accepts is truncated silently and findings past the cut vanish from a scan that reports success, so a model that declares no limit refuses to start rather than falling back to a guess. Adding a backend means declaring its real limit there — not widening the guard.

The ONNX backend defaults to `onnx/model_quantized.onnx` (int8, 1.5 GB weights), not q4f16. On the 30-text synthetic CPU comparison, int8 matched fp16 ONNX on all 55 findings; q4f16 had 4 misses and 5 additions. On CUDA/RTX 3080, ONNX Runtime 1.30 reported no valid int4 MoE GEMM configuration and returned severely incomplete q4f16 findings, so q4/q4f16 are refused whenever CUDAExecutionProvider is active. `PRIVACY_FILTER_ONNX_FILE` makes exports selectable for CPU; changing the CUDA-safe default requires another recall comparison. The raw backend must preserve the exact `TokenClassificationPipeline` `simple` grouping semantics in `token_classification.py`; its E-/S- behavior is odd but downstream merging depends on it.

`PRIVACY_FILTER_ONNX_PROVIDER=auto` prefers CUDA and falls back to CPU; `cuda` requires CUDA and never falls back silently; `cpu` forces CPU and receives no GPU device mapping. The int8 CUDA path measured 534 MiB resident after a small scan, 750 MiB / 0.406s at 1024 tokens, 2.29 GiB / 0.733s at 2048, and 5.36 GiB / 1.54s at 4096. Do not infer peak VRAM from the 1.5 GB weight file.

On a cold ONNX CUDA start with no explicit `PRIVACY_FILTER_MAX_TOKENS`, `start-service` selects 4096 tokens at 6144 MiB free, 2048 at 3072 MiB, 1024 at 1536 MiB, and 256 below that. These thresholds leave about 0.75 GiB above the measured peaks. Keep the explicit override authoritative, and keep adaptive per-request shrinking because VRAM availability can change after startup.

ONNX Runtime exposes allocator OOM as a broad `Fail`, not a distinct exception. `OnnxClassifier` recognizes only explicit allocator markers and raises `OnnxOutOfMemory` for adaptive shrinking; unrelated ORT failures propagate. ONNX CPU retries only native `MemoryError`. Never broaden this to catch every ORT failure as memory pressure.

Offsets are always original-text coordinates. Windows overlap, spans get re-planned at smaller budgets on OOM, and findings are merged across windows. Everything downstream, including `location_for` in the hook and `redact_text`, depends on offsets referring to the text the caller submitted.

Reviewed false positives are keyed on a content digest, never a line number. A line-pinned entry keeps suppressing whatever later occupies that line, including real data with the same label. The old `path/line/label` format is rejected rather than honoured.

The hook does not scan its own control files. `.privacy-filter-reviewed-findings` and `.privacy-filter-binary-allowlist` are skipped with a printed note, because the model reads a hex digest as a secret and allowlisting those findings would generate more digests.

The service owns chunking. Callers submit whole files. Never pre-chunk, truncate, or loop over slices in a client.

## Commits

Single-line commit messages, no body.

This repository's own hook scans its commits. When it blocks, it prints a paste-ready allowlist line. Read the flagged content before adding one; the entry's digest is committed, so only allowlist findings confirmed to be non-sensitive.
