# AGENTS.md

Instructions for coding agents working in this repository. `CLAUDE.md` is a symlink to this file.

This is a local PII scanner: a GPU service in a container, a CLI, and a git pre-commit hook. It handles text people expect never to leave their machine, and its value is entirely in being trustworthy about that. Read the invariants before changing behaviour.

## Invariants

Breaking any of these is a bug even if the tests pass.

1. Fail closed. Anything that cannot be scanned completely blocks the commit or returns an error. Never add a path where unscanned content produces a success exit code. Exit codes other than 0 and 1 all mean "not scanned".
2. Never emit raw matched text. JSON findings omit `text` unless `--include-text` is passed. Hook output, logs, service errors and health responses never contain scanned text. Report offsets, labels and scores.
3. Keep the logic testable without a GPU. `text_chunking.py` and `adaptive_scan.py` import nothing heavier than the standard library, the model and CUDA error types are injected, and the service module is exercised against stubs. New logic belongs in the pure modules; `privacy_filter_service.py` stays thin wiring. The suite must keep running on a plain interpreter with no torch and no FastAPI.
4. All fixtures are synthetic. Never add real personal data, or anything derived from it, to tests, docs, logs or commit messages.
5. Never suggest bypassing the hook. No `--no-verify`, no "skip the scan this once". `tests/test_pre_commit_scan_policy.py` greps the source tree for this and fails.

## Layout

| Path | Notes |
| --- | --- |
| `scripts/text_chunking.py` | Window planning. Pure, no dependencies. |
| `scripts/adaptive_scan.py` | Budget ladder, OOM retries, dedup, redaction. Pure, dependencies injected. |
| `scripts/privacy_filter_service.py` | FastAPI app. Only file that imports torch or transformers. Runs inside the container. |
| `scripts/privacy_filter_client.py` | CLI. Standard library only. |
| `scripts/service_control.py` | Container lifecycle, locking, leases. Standard library only. |
| `scripts/git_pre_commit_pii.py` | Pre-commit scanner. Standard library only. |
| `skill/SKILL.md` | Claude Code skill. Paths are placeholders; `./install-skill` substitutes them. |

Only `privacy_filter_service.py`, `adaptive_scan.py` and `text_chunking.py` are copied into the image. Anything the service needs at runtime must be added to the `Containerfile`.

## Working on it

Run the suite with `python3 -m unittest discover -s tests`. It needs no GPU and takes about two seconds.

Use `./privacy-filter-service ensure` to bring the service up, and `./privacy-filter-service stop` to take it down. Do not run `./start-service` directly while anything might be scanning: it ends in `podman run --replace`, and a scan that outlasts podman's ten-second stop grace is SIGKILLed, which the caller sees as a bare connection failure. `ensure` holds a lock and reuses a healthy service in about 50ms. `stop` refuses while another caller holds a lease or a scan is in flight; `--force` overrides that.

To try a different model without code changes: `PRIVACY_FILTER_MODEL=<hf-id> ./privacy-filter-service ensure`. `start-service` forwards the budget and model environment variables into the container.

Changing the CLI surface, exit codes or environment variables means updating `README.md` and `skill/SKILL.md` together, then re-running `./install-skill`.

## Measured numbers

Do not re-derive these; they were measured on an RTX 3080 sharing the GPU with a desktop session.

| | |
| --- | --- |
| Cold start to ready | 6 to 10 seconds, which is why there is no keep-warm mode |
| Warm `ensure` | about 50ms |
| 61 KB scan | under 2 seconds warm |
| 1 MiB staged diff | about 6 minutes |
| Container image | 5.24 GB, of which the pip layer is 5.11 GB |

## Traps

The tokenizer does not know the model's real context limit. `max_token_budget()` reads `tokenizer.model_max_length`, which some models set far above what the architecture accepts, and the fallback for absurd values is a fixed 4096 that can also be too large. A window bigger than the model accepts is truncated silently, and findings past the cut vanish from a scan that reports success. See issues #3 and #4 before adding a model.

Offsets are always original-text coordinates. Windows overlap, spans get re-planned at smaller budgets on OOM, and findings are merged across windows. Everything downstream, including `location_for` in the hook and `redact_text`, depends on offsets referring to the text the caller submitted.

Reviewed false positives are keyed on a content digest, never a line number. A line-pinned entry keeps suppressing whatever later occupies that line, including real data with the same label. The old `path/line/label` format is rejected rather than honoured.

The hook does not scan its own control files. `.privacy-filter-reviewed-findings` and `.privacy-filter-binary-allowlist` are skipped with a printed note, because the model reads a hex digest as a secret and allowlisting those findings would generate more digests.

The service owns chunking. Callers submit whole files. Never pre-chunk, truncate, or loop over slices in a client.

## Commits

Single-line commit messages, no body.

This repository's own hook scans its commits. When it blocks, it prints a paste-ready allowlist line. Read the flagged content before adding one; the entry's digest is committed, so only allowlist findings confirmed to be non-sensitive.
