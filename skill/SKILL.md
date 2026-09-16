---
name: privacy-filter
description: Run OpenAI Privacy Filter locally to detect or redact personally identifiable information before content is sent to remote models, APIs, logs, tickets, prompts, or shared artifacts. Use when the user asks to check text, files, prompts, logs, exports, documents, code, or datasets for PII or secrets, or asks to sanitize/redact private content.
allowed-tools: Bash(/path/to/privacy-filter-local/privacy-filter:*), Bash(/path/to/privacy-filter-local/privacy-filter-service:*)
---

# Local Privacy Filter

Use the local OpenAI Privacy Filter wrapper at:

```bash
/path/to/privacy-filter-local/privacy-filter
```

The wrapper calls a local-only service at `127.0.0.1:8757`. The service runs `openai/privacy-filter` in Podman with GPU required; startup fails if CUDA is unavailable. Do not paste raw sensitive text into remote tools, web search, hosted inference APIs, issue comments, PR comments, or model prompts before running this local filter.

Submit whole files. The service splits large inputs into overlapping token windows itself, shrinks the window when the GPU is short of memory, and reports findings against the original offsets — never pre-chunk, truncate, or loop over slices of the input yourself.

## Service lifecycle

Use one command for the whole lifecycle. Do **not** run `start-service` and poll `health` by hand: `start-service` is the low-level primitive and its `podman run --replace` kills any scan already in flight.

```bash
/path/to/privacy-filter-local/privacy-filter-service ensure   # start if needed, block until ready
/path/to/privacy-filter-local/privacy-filter-service status   # report state, change nothing
/path/to/privacy-filter-local/privacy-filter-service stop     # stop and free VRAM
```

`ensure` is idempotent, takes a lock so concurrent callers cannot replace each other's container, and prints `started` or `reused` — run it before any scan and it costs ~50ms when the service is already up. A cold start takes about 6-10s, so there is no keep-warm mode; stop it when you are done and give the VRAM back.

`status` exit codes: `0` healthy, `1` down, `2` running but not ready, `3` running without a GPU.

### Concurrency

Commits and scans can overlap safely:

- `ensure` serializes on a lock, so simultaneous callers never replace each other's container — the loser gets `reused`.
- The service serializes GPU work per window, so two scans in flight at once both complete.
- `stop` **refuses** while another caller holds a lease or a scan is running, printing `refused: ...` and exiting `6`. Pass `--force` only when you want the VRAM back regardless and accept killing that work.

The pre-commit hook takes a *lease* for the duration of a commit and releases it afterwards, so the service is only torn down by the last caller out. You do not need leases for one-off scans — the running-scan guard already covers those.

Detailed health, including token budgets:

```bash
/path/to/privacy-filter-local/privacy-filter health
```

## Commands

Check stdin for PII. Exit code is `1` when findings exist and `0` when none are found (see Exit codes below for failures):

```bash
/path/to/privacy-filter-local/privacy-filter check < input.txt
```

Redact stdin:

```bash
/path/to/privacy-filter-local/privacy-filter redact < input.txt
```

Check or redact a file:

```bash
/path/to/privacy-filter-local/privacy-filter check --file /path/to/file.txt
/path/to/privacy-filter-local/privacy-filter redact --file /path/to/file.txt
```

Return JSON with redacted text and findings:

```bash
/path/to/privacy-filter-local/privacy-filter redact --json < input.txt
```

By default JSON findings omit raw matched text. Add `--include-text` only when the user explicitly needs the matched substrings and the output will stay local.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | No findings |
| 1 | Findings reported |
| 2 | Invalid or undecodable input (missing file, not UTF-8) |
| 3 | Service unreachable |
| 4 | Service is not GPU-backed |
| 5 | GPU out of memory even at the minimum window; nothing was scanned |
| 6 | Other service error |
| 7 | Scan did not finish before the timeout; nothing was scanned |

Codes 2 to 7 all mean the content was **not** scanned. Treat them as "still unsafe to send anywhere", never as a pass. For code 5, free VRAM (a running game is the usual cause), lower `PRIVACY_FILTER_MAX_TOKENS`, or retry.

## Large inputs and GPU memory

The service starts at `PRIVACY_FILTER_MAX_TOKENS` per window. When a window does not fit in VRAM it discards that window's results, frees the cache, and retries only that span at `PRIVACY_FILTER_TOKEN_SHRINK_FACTOR` of the budget, down to `PRIVACY_FILTER_MIN_TOKENS`. Windows that already succeeded are not rescanned; if the minimum still does not fit, the request fails closed with HTTP 503 and no partial findings.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PRIVACY_FILTER_MAX_TOKENS` | 4096 | Starting window size, capped by the tokenizer context |
| `PRIVACY_FILTER_MIN_TOKENS` | 256 | Smallest window tried before giving up |
| `PRIVACY_FILTER_TOKEN_SHRINK_FACTOR` | 0.5 | Backoff applied per retry |
| `PRIVACY_FILTER_CHUNK_OVERLAP_TOKENS` | 64 | Tokens each window re-reads, so PII cannot hide in a cut |
| `PRIVACY_FILTER_TIMEOUT` | 900 | Client-side seconds to wait for a scan |
| `PRIVACY_FILTER_HEALTH_TIMEOUT` | 30 | Client-side seconds to wait for `health` |

A timeout (exit 7) is reported separately from an unreachable service (exit 3): a slow scan usually means the service is busy, not stopped, so restarting it is the wrong move. Very small window sizes make scans dramatically slower — one forward pass per window — so lower `PRIVACY_FILTER_MAX_TOKENS` only as far as the GPU actually needs.

`privacy-filter health` reports these live, along with the retry ladder. GPU inference is serialized in-process, so two concurrent requests cannot allocate transient memory at the same time.

## Git pre-commit hook

`git-pre-commit-pii` scans staged additions and fails closed: anything it cannot scan completely blocks the commit. That includes large diffs (windowed by the service, never skipped for size), binary files, non-UTF-8 content, and a service that is down or out of memory. Fix the cause and commit again; do not advise bypassing the hook.

Binary files can only be opted out deliberately, by fnmatch pattern, in `PRIVACY_FILTER_BINARY_ALLOWLIST` (colon- or comma-separated) or one pattern per line in a repo-root `.privacy-filter-binary-allowlist` file. Add a pattern only once the user has confirmed those files carry no PII.

### Reviewed false positives

Confirmed false positives go in a repo-root `.privacy-filter-reviewed-findings`, one tab-separated entry per line:

```
path<TAB>label<TAB>digest
src/index.ts	private_url	a1b2c3d4e5f60718
```

The digest is a truncated SHA-256 of the exact matched text, computed locally and never transmitted. **Keying on content, not line number, is deliberate**: a line-pinned entry would go on suppressing whatever later occupied that line, including real PII with the same label. An entry therefore stops applying the moment the text changes.

You cannot write these by hand. When the hook blocks, it prints a paste-ready line for each finding; add only the ones the user has actually reviewed and confirmed harmless. Entries in the older `path<TAB>line<TAB>label` format are rejected outright — the hook fails closed and tells you to regenerate.

Note that the digest of a reviewed entry is committed, so only allowlist findings whose content you have confirmed is not sensitive. That is what "reviewed false positive" means; never use this file to silence a real secret.

## Agent Workflow

1. For user-provided text that may contain PII, save or pipe it directly to the local command. Avoid repeating the raw text in the chat.
2. Prefer `redact` before summarizing, debugging, or forwarding content into any remote service.
3. Use `check` when the user only needs a pass/fail or finding offsets.
4. Treat this as a privacy aid, not a compliance guarantee. For medical, legal, financial, HR, education, government, or other high-sensitivity content, tell the user that human review is still needed.

## Labels

The model can report `private_person`, `private_address`, `private_email`, `private_phone`, `private_url`, `private_date`, `account_number`, and `secret`.
