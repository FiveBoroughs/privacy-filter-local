# privacy-filter-local

Run OpenAI's [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) model locally, on your own GPU, to find or redact personal data before text leaves your machine. It can also block git commits that would record it.

Nothing is sent anywhere. The model runs in a Podman container bound to `127.0.0.1`, and the CLI, the service and the git hook all fail closed: if something cannot be scanned, they say so and stop.

## Requirements

- An NVIDIA GPU with roughly 3 GB free VRAM. The service checks at startup and refuses to start below that.
- Podman with GPU support (the NVIDIA container toolkit / CDI).
- Python 3.11+ on the host for the CLI and hook. Standard library only.

The container loads the checkpoint in its native bfloat16, ~2.8 GB. Left alone, transformers upcasts to fp32 and doubles that, which will not share a 10 GB card with a game.

## Quick start

```bash
git clone https://github.com/FiveBoroughs/privacy-filter-local.git ~/Code/privacy-filter-local
cd ~/Code/privacy-filter-local

./privacy-filter-service ensure      # build (cached), start, wait until ready
echo "Contact Jane Doe at jane@example.com" | ./privacy-filter check
echo "Contact Jane Doe at jane@example.com" | ./privacy-filter redact
./privacy-filter-service stop        # give the VRAM back
```

`check` exits `1` when it finds something, `0` when it does not. Cold start is about 6-10 seconds. There is no keep-warm mode: pinning 2.8 GB of VRAM to save six seconds is a bad trade.

## Commands

| Command | Purpose |
| --- | --- |
| `privacy-filter check [--file F]` | Report findings as JSON; exit `1` if any |
| `privacy-filter redact [--file F]` | Print the text with each finding replaced by `[LABEL]` |
| `privacy-filter health` | Model, GPU, token budgets, in-flight scans |
| `privacy-filter-service ensure\|status\|stop\|release\|restart` | Container lifecycle |
| `git-pre-commit-pii` | Scan staged additions; block the commit on findings |

Exit codes are documented in `privacy-filter --help`. Anything other than `0` or `1` means the content was not scanned, so treat it as unsafe.

By default JSON findings omit the matched text. `--include-text` includes it; use it only when the output stays local.

## Large inputs

Submit whole files. The service tokenizes, splits the input into overlapping windows, and runs them sequentially:

- Windows overlap by `PRIVACY_FILTER_CHUNK_OVERLAP_TOKENS` (64). Cuts snap to whitespace, which is not enough on its own: a full name or street address spans several tokens.
- If a window does not fit in VRAM, only that span is re-planned at half the budget and retried, down to `PRIVACY_FILTER_MIN_TOKENS`. Windows that already succeeded are not rescanned.
- If even the minimum does not fit, the request fails with HTTP 503 and returns no partial findings.
- Duplicate findings from overlapping windows are coalesced, and offsets always refer to the original text.

Tuning knobs: `PRIVACY_FILTER_MAX_TOKENS` (4096), `PRIVACY_FILTER_MIN_TOKENS` (256), `PRIVACY_FILTER_TOKEN_SHRINK_FACTOR` (0.5), `PRIVACY_FILTER_CHUNK_OVERLAP_TOKENS` (64), `PRIVACY_FILTER_TIMEOUT` (900). `privacy-filter health` reports the live values.

Smaller windows are much slower, because the cost is per forward pass rather than per token. Lower the maximum only as far as your GPU needs.

No window is ever larger than the model's own maximum sequence length, which the service reads from the model config at startup and reports as `context_limit`. A larger window would be truncated silently and every finding past the cut would be lost, so the default budget gives way to a shorter-context model, while a `PRIVACY_FILTER_MAX_TOKENS` you set above what the model accepts is a startup error naming both numbers rather than a silent clamp. A model whose config declares no sequence length at all refuses to start: set `PRIVACY_FILTER_CONTEXT_LIMIT` to its real maximum. The tokenizer's own `model_max_length` can only lower the limit, never establish it — several models advertise far more there than the architecture accepts.

## Git hook

```bash
cp hooks/pre-commit.example .git/hooks/pre-commit   # or ~/.config/git/hooks for all repos
chmod +x .git/hooks/pre-commit
```

The hook scans added lines in staged changes, reports `path:line:column: label score`, and never prints the matched text. It starts the service if needed and stops it afterwards, taking a lease so concurrent commits do not tear the service out from under each other.

It blocks on anything it cannot scan completely: binary files, non-UTF-8 content, a service that is down or out of memory. The exception is its own two config files below, which it skips with a note. Scanning them is circular: the model reads a hex digest as a secret. Two escape hatches:

- Binary files: fnmatch patterns in `.privacy-filter-binary-allowlist` (repo root) or `PRIVACY_FILTER_BINARY_ALLOWLIST`.
- Reviewed false positives: `path<TAB>label<TAB>digest` lines in `.privacy-filter-reviewed-findings`, where the digest is a truncated SHA-256 of the matched text. The hook prints a paste-ready line whenever it blocks.

Entries are keyed on a digest of the content. A line-pinned entry would keep suppressing whatever later occupied that line, including real data with the same label. Because the digest is committed, only allowlist findings you have confirmed are not sensitive.

## Security model

- No network egress. The model is local; text is never sent to an API. The service binds `127.0.0.1` only.
- No authentication. Any local process can POST to `127.0.0.1:8757`, which is the same trust boundary as your own files. A web page cannot reach it: there are no CORS headers, cross-origin preflight is rejected, and non-JSON content types are refused. A hostile local process could submit text and read findings.
- The container runs with SELinux labelling disabled (`--security-opt=label=disable`) so it can reach the GPU device nodes. That weakens container confinement on the host; if it is unacceptable in your environment, work out a labelled CDI setup instead.
- Findings omit matched text by default in JSON, hook output, logs and errors. Only `--include-text` reveals it.
- Do not treat a clean scan as a compliance control. The model has false positives (example dates in docs, digits inside SVG path data) and false negatives. It misses things.

## Tests

```bash
python3 -m unittest discover -s tests
```

The suite runs on a plain interpreter with no GPU, no torch and no FastAPI: the chunk planner and the adaptive scanner are dependency-free, the model and CUDA error types are injected, and the service module is exercised against stubs. All fixtures are synthetic.

## Layout

| Path | Contents |
| --- | --- |
| `scripts/text_chunking.py` | Overlapping window planning (pure) |
| `scripts/adaptive_scan.py` | Budget ladder, OOM retries, dedup, redaction (pure) |
| `scripts/privacy_filter_service.py` | FastAPI service; runs inside the container |
| `scripts/privacy_filter_client.py` | CLI |
| `scripts/service_control.py` | Container lifecycle, locking, leases |
| `scripts/git_pre_commit_pii.py` | Pre-commit scanner |
| `skill/SKILL.md` | Claude Code skill; install with `./install-skill` |
