# privacy-filter-local

Run OpenAI's [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) model locally, on an NVIDIA GPU with torch or on the CPU with ONNX Runtime, to find or redact personal data before text leaves your machine. It can also block git commits that would record it.

Nothing is sent anywhere. The model runs in a Podman container bound to `127.0.0.1`, and the CLI, the service and the git hook all fail closed: if something cannot be scanned, they say so and stop.

## Requirements

- Podman.
- Python 3.11+ on the host for the CLI and hook. Standard library only.
- For the default torch backend: an NVIDIA GPU with roughly 3 GB free VRAM, plus the NVIDIA container toolkit / CDI. Startup refuses below that threshold.
- For the ONNX backend: an NVIDIA GPU is optional. `auto` uses CUDA when the device/toolkit is present and otherwise falls back to CPU; `cpu` forces CPU-only operation.

The default container loads the torch checkpoint in its native bfloat16, ~2.8 GB. Left alone, transformers upcasts to fp32 and doubles that, which will not share a 10 GB card with a game. ONNX defaults to the 1.5 GB int8 export: it matched fp16 ONNX on the synthetic comparison corpus and is safe on both CUDA and CPU. The smaller q4f16 export changed findings on CPU and returned incomplete findings through CUDA on the tested RTX 3080, so CUDA refuses q4 exports.

## Quick start

```bash
git clone https://github.com/FiveBoroughs/privacy-filter-local.git ~/Code/privacy-filter-local
cd ~/Code/privacy-filter-local

./privacy-filter-service ensure      # default: torch on NVIDIA GPU
echo "Contact Jane Doe at jane@example.com" | ./privacy-filter check
echo "Contact Jane Doe at jane@example.com" | ./privacy-filter redact
./privacy-filter-service stop        # release the container and its resources
```

ONNX with CUDA preferred and CPU fallback:

```bash
PRIVACY_FILTER_BACKEND=onnx ./privacy-filter-service ensure
```

Force CPU-only ONNX:

```bash
PRIVACY_FILTER_BACKEND=onnx PRIVACY_FILTER_ONNX_PROVIDER=cpu \
  ./privacy-filter-service ensure
```

Use `restart`, not `ensure`, when changing the backend of an already healthy service: `ensure` deliberately reuses anything healthy rather than replacing an in-flight scanner.

`check` exits `1` when it finds something, `0` when it does not. A cold torch start is about 6-10 seconds. There is no keep-warm mode by default: pinning 2.8 GB of VRAM to save six seconds is a bad trade.

### Backends

| Backend | Select with | Runtime | Default weights | Image |
| --- | --- | --- | --- | --- |
| torch | `PRIVACY_FILTER_BACKEND=torch` (default) | NVIDIA CUDA | bf16 checkpoint, ~2.8 GB | 6.28 GB |
| ONNX | `PRIVACY_FILTER_BACKEND=onnx` | CUDA preferred, CPU fallback | `onnx/model_quantized.onnx`, int8, ~1.5 GB | 1.51 GB |

`PRIVACY_FILTER_ONNX_PROVIDER` accepts `auto` (default), `cuda` or `cpu`. `cuda` refuses to start when CUDAExecutionProvider cannot load; it never falls back silently. `auto` maps an NVIDIA device when available and otherwise starts with CPUExecutionProvider.

On a cold ONNX CUDA start, when `PRIVACY_FILTER_MAX_TOKENS` is unset, the launcher measures free VRAM and selects the largest measured budget with roughly 0.75 GiB of headroom: 4096 tokens at 6 GiB free, 2048 at 3 GiB, 1024 at 1.5 GiB, and 256 below that. The per-request OOM ladder still adapts if availability changes later. An explicit `PRIVACY_FILTER_MAX_TOKENS` always overrides startup selection.

Choose another published export with `PRIVACY_FILTER_ONNX_FILE`, for example `onnx/model_fp16.onnx`. The graph and all of its external data files are downloaded together. q4/q4f16 may be selected only with the CPU provider: CUDA startup refuses them because ONNX Runtime 1.30 returned incomplete findings on the tested RTX 3080. `privacy-filter health` reports the live `backend`, `provider`, `model_file` and `dtype`.

## Commands

| Command | Purpose |
| --- | --- |
| `privacy-filter check [--file F]` | Report findings as JSON; exit `1` if any |
| `privacy-filter redact [--file F]` | Print the text with each finding replaced by `[LABEL]` |
| `privacy-filter health` | Backend, model, provider, token budgets, in-flight scans |
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

Tuning knobs: `PRIVACY_FILTER_MAX_TOKENS` (4096 unless ONNX CUDA selects a smaller cold-start budget), `PRIVACY_FILTER_MIN_TOKENS` (256), `PRIVACY_FILTER_TOKEN_SHRINK_FACTOR` (0.5), `PRIVACY_FILTER_CHUNK_OVERLAP_TOKENS` (64), `PRIVACY_FILTER_TIMEOUT` (900). `privacy-filter health` reports the live values.

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
- The container runs with SELinux labelling disabled (`--security-opt=label=disable`) so the shared model cache is reachable and a CUDA-selected backend can reach GPU device nodes. That weakens container confinement on the host; if it is unacceptable in your environment, work out a labelled volume/CDI setup instead.
- Findings omit matched text by default in JSON, hook output, logs and errors. Only `--include-text` reveals it.
- Do not treat a clean scan as a compliance control. The model has false positives (example dates in docs, digits inside SVG path data) and false negatives. It misses things.

## Tests

```bash
python3 -m unittest discover -s tests
```

The suite runs on a plain interpreter with no GPU, torch, ONNX Runtime or FastAPI: window planning, token grouping and adaptive scanning are dependency-free, heavy runtimes are loaded only for their selected backend, and the service module is exercised against stubs. All fixtures are synthetic.

## Layout

| Path | Contents |
| --- | --- |
| `scripts/text_chunking.py` | Overlapping window planning (pure) |
| `scripts/adaptive_scan.py` | Budget ladder, OOM retries, dedup, redaction (pure) |
| `scripts/token_classification.py` | Transformers-compatible BIO grouping (pure) |
| `scripts/onnx_backend.py` | Raw ONNX Runtime classifier and fast-tokenizer adapter |
| `scripts/privacy_filter_service.py` | FastAPI service; runs inside the container |
| `scripts/privacy_filter_client.py` | CLI |
| `scripts/service_control.py` | Container lifecycle, locking, leases |
| `scripts/git_pre_commit_pii.py` | Pre-commit scanner |
| `skill/SKILL.md` | Claude Code skill; install with `./install-skill` |
