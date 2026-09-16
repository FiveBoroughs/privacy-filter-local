"""Pure, dependency-free helpers for splitting large inputs into windows.

The Privacy Filter model has a 128k context, so length is not the problem -- but
each forward pass still allocates transient GPU memory that grows with the token
count. On a VRAM-constrained card (e.g. while a game holds most of the 3080's
10GB) a big lockfile can tip an already-squeezed GPU into a CUDA OOM that a small
commit would survive. Chunking caps the per-request token count to bound that
transient allocation, cutting a 108KB blob into windows the service can process
one at a time.

Windows overlap by a configurable number of tokens. Snapping cuts to whitespace
keeps single tokens intact, but multi-token entities -- a full name, a street
address, an account number split by punctuation -- straddle whitespace, so a
contiguous split can hide PII from the model entirely. An overlap of N tokens
guarantees that any entity of at most N tokens appears whole in at least one
window; the duplicate findings that overlap produces are coalesced downstream.

This module intentionally imports nothing heavy (no torch/transformers) so the
planning logic can be unit-tested on a plain interpreter, away from the GPU
container where the model actually runs.
"""

from __future__ import annotations

Span = tuple[int, int]


def snap_to_whitespace(text: str, lo: int, hi: int) -> int:
    """Return a cut position in ``(lo, hi]`` that follows the last whitespace.

    Cutting just after a whitespace character keeps non-whitespace tokens (the
    things the model labels as PII) intact within a single window. When the
    window holds no whitespace at all -- e.g. one enormous minified line -- there
    is nothing to snap to, so the hard boundary ``hi`` is returned unchanged.
    """
    index = hi - 1
    while index > lo:
        if text[index].isspace():
            return index + 1
        index -= 1
    return hi


def clamp_overlap(budget: int, overlap: int) -> int:
    """Clamp ``overlap`` to something a window of ``budget`` tokens can carry.

    Half the budget is the hard ceiling: each window advances by at least
    ``budget - overlap`` tokens, so a larger overlap would shrink the stride
    towards zero and blow up the number of forward passes.
    """
    return max(0, min(overlap, budget // 2))


def plan_chunks(text: str, offsets: list[Span], budget: int, overlap: int = 0) -> list[Span]:
    """Plan ``(start, end)`` character spans covering ``text``.

    ``offsets`` is the per-token ``(char_start, char_end)`` mapping produced by a
    fast tokenizer with ``add_special_tokens=False``. Each returned span holds at
    most ``budget`` tokens -- overlap included -- and, wherever possible, ends on
    a whitespace boundary. The spans are returned in order, together cover
    ``[0, len(text))``, and each one starts at most ``overlap`` tokens before its
    predecessor ended, so any entity of at most ``overlap`` tokens that straddles
    a cut appears whole in the following window.

    Offsets are always original ``text`` coordinates; callers can slice ``text``
    directly and map findings back by adding the span start.

    A synthetic one-offset-per-character mapping can be passed as a conservative
    fallback when no fast tokenizer is available: because every token spans at
    least one character, ``budget`` characters never exceed ``budget`` tokens.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if not text:
        return []
    token_count = len(offsets)
    if token_count <= budget:
        return [(0, len(text))]

    overlap = clamp_overlap(budget, overlap)
    spans: list[Span] = []
    start_char = 0
    token_index = 0
    while token_index < token_count:
        end_token = token_index + budget
        if end_token >= token_count:
            spans.append((start_char, len(text)))
            break

        hard_cut = offsets[end_token][0]
        cut = snap_to_whitespace(text, start_char, hard_cut)
        end_index = first_token_from(offsets, token_index, cut)

        if end_index <= token_index:
            # No whitespace to snap to and snapping made no progress; fall back
            # to the hard token boundary so the loop always advances.
            cut = hard_cut
            end_index = end_token

        spans.append((start_char, cut))

        # Rewind ``overlap`` tokens so the next window re-reads the boundary
        # region; never rewind past forward progress.
        next_index = max(token_index + 1, end_index - overlap)
        # ``min`` keeps the windows gap-free when the rewind lands after the cut
        # (overlap 0, cut inside the whitespace run before the next token).
        start_char = min(cut, offsets[next_index][0])
        token_index = next_index

    return spans


def first_token_from(offsets: list[Span], token_index: int, cut: int) -> int:
    """Index of the first token at or after ``cut``, scanning from ``token_index``."""
    index = token_index
    while index < len(offsets) and offsets[index][0] < cut:
        index += 1
    return index
