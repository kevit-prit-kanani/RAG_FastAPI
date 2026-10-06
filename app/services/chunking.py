from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

from app.models.schemas import Chunk, PageText

TARGET_TOKENS = 420
MAX_TOKENS = 520
OVERLAP_TOKENS = 60

BULLET_RE = re.compile(r"^(?:[+•●▪◦‣*-]|\d+[.)])\s+")
PERSON_RE = re.compile(r"^(?:Dr\.?|Prof\.?|Ms\.?|Mrs\.?|Mr\.?)\s+[A-Z][A-Za-z.\' -]{2,80}$")
HEADING_PATTERNS = (
    re.compile(r"^year\s+\d+$", re.IGNORECASE),
    re.compile(r"^trimester\s+[ivx]+$", re.IGNORECASE),
    re.compile(r"^(?:USPs? of(?: the)? program|faculties? of healthcare program|initiatives|industry engagement|career opportunities|student achievements|syllabus structure|student life journey|(?:interim )?placement facts.*|some of our recruiters|overview|program highlights?)$", re.IGNORECASE),
)
TABLE_HINT_RE = re.compile(r"\s{2,}|\t|\|", re.IGNORECASE)


def _encoding():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


ENCODING = _encoding()


def token_count(text: str) -> int:
    if ENCODING is None:
        return max(1, round(len(text.split()) * 1.3))
    return len(ENCODING.encode(text))


def _clean_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^[~^`]+(?=\d)", "", line)
    if re.fullmatch(r"[-—_.]+", line):
        return ""
    return line


def _is_heading(line: str) -> bool:
    if not line or len(line) > 100 or line.endswith((".", ",", ":", ";")):
        return False
    if BULLET_RE.match(line) or PERSON_RE.match(line):
        return False
    return any(pattern.match(line) for pattern in HEADING_PATTERNS) and token_count(line) <= 16


def _classify(line: str) -> str:
    if _is_heading(line):
        return "heading"
    if BULLET_RE.match(line):
        return "bullet_list"
    if TABLE_HINT_RE.search(line) and line.count("  ") >= 2:
        return "table"
    return "paragraph"


@dataclass(slots=True)
class Block:
    text: str
    block_type: str


def page_to_blocks(page: PageText) -> list[Block]:
    raw_lines = [_clean_line(x) for x in page.text.splitlines()]
    raw_lines = [x for x in raw_lines if x]
    blocks: list[Block] = []
    i = 0
    while i < len(raw_lines):
        current = raw_lines[i]
        kind = _classify(current)
        if kind == "heading":
            blocks.append(Block(current, "heading"))
            i += 1
            continue

        if kind == "bullet_list":
            bullets = [current]
            i += 1
            while i < len(raw_lines) and not _is_heading(raw_lines[i]):
                if BULLET_RE.match(raw_lines[i]):
                    bullets.append(raw_lines[i])
                    i += 1
                    continue
                if bullets and not PERSON_RE.match(raw_lines[i]):
                    bullets.append(raw_lines[i])
                    i += 1
                    continue
                break
            blocks.append(Block("\n".join(bullets), "bullet_list"))
            continue

        if PERSON_RE.match(current):
            person_lines = [current]
            i += 1
            while i < len(raw_lines) and not _is_heading(raw_lines[i]) and not PERSON_RE.match(raw_lines[i]) and not BULLET_RE.match(raw_lines[i]):
                person_lines.append(raw_lines[i])
                i += 1
            blocks.append(Block("\n".join(person_lines), "mixed"))
            continue

        lines = [current]
        i += 1
        while i < len(raw_lines) and _classify(raw_lines[i]) in {"paragraph", "table"} and not PERSON_RE.match(raw_lines[i]):
            lines.append(raw_lines[i])
            i += 1
        merged = " ".join(lines)
        kind = "table" if any(_classify(x) == "table" for x in lines) else "paragraph"
        blocks.append(Block(merged, kind))
    return blocks


def _split_by_tokens(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Split text while strictly enforcing the configured token ceiling.

    The fallback path uses an intentionally conservative words-per-token
    estimate and performs a post-check so the documented hard cap remains
    true even when tokenization differs between environments.
    """
    if ENCODING is None:
        words = text.split()
        estimated_window = max(1, int(max_tokens / 1.3))
        if len(words) <= estimated_window:
            return [text]

        word_overlap = max(0, int(overlap / 1.3))
        step = max(1, estimated_window - word_overlap)
        result: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + estimated_window, len(words))
            piece_words = words[start:end]
            piece = " ".join(piece_words).strip()
            # Guarantee the fallback estimator also respects MAX_TOKENS.
            while len(piece_words) > 1 and token_count(piece) > max_tokens:
                piece_words.pop()
                piece = " ".join(piece_words).strip()
            result.append(piece)
            if end >= len(words):
                break
            start += step
        return result

    tokens = ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    result: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        piece_tokens = tokens[start:end]
        piece = ENCODING.decode(piece_tokens).strip()
        # Be defensive against tokenizer/decoder differences: back off until
        # the public token_count() agrees with the hard ceiling.
        while len(piece_tokens) > 1 and token_count(piece) > max_tokens:
            piece_tokens = piece_tokens[:-1]
            piece = ENCODING.decode(piece_tokens).strip()
        result.append(piece)
        if end >= len(tokens):
            break
        start = max(start + 1, end - overlap)
    return result


def chunk_pages(pages: Iterable[PageText], document_id: UUID, source_filename: str = "unknown.pdf") -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        blocks = page_to_blocks(page)
        section_path: list[str] = []
        current_parts: list[str] = []
        current_types: list[str] = []
        seq = 0

        def flush() -> None:
            nonlocal seq, current_parts, current_types
            if not current_parts:
                return
            base = "\n\n".join(current_parts).strip()
            for piece_index, piece in enumerate(_split_by_tokens(base)):
                seq += 1
                chunk_id = f"{document_id}:p{page.page_number:02d}:c{seq:02d}:{piece_index:02d}"
                embedding_text = "\n".join(section_path + [piece]).strip()
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source_filename=source_filename,
                        page_number=page.page_number,
                        section_path=section_path.copy(),
                        block_type="mixed" if len(set(current_types)) > 1 else current_types[0],
                        content=piece,
                        embedding_text=embedding_text,
                        token_count=token_count(piece),
                    )
                )
            current_parts = []
            current_types = []

        for block in blocks:
            if block.block_type == "heading":
                flush()
                section_path = (section_path + [block.text])[-3:]
                continue
            candidate = "\n\n".join(current_parts + [block.text]).strip()
            if current_parts and token_count(candidate) > TARGET_TOKENS:
                flush()
            current_parts.append(block.text)
            current_types.append(block.block_type)
        flush()
    return chunks
