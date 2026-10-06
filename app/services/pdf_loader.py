from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from typing import BinaryIO

import fitz
import pytesseract
from PIL import Image

from app.models.schemas import PageText

NATIVE_TEXT_MIN_CHARS = 40
OCR_DPI = 180


@dataclass(slots=True)
class OCRWord:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float
    block_num: int
    par_num: int
    line_num: int


def _ocr_page(page: fitz.Page) -> tuple[str, float]:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72), alpha=False)
    image = Image.open(BytesIO(pixmap.tobytes("png")))
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 3")

    words: list[OCRWord] = []
    confidences: list[float] = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = 0.0
        if conf >= 0:
            confidences.append(conf)
        words.append(
            OCRWord(
                text=text,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
                conf=conf,
                block_num=int(data["block_num"][i]),
                par_num=int(data["par_num"][i]),
                line_num=int(data["line_num"][i]),
            )
        )

    lines: dict[tuple[int, int, int], list[OCRWord]] = {}
    for word in words:
        lines.setdefault((word.block_num, word.par_num, word.line_num), []).append(word)

    ordered_lines: list[tuple[int, int, str]] = []
    for line_words in lines.values():
        line_words.sort(key=lambda w: w.x)
        ordered_lines.append((min(w.x for w in line_words), min(w.y for w in line_words), " ".join(w.text for w in line_words)))
    ordered_lines.sort(key=lambda item: (item[1], item[0]))

    text = "\n".join(item[2] for item in ordered_lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, avg_conf


def extract_pages(file_obj: BinaryIO) -> list[PageText]:
    raw = file_obj.read()
    doc = fitz.open(stream=raw, filetype="pdf")
    pages: list[PageText] = []
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            native = " ".join(part.strip() for part in page.get_text("text").splitlines() if part.strip())
            if len(native) >= NATIVE_TEXT_MIN_CHARS:
                pages.append(PageText(page_number=page_index + 1, text=native, extraction_method="native"))
            else:
                ocr_text, confidence = _ocr_page(page)
                pages.append(
                    PageText(
                        page_number=page_index + 1,
                        text=ocr_text,
                        extraction_method="ocr",
                        ocr_confidence=confidence,
                    )
                )
    finally:
        doc.close()
    return pages
