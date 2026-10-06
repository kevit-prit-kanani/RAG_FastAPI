from __future__ import annotations

import json
import sys
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import UUID

from app.services.chunking import chunk_pages
from app.services.pdf_loader import extract_pages


def main(path: str) -> None:
    pdf = Path(path)
    raw = pdf.read_bytes()
    document_id = UUID(bytes=sha256(raw).digest()[:16], version=4)
    pages = extract_pages(BytesIO(raw))
    chunks = chunk_pages(pages, document_id, pdf.name)
    print(json.dumps({
        "document_id": str(document_id),
        "source_filename": pdf.name,
        "pages": len(pages),
        "native_pages": sum(p.extraction_method == "native" for p in pages),
        "ocr_pages": sum(p.extraction_method == "ocr" for p in pages),
        "chunks": len(chunks),
        "sample_chunks": [c.model_dump(mode="json") for c in chunks[:10]],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/preview_pdf.py <file.pdf>")
    main(sys.argv[1])
