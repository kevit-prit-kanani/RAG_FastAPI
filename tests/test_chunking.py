from uuid import uuid4

from app.models.schemas import PageText
from app.services.chunking import chunk_pages, page_to_blocks, token_count


def test_heading_and_bullets():
    page = PageText(page_number=7, extraction_method="ocr", text="""USPs of the Program\n\n• First benefit\n• Second benefit\n\nFaculties of Healthcare Program\n\nDr. Example\nAssociate Professor\n""")
    blocks = page_to_blocks(page)
    assert blocks[0].block_type == "heading"
    assert blocks[1].block_type == "bullet_list"
    chunks = chunk_pages([page], uuid4(), "test.pdf")
    assert len(chunks) >= 2
    assert chunks[0].section_path == ["USPs of the Program"]
    assert "USPs of the Program" in chunks[0].embedding_text


def test_long_text_is_split():
    page = PageText(page_number=1, extraction_method="native", text="Overview\n\n" + ("healthcare program information " * 500))
    chunks = chunk_pages([page], uuid4(), "test.pdf")
    assert len(chunks) > 1
    assert all(token_count(c.content) <= 520 for c in chunks)

def test_long_text_hard_cap_is_strict():
    page = PageText(page_number=1, extraction_method="native", text=("healthcare program information " * 1200))
    chunks = chunk_pages([page], uuid4(), "test.pdf")
    assert chunks
    assert max(token_count(c.content) for c in chunks) <= 520

