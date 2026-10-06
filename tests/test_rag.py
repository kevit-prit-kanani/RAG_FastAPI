import pytest

from app.models.schemas import ChatRequest, Citation, GroundedAnswer, RetrievedChunk, SearchToolArgs
from app.services.rag import RAGService


class FakeConversationRepo:
    def __init__(self):
        self.history = []
        self.saved = None

    async def get_history(self, conversation_id, limit=8):
        from uuid import uuid4
        return conversation_id or uuid4(), list(self.history)

    async def append(self, conversation_id, user_content, assistant_content):
        self.saved = (conversation_id, user_content, assistant_content)


class FakeRetriever:
    async def search(self, request):
        return [
            RetrievedChunk(
                chunk_id="c1",
                page_number=7,
                section_path=["Faculties of Healthcare Program"],
                content="Dr. Example — Professor",
                fused_score=1.0,
            )
        ]


class FakeLLM:
    def __init__(self):
        self.plan_calls = 0
        self.answer_calls = 0

    async def plan_search(self, question, history):
        self.plan_calls += 1
        if self.plan_calls < 2:
            raise ValueError("bad function arguments")
        return SearchToolArgs(
            queries=[question, "Healthcare Program faculty"], keywords=["faculty"], top_k=8, filters=None
        )

    async def generate_grounded_answer(self, question, context, history):
        self.answer_calls += 1
        return GroundedAnswer(
            answer="Dr. Example is a professor.",
            citations=[Citation(chunk_id="c1", page_number=7)],
            grounded=True,
            insufficient_evidence=False,
        )


@pytest.mark.asyncio
async def test_retries_only_llm_boundaries_and_saves_conversation():
    llm = FakeLLM()
    conversation = FakeConversationRepo()
    service = RAGService(llm, FakeRetriever(), conversation)

    answer, conversation_id, _ = await service.answer(ChatRequest(question="Who is the faculty?"))

    assert llm.plan_calls == 2
    assert llm.answer_calls == 1
    assert answer.grounded is True
    assert conversation.saved[0] == conversation_id


@pytest.mark.asyncio
async def test_invalid_citation_fails_closed_after_three_attempts():
    class BadLLM(FakeLLM):
        async def generate_grounded_answer(self, question, context, history):
            self.answer_calls += 1
            return GroundedAnswer(
                answer="unsupported",
                citations=[Citation(chunk_id="not-in-results", page_number=99)],
                grounded=True,
                insufficient_evidence=False,
            )

    llm = BadLLM()
    service = RAGService(llm, FakeRetriever(), FakeConversationRepo())

    answer, _, _ = await service.answer(ChatRequest(question="Who is the faculty?"))
    assert llm.answer_calls == 3
    assert answer.insufficient_evidence is True
    assert answer.grounded is False


@pytest.mark.asyncio
async def test_unsupported_number_fails_closed():
    class NumericLLM(FakeLLM):
        async def generate_grounded_answer(self, question, context, history):
            self.answer_calls += 1
            return GroundedAnswer(
                answer="Dr. Example is a professor with a 2026 salary.",
                citations=[Citation(chunk_id="c1", page_number=7)],
                grounded=True,
                insufficient_evidence=False,
            )

    llm = NumericLLM()
    service = RAGService(llm, FakeRetriever(), FakeConversationRepo())
    answer, _, _ = await service.answer(ChatRequest(question="Who is the faculty?"))
    assert llm.answer_calls == 3
    assert answer.insufficient_evidence is True
