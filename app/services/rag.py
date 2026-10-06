from __future__ import annotations

import re
from uuid import UUID

from app.models.schemas import ChatRequest, GroundedAnswer, RetrievedChunk, SearchRequest, SearchToolArgs

MAX_LLM_RETRIES = 3
SAFE_ABSTENTION = GroundedAnswer(
    answer="I do not have enough evidence in the uploaded documents to answer that reliably.",
    citations=[],
    grounded=False,
    insufficient_evidence=True,
)


class RAGService:
    def __init__(self, llm, retriever, conversation_repository) -> None:
        self.llm = llm
        self.retriever = retriever
        self.conversation_repository = conversation_repository

    async def answer(self, request: ChatRequest) -> tuple[GroundedAnswer, "UUID", list[RetrievedChunk]]:
        conversation_id, history = await self.conversation_repository.get_history(request.conversation_id)
        plan = await self._plan_with_retries(request.question, history)
        plan = _normalize_search_plan(request.question, plan, request.top_k)

        search_request = SearchRequest(
            query=request.question,
            alternate_queries=[q for q in plan.queries if q.casefold() != request.question.casefold()],
            keywords=plan.keywords,
            top_k=plan.top_k,
            filters=plan.filters,
        )
        retrieved = await self.retriever.search(search_request)

        # Deterministic abstention before generation is the strongest guard against
        # hallucination when retrieval has no usable evidence.
        if not retrieved:
            answer = SAFE_ABSTENTION
        else:
            context = _format_context(retrieved)
            answer = await self._answer_with_retries(request.question, context, history, retrieved)

        await self.conversation_repository.append(conversation_id, request.question, answer.answer)
        return answer, conversation_id, retrieved

    async def _plan_with_retries(self, question: str, history: list[dict[str, str]]) -> SearchToolArgs:
        last_error: Exception | None = None
        for _attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                return await self.llm.plan_search(question, history)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Search planning failed after {MAX_LLM_RETRIES} attempts: {last_error}") from last_error

    async def _answer_with_retries(
        self,
        question: str,
        context: str,
        history: list[dict[str, str]],
        retrieved: list[RetrievedChunk],
    ) -> GroundedAnswer:
        last_error: Exception | None = None
        valid_chunk_ids = {chunk.chunk_id for chunk in retrieved}
        valid_pages = {chunk.chunk_id: chunk.page_number for chunk in retrieved}
        cited_context = {chunk.chunk_id: chunk.content for chunk in retrieved}

        for _attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                answer = await self.llm.generate_grounded_answer(question, context, history)
                _validate_grounding(answer, valid_chunk_ids, valid_pages, cited_context)
                return answer
            except Exception as exc:
                last_error = exc

        # Fail closed: after the bounded retry budget, return a safe abstention rather
        # than surfacing a hallucinated answer or forcing the API caller to retry.
        return SAFE_ABSTENTION


def _normalize_search_plan(question: str, plan: SearchToolArgs, requested_top_k: int) -> SearchToolArgs:
    queries = _dedupe([question, *plan.queries])[:4]
    keywords = _dedupe(plan.keywords)[:12]
    return plan.model_copy(update={"queries": queries, "keywords": keywords, "top_k": min(plan.top_k, requested_top_k)})


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(value.split()).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[CHUNK {index}: {chunk.chunk_id} | page {chunk.page_number} | "
        f"section: {' > '.join(chunk.section_path) or '[unspecified]'}]\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )


def _validate_grounding(
    answer: GroundedAnswer,
    valid_chunk_ids: set[str],
    valid_pages: dict[str, int],
    cited_context: dict[str, str],
) -> None:
    for citation in answer.citations:
        if citation.chunk_id not in valid_chunk_ids:
            raise ValueError(f"Citation references a chunk not present in retrieval: {citation.chunk_id}")
        if citation.page_number != valid_pages[citation.chunk_id]:
            raise ValueError(f"Citation page mismatch for {citation.chunk_id}")

    if answer.grounded and not answer.citations:
        raise ValueError("Grounded answer must include at least one citation")
    if answer.insufficient_evidence and answer.grounded:
        raise ValueError("Answer cannot be both grounded and insufficient_evidence")
    if not answer.grounded:
        return

    evidence_text = " ".join(cited_context[citation.chunk_id] for citation in answer.citations)
    _validate_numeric_grounding(answer.answer, evidence_text)
    _validate_text_overlap(answer.answer, evidence_text)


def _validate_numeric_grounding(answer: str, evidence: str) -> None:
    """Do not allow numeric values that are absent from retrieved evidence."""
    answer_numbers = {token for token in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:-\d+)?(?![A-Za-z])", answer)}
    evidence_numbers = set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:-\d+)?(?![A-Za-z])", evidence))
    unknown = sorted(answer_numbers - evidence_numbers)
    if unknown:
        raise ValueError(f"Answer contains unsupported numeric value(s): {', '.join(unknown)}")


def _validate_text_overlap(answer: str, evidence: str) -> None:
    """Lightweight claim-support gate; avoids a second LLM grader and fails closed only on weak overlap."""
    evidence_tokens = set(re.findall(r"[a-z0-9]{4,}", evidence.casefold()))
    sentences = [part.strip() for part in re.split(r"[.!?]\s+|\n+", answer) if part.strip()]
    for sentence in sentences:
        tokens = set(re.findall(r"[a-z0-9]{4,}", sentence.casefold()))
        if tokens and not (tokens & evidence_tokens):
            raise ValueError("Answer sentence has no meaningful lexical support in cited evidence")
