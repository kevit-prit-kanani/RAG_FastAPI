from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.models.schemas import ChatResponse


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_facts: tuple[tuple[str, ...], ...]
    topic_terms: tuple[str, ...]
    expected_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_id: str
    accuracy: float
    relevance: float
    citation_grounding: float
    detected_fact_ids: tuple[int, ...]
    cited_pages: tuple[int, ...]
    grounded: bool
    insufficient_evidence: bool


def normalize(text: str) -> str:
    text = text.casefold().replace("₹", "rs ")
    text = re.sub(r"[^a-z0-9.%+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_fact(answer: str, alternatives: Iterable[str]) -> bool:
    normalized = normalize(answer)
    return any(normalize(phrase) in normalized for phrase in alternatives)


def evaluate_response(case: EvaluationCase, response: ChatResponse) -> EvaluationResult:
    detected = tuple(
        index for index, alternatives in enumerate(case.expected_facts)
        if contains_fact(response.answer, alternatives)
    )
    accuracy = len(detected) / len(case.expected_facts) if case.expected_facts else 0.0

    normalized_answer = normalize(response.answer)
    topic_hits = sum(1 for term in case.topic_terms if normalize(term) in normalized_answer)
    relevance = topic_hits / len(case.topic_terms) if case.topic_terms else 1.0

    expected_pages = set(case.expected_pages)
    cited_pages = tuple(sorted({citation.page_number for citation in response.citations}))
    citation_grounding = 1.0 if expected_pages.intersection(cited_pages) else 0.0
    if response.insufficient_evidence and not response.citations:
        citation_grounding = 1.0

    return EvaluationResult(
        case_id=case.case_id,
        accuracy=accuracy,
        relevance=relevance,
        citation_grounding=citation_grounding,
        detected_fact_ids=detected,
        cited_pages=cited_pages,
        grounded=response.grounded,
        insufficient_evidence=response.insufficient_evidence,
    )


def consistency_score(results: list[EvaluationResult]) -> float:
    """Fact-set consistency across repeated runs; 1.0 means identical fact coverage."""
    if len(results) < 2:
        return 1.0
    sets = [set(result.detected_fact_ids) for result in results]
    pair_scores: list[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            left = sets[left_index]
            right = sets[right_index]
            union = left | right
            pair_scores.append(1.0 if not union else len(left & right) / len(union))
    return sum(pair_scores) / len(pair_scores) if pair_scores else 1.0


def aggregate(results: list[EvaluationResult], consistency: float) -> dict[str, float]:
    if not results:
        return {"accuracy": 0.0, "relevance": 0.0, "citation_grounding": 0.0, "consistency": consistency}
    return {
        "accuracy": sum(item.accuracy for item in results) / len(results),
        "relevance": sum(item.relevance for item in results) / len(results),
        "citation_grounding": sum(item.citation_grounding for item in results) / len(results),
        "consistency": consistency,
    }


def overall_score(metrics: dict[str, float]) -> float:
    return (
        0.55 * metrics["accuracy"]
        + 0.20 * metrics["relevance"]
        + 0.15 * metrics["citation_grounding"]
        + 0.10 * metrics["consistency"]
    )
