from app.models.schemas import ChatResponse, Citation
from app.services.evaluation import EvaluationCase, consistency_score, evaluate_response, overall_score
from uuid import uuid4


def test_evaluation_scores_expected_facts_and_pages():
    case = EvaluationCase(
        case_id="placement",
        question="placement",
        expected_facts=(("18.00",), ("10.93",), ("12.05",)),
        topic_terms=("placement", "salary"),
        expected_pages=(11,),
    )
    response = ChatResponse(
        conversation_id=uuid4(),
        answer="Placement salary figures are 18.00, 10.93 and 12.05.",
        citations=[Citation(chunk_id="c11", page_number=11)],
        grounded=True,
        insufficient_evidence=False,
    )
    result = evaluate_response(case, response)
    assert result.accuracy == 1.0
    assert result.citation_grounding == 1.0


def test_consistency_uses_fact_set_jaccard():
    case = EvaluationCase("x", "x", (("a",), ("b",)), ("a",), (1,))
    r1 = evaluate_response(case, ChatResponse(conversation_id=uuid4(), answer="a", citations=[], grounded=False, insufficient_evidence=True))
    r2 = evaluate_response(case, ChatResponse(conversation_id=uuid4(), answer="a b", citations=[], grounded=False, insufficient_evidence=True))
    assert consistency_score([r1, r2]) == 0.5


def test_overall_score_is_weighted():
    metrics = {"accuracy": 1.0, "relevance": 0.5, "citation_grounding": 1.0, "consistency": 0.5}
    assert abs(overall_score(metrics) - 0.85) < 1e-12
