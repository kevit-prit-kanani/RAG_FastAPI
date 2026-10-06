from __future__ import annotations

import argparse
import asyncio
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from app.models.schemas import ChatResponse
from app.services.evaluation import EvaluationCase, aggregate, consistency_score, evaluate_response, overall_score
CASES_FILE = ROOT / "evaluation" / "questions.json"
RESULTS_DIR = ROOT / "evaluation" / "results"


def load_cases() -> list[EvaluationCase]:
    raw = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            case_id=item["case_id"],
            question=item["question"],
            expected_facts=tuple(tuple(fact) for fact in item["expected_facts"]),
            topic_terms=tuple(item["topic_terms"]),
            expected_pages=tuple(item["expected_pages"]),
        )
        for item in raw
    ]


def result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "accuracy": result.accuracy,
        "relevance": result.relevance,
        "citation_grounding": result.citation_grounding,
        "detected_fact_ids": list(result.detected_fact_ids),
        "cited_pages": list(result.cited_pages),
        "grounded": result.grounded,
        "insufficient_evidence": result.insufficient_evidence,
    }


async def post_chat(base_url: str, question: str) -> ChatResponse:
    payload = json.dumps({"question": question, "conversation_id": None, "top_k": 8}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def call() -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST /chat failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {base_url}: {exc}") from exc

    body = json.loads(await asyncio.to_thread(call))
    return ChatResponse.model_validate(body)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG API against the provided brochure questions.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeats", type=int, default=2, choices=range(1, 6))
    args = parser.parse_args()

    cases = load_cases()
    report: dict[str, Any] = {"base_url": args.base_url, "repeats": args.repeats, "cases": []}

    for case in cases:
        run_results = []
        responses = []
        for _ in range(args.repeats):
            response = await post_chat(args.base_url, case.question)
            responses.append(response.model_dump(mode="json"))
            run_results.append(evaluate_response(case, response))

        consistency = consistency_score(run_results)
        metrics = aggregate(run_results, consistency)
        report["cases"].append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "metrics": metrics,
                "overall": overall_score(metrics),
                "runs": [result_to_dict(result) for result in run_results],
                "responses": responses,
            }
        )

    totals = {
        metric: sum(item["metrics"][metric] for item in report["cases"]) / len(report["cases"])
        for metric in ("accuracy", "relevance", "citation_grounding", "consistency")
    }
    report["aggregate"] = {"metrics": totals, "overall": overall_score(totals)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "latest.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Saved detailed report to {output}")


if __name__ == "__main__":
    asyncio.run(main())
