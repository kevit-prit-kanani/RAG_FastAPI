from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings
from app.models.schemas import GroundedAnswer, SearchToolArgs


SEARCH_TOOL_NAME = "search_knowledge_base"

SEARCH_TOOL_DESCRIPTION = (
    "Search the uploaded document collection for evidence needed to answer the user's question. "
    "Create up to four focused queries when the question has multiple parts, and include exact "
    "terms, names, dates, or numeric labels in keywords. Do not answer the question yourself."
)


class OpenAIClient:
    """Thin provider adapter. This is the boundary we can replace with LiteLLM later."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.require_openai())

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=texts,
            dimensions=self.settings.openai_embedding_dimensions,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}")
        for index, vector in enumerate(vectors):
            if len(vector) != self.settings.openai_embedding_dimensions:
                raise RuntimeError(
                    f"Embedding dimension mismatch at index {index}: "
                    f"expected {self.settings.openai_embedding_dimensions}, got {len(vector)}"
                )
        return vectors

    async def plan_search(self, question: str, history: list[dict[str, str]]) -> SearchToolArgs:
        tool = {
            "type": "function",
            "name": SEARCH_TOOL_NAME,
            "description": SEARCH_TOOL_DESCRIPTION,
            "parameters": _strict_schema(SearchToolArgs),
            "strict": True,
        }

        history_text = _history_text(history)
        response = await self.client.responses.create(
            model=self.settings.openai_chat_model,
            instructions=(
                "You are the retrieval planner for a document-grounded RAG system. "
                "You have access to exactly one tool: search_knowledge_base. "
                "Do not answer the user's question. Always call the tool. "
                "Use the conversation history only to resolve references such as 'that program' or 'it'. "
                "For multi-part questions, treat each distinct fact set as an evidence hop and create a focused query for it. "
                "Reformulate ambiguous wording into document terminology, and include exact names, dates, years, or numeric labels as keywords. "
                "The tool must contain focused retrieval queries and useful exact keywords."
            ),
            input=[
                {
                    "role": "user",
                    "content": f"Conversation history:\n{history_text or '[none]'}\n\nCurrent question:\n{question}",
                }
            ],
            tools=[tool],
            tool_choice={"type": "function", "name": SEARCH_TOOL_NAME},
            parallel_tool_calls=False,
        )

        calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if len(calls) != 1:
            raise ValueError(f"Expected exactly one {SEARCH_TOOL_NAME} call; got {len(calls)}")
        call = calls[0]
        return SearchToolArgs.model_validate_json(call.arguments)

    async def generate_grounded_answer(
        self,
        question: str,
        retrieved_context: str,
        history: list[dict[str, str]],
    ) -> GroundedAnswer:
        history_text = _history_text(history)
        prompt = (
            f"Conversation history:\n{history_text or '[none]'}\n\n"
            f"Current question:\n{question}\n\n"
            f"Retrieved evidence:\n{retrieved_context or '[NO EVIDENCE]'}"
        )

        response = await self.client.responses.parse(
            model=self.settings.openai_chat_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are the final answer generator for a document-grounded RAG system. "
                        "Answer only from the retrieved evidence. Do not use outside knowledge. "
                        "Every factual claim must be supported by one or more supplied chunks. "
                        "Use exact numbers and labels from the evidence; never invent or 'correct' OCR. "
                        "Citations must reference only supplied chunk IDs and page numbers. "
                        "When evidence is insufficient, say so explicitly and set insufficient_evidence=true. "
                        "For multi-hop questions, combine only the supported evidence from the retrieved chunks; do not infer an unsupported link between facts. "
                        "When you can answer from evidence, set grounded=true and provide citations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            text_format=GroundedAnswer,
        )

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError("Final model response did not contain parsed GroundedAnswer output")
        if not isinstance(parsed, GroundedAnswer):
            parsed = GroundedAnswer.model_validate(parsed)
        return parsed


def _strict_schema(model: type[Any]) -> dict[str, Any]:
    schema = model.model_json_schema()
    _make_strict(schema)
    return schema


def _make_strict(node: dict[str, Any]) -> None:
    if node.get("type") == "object":
        properties = node.get("properties", {})
        node["required"] = list(properties.keys())
        node["additionalProperties"] = False
        for child in properties.values():
            if isinstance(child, dict):
                _make_strict(child)
    for key in ("items",):
        child = node.get(key)
        if isinstance(child, dict):
            _make_strict(child)
    for definition in node.get("$defs", {}).values():
        if isinstance(definition, dict):
            _make_strict(definition)


def _history_text(history: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in history[-8:])
