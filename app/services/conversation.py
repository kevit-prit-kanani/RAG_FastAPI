from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.core.config import Settings
from app.db.mongodb import MongoDB


class ConversationRepository:
    def __init__(self, mongo: MongoDB, settings: Settings) -> None:
        self.collection = mongo.conversations
        self.settings = settings

    async def get_history(self, conversation_id: UUID | None, limit: int = 8) -> tuple[UUID, list[dict[str, str]]]:
        conversation_id = conversation_id or uuid4()
        row = await self.collection.find_one({"_id": str(conversation_id)})
        if not row:
            return conversation_id, []
        messages = row.get("messages", [])
        return conversation_id, [
            {"role": str(m["role"]), "content": str(m["content"])} for m in messages[-limit:]
        ]

    async def append(self, conversation_id: UUID, user_content: str, assistant_content: str) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"_id": str(conversation_id)},
            {
                "$setOnInsert": {"created_at": now},
                "$set": {"updated_at": now},
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": user_content, "created_at": now},
                            {"role": "assistant", "content": assistant_content, "created_at": now},
                        ]
                    }
                },
            },
            upsert=True,
        )
