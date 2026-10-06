from pymongo import AsyncMongoClient

from app.core.config import Settings


class MongoDB:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncMongoClient(settings.require_mongodb())
        self.db = self.client[settings.mongodb_db]
        self.chunks = self.db[settings.mongodb_chunks_collection]
        self.conversations = self.db[settings.mongodb_conversations_collection]

    async def ping(self) -> None:
        await self.client.admin.command("ping")

    async def close(self) -> None:
        await self.client.close()
