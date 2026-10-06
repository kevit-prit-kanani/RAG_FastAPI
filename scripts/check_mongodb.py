import asyncio

from app.core.config import get_settings
from app.db.indexes import list_search_indexes
from app.db.mongodb import MongoDB


async def main() -> None:
    settings = get_settings()
    mongo = MongoDB(settings)
    try:
        await mongo.ping()
        collections = await mongo.db.list_collection_names()
        regular_indexes = await mongo.chunks.index_information() if settings.mongodb_chunks_collection in collections else {}
        document_count = await mongo.chunks.count_documents({}) if settings.mongodb_chunks_collection in collections else 0
        search_indexes = await list_search_indexes(mongo) if settings.mongodb_chunks_collection in collections else []

        print("MongoDB connection: OK")
        print(f"Database: {settings.mongodb_db}")
        print(f"Chunks collection: {settings.mongodb_chunks_collection}")
        print(f"Chunks collection exists: {settings.mongodb_chunks_collection in collections}")
        print(f"Chunk documents: {document_count}")
        print(f"Regular indexes: {list(regular_indexes)}")
        print("Atlas Search indexes:")
        if not search_indexes:
            print("  NONE")
        for index in search_indexes:
            print(f"  {index.get('name')} | type={index.get('type')} | status={index.get('status')}")
    finally:
        await mongo.close()


if __name__ == "__main__":
    asyncio.run(main())
