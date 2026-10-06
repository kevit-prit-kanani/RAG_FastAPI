import asyncio

from app.core.config import get_settings
from app.db.indexes import create_search_indexes, list_search_indexes, wait_for_search_indexes
from app.db.mongodb import MongoDB


async def main() -> None:
    settings = get_settings()
    mongo = MongoDB(settings)
    try:
        await mongo.ping()
        print(f"MongoDB database: {settings.mongodb_db}")
        print(f"Chunks collection: {settings.mongodb_chunks_collection}")

        before = await list_search_indexes(mongo)
        print(f"Existing Atlas Search indexes: {before}")

        created = await create_search_indexes(mongo, settings)
        if created:
            print(f"Created/requested: {', '.join(created)}")
        else:
            print("No new Atlas Search indexes were needed.")

        ready = await wait_for_search_indexes(mongo, settings)
        print("Configured Atlas Search indexes:")
        for index in ready:
            print(f"- {index.get('name')}: {index.get('type', 'search')} [{index.get('status')}]")
    finally:
        await mongo.close()


if __name__ == "__main__":
    asyncio.run(main())
