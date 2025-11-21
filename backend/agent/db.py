# services/db.py
import os
from pymongo import MongoClient

MONGODB_URI = os.environ["MONGODB_CONNECTION_URL"]

_mongo_client: MongoClient | None = None

def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGODB_URI)
    return _mongo_client


def get_diagram_mongo_client() -> MongoClient:
    """
    Diagram KB Mongo client.
    Uses DIAGRAM_MONGODB_URI so it can be on a different cluster/DB.
    """
    uri = os.getenv("DIAGRAM_MONGODB_URI", "mongodb+srv://admin:mukulagentic@agentic.j33le.mongodb.net/?appName=agentic")
    return MongoClient(uri)