# services/doc_store.py
from datetime import datetime
from typing import List, Optional, Dict, Any

from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings


# Single embeddings instance for reuse
_embeddings = OpenAIEmbeddings()


def _build_embedding_text(
    title: str,
    description: str,
    tags: Optional[List[str]] = None,
    content: Optional[str] = None,
    max_chars: int = 4000,
) -> str:
    """
    Build a single text string that will be used for embedding.
    """
    parts = [
        f"Title: {title}",
        f"Description: {description}",
    ]
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    if content:
        parts.append("Content: " + content)

    full_text = "\n".join(parts)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
    return full_text


def upsert_doc(
    client: MongoClient,
    doc_id: str,
    title: str,
    url: str,
    description: str,
    tags: Optional[List[str]] = None,
    content: Optional[str] = None,
    db_name: str = "knowledge_base",
    collection_name: str = "docs",
) -> Dict[str, Any]:
    """
    Insert or update a document into the knowledge_base.docs collection,
    including an embedding for vector search.
    """
    collection = client[db_name][collection_name]

    embedding_text = _build_embedding_text(
        title=title,
        description=description,
        tags=tags,
        content=content,
    )

    # FIXED: use _embeddings (not __embeddings)
    embedding_vector = _embeddings.embed_query(embedding_text)

    now = datetime.utcnow()

    doc = {
        "doc_id": doc_id,
        "title": title,
        "url": url,
        "description": description,
        "tags": tags or [],
        "content": content,
        "embedding_text": embedding_text,
        "embedding": embedding_vector,
        "updated_at": now,
    }

    # Preserve created_at on update
    existing = collection.find_one({"doc_id": doc_id}, {"created_at": 1})
    if existing and "created_at" in existing:
        doc["created_at"] = existing["created_at"]
    else:
        doc["created_at"] = now

    # Upsert the document
    collection.update_one(
        {"doc_id": doc_id},
        {"$set": doc},
        upsert=True,
    )

    return doc
