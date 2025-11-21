# services/support_store.py
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings

# Single embeddings instance
_embeddings = OpenAIEmbeddings()


def _build_support_embedding_text(
    title: str,
    summary: str,
    content: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_chars: int = 4000,
) -> str:
    """
    Build a single text string for embeddings from support case fields.
    NOTE: We intentionally DO NOT include 'resolution' here, since we don't
    want to search by resolution text – only by problem description / context.
    """
    parts = [
        f"Title: {title}",
        f"Summary: {summary}",
    ]
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    if metadata:
        # keep it light, just a few important fields
        doc_link = metadata.get("doc_link")
        file_name = metadata.get("file")
        if doc_link:
            parts.append(f"Doc link: {doc_link}")
        if file_name:
            parts.append(f"File: {file_name}")
    if content:
        parts.append("Conversation / Content: " + content)

    full = "\n".join(parts)
    if len(full) > max_chars:
        full = full[:max_chars]
    return full


def upsert_support_case_from_json(
    client: MongoClient,
    case_json: Dict[str, Any],
    db_name: str = "support_knowledge_base",
    collection_name: str = "support_cases",
) -> Dict[str, Any]:
    """
    Upsert a support case (support query thread) into MongoDB with an embedding.
    `case_json` is your formatted JSON like in your example.

    Recognized fields:
      - title (str)
      - summary (str)
      - content (str, optional)
      - metadata (dict, optional)
      - tags (list[str], optional)
      - resolution (str, optional)  <-- NEW
      - embedding_text (str, optional)
      - embedding (list[float], optional)
      - embedding_model (str, optional)
    """
    collection = client[db_name][collection_name]

    title: str = case_json.get("title", "")
    content: Optional[str] = case_json.get("content")
    summary: str = case_json.get("summary", "")
    metadata: Dict[str, Any] = case_json.get("metadata") or {}
    tags: List[str] = metadata.get("tags") or case_json.get("tags") or []

    # NEW: optional resolution
    resolution: Optional[str] = case_json.get("resolution")

    # Choose a stable id for the case
    case_id: str = (
        case_json.get("case_id")
        or metadata.get("file")
        or metadata.get("doc_link")
        or title
    )

    # Use provided embedding_text if present, otherwise build one
    embedding_text: str = case_json.get("embedding_text") or _build_support_embedding_text(
        title=title,
        summary=summary,
        content=content,
        tags=tags,
        metadata=metadata,
    )

    # If embedding already provided in JSON, use it; else compute it
    embedding_vector = case_json.get("embedding")
    if embedding_vector is None:
        embedding_vector = _embeddings.embed_query(embedding_text)

    now = datetime.utcnow()

    doc: Dict[str, Any] = {
        "case_id": case_id,
        "title": title,
        "content": content,
        "summary": summary,
        "metadata": metadata,
        "tags": tags,
        "resolution": resolution,           # <-- store resolution separately
        "embedding_text": embedding_text,
        "embedding": embedding_vector,
        "embedding_model": case_json.get("embedding_model", None),
        "embedding_dim": len(embedding_vector),
        "updated_at": now,
    }

    existing = collection.find_one({"case_id": case_id}, {"created_at": 1})
    if existing and "created_at" in existing:
        doc["created_at"] = existing["created_at"]
    else:
        doc["created_at"] = now

    collection.update_one({"case_id": case_id}, {"$set": doc}, upsert=True)
    return doc
