"""Qdrant persistence with a small deterministic dummy embedding (no external model)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

COLLECTION = "saathi_patients"
VECTOR_DIM = 64
_DEFAULT_URL = "http://127.0.0.1:6333"

_client_singleton: QdrantClient | None = None


def _client() -> QdrantClient:
    """Single client per process (required for ``path=\":memory:\"``)."""
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    raw = os.environ.get("QDRANT_URL", _DEFAULT_URL).strip()
    if raw.lower() in (":memory:", "memory"):
        logger.info("qdrant.client mode=in_memory")
        _client_singleton = QdrantClient(path=":memory:")
    else:
        api_key = (os.environ.get("QDRANT_API_KEY") or "").strip() or None
        safe_url = raw.split("@")[-1]
        logger.info(
            "qdrant.client mode=remote url=%s api_key=%s",
            safe_url,
            "set" if api_key else "unset",
        )
        kwargs: dict[str, Any] = {"url": raw}
        if api_key:
            kwargs["api_key"] = api_key
        try:
            client = QdrantClient(**kwargs)
            client.get_collections()
            _client_singleton = client
        except Exception as e:
            logger.warning(
                "qdrant.client remote_failed url=%s err=%s — falling back to in-memory",
                safe_url, e,
            )
            _client_singleton = QdrantClient(path=":memory:")
    return _client_singleton


def dummy_embedding(text: str) -> list[float]:
    """Fixed-size pseudo-vector from text (stable for the same input)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out: list[float] = []
    while len(out) < VECTOR_DIM:
        for b in h:
            out.append((b / 255.0) * 2.0 - 1.0)
            if len(out) >= VECTOR_DIM:
                break
        h = hashlib.sha256(h).digest()
    return out


def create_collection() -> None:
    client = _client()
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        logger.info("qdrant.collection exists name=%s", COLLECTION)
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
    )
    logger.info("qdrant.collection created name=%s dim=%s", COLLECTION, VECTOR_DIM)


def store_patient(data: dict[str, Any]) -> str:
    """Upsert one patient point; returns ``patient_id`` (UUID string)."""
    client = _client()
    pid = str(uuid.uuid4())
    payload = dict(data)
    payload["patient_id"] = pid
    embed_key = json.dumps(payload, sort_keys=True, default=str)
    vector = dummy_embedding(embed_key)
    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(id=pid, vector=vector, payload=payload),
        ],
    )
    logger.info(
        "qdrant.store ok patient_id=%s name=%s risk=%s",
        pid,
        payload.get("patient_name"),
        payload.get("risk_level"),
    )
    return pid


def get_patient_by_id(patient_id: str) -> dict[str, Any] | None:
    client = _client()
    try:
        found = client.retrieve(
            collection_name=COLLECTION,
            ids=[patient_id],
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return None
    if not found:
        return None
    p = found[0]
    if not p.payload:
        return None
    return dict(p.payload)


def get_all_patients() -> list[dict[str, Any]]:
    client = _client()
    rows: list[dict[str, Any]] = []
    offset: str | int | None = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in batch:
            if p.payload:
                rows.append(dict(p.payload))
        if offset is None:
            break
    return rows
