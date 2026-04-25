"""Qdrant persistence with Gemini embeddings, payload filtering, and patient dedup."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

COLLECTION = "saathi_patients"
VECTOR_DIM_GEMINI = 256
VECTOR_DIM_FALLBACK = 256
_EMBED_MODEL = "text-embedding-004"
_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{_EMBED_MODEL}:embedContent"
_EMBED_TIMEOUT = 15
_DEFAULT_URL = "http://127.0.0.1:6333"

_client_singleton: QdrantClient | None = None
_active_vector_dim: int = VECTOR_DIM_FALLBACK


def _gemini_key() -> str | None:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None


def _client() -> QdrantClient:
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
        logger.info("qdrant.client mode=remote url=%s api_key=%s", safe_url, "set" if api_key else "unset")
        kwargs: dict[str, Any] = {"url": raw}
        if api_key:
            kwargs["api_key"] = api_key
        try:
            client = QdrantClient(**kwargs)
            client.get_collections()
            _client_singleton = client
        except Exception as e:
            logger.warning("qdrant.client remote_failed url=%s err=%s — falling back to in-memory", safe_url, e)
            _client_singleton = QdrantClient(path=":memory:")
    return _client_singleton


# ---------------------------------------------------------------------------
# Embeddings: Gemini text-embedding-004 with SHA256 fallback
# ---------------------------------------------------------------------------

def _patient_text(data: dict[str, Any]) -> str:
    """Build a human-readable summary for semantic embedding."""
    parts: list[str] = []
    if data.get("patient_name"):
        parts.append(str(data["patient_name"]))
    if data.get("age_years") is not None:
        parts.append(f"{data['age_years']} years")
    if data.get("gender"):
        parts.append(str(data["gender"]))
    if data.get("visit_type"):
        parts.append(f"visit type {data['visit_type']}")
    if data.get("pregnancy_months"):
        parts.append(f"pregnant {data['pregnancy_months']} months")
    if data.get("blood_pressure"):
        parts.append(f"BP {data['blood_pressure']}")
    if data.get("hemoglobin_g_dl") is not None:
        parts.append(f"Hb {data['hemoglobin_g_dl']}")
    if data.get("temperature_f") is not None:
        parts.append(f"temperature {data['temperature_f']}F")
    if data.get("spo2_percent") is not None:
        parts.append(f"SpO2 {data['spo2_percent']}%")
    if data.get("random_blood_sugar_mg_dl") is not None:
        parts.append(f"blood sugar {data['random_blood_sugar_mg_dl']}")
    if data.get("symptoms"):
        parts.append("symptoms: " + ", ".join(data["symptoms"]))
    if data.get("diagnosis"):
        parts.append(f"diagnosis {data['diagnosis']}")
    if data.get("emergency_signs"):
        parts.append("emergency signs: " + ", ".join(data["emergency_signs"]))
    if data.get("medicines_given"):
        parts.append("medicines: " + ", ".join(data["medicines_given"]))
    if data.get("vaccines_given"):
        parts.append("vaccines: " + ", ".join(data["vaccines_given"]))
    if data.get("referral_facility"):
        parts.append(f"referred to {data['referral_facility']}")
    if data.get("referral_reason"):
        parts.append(f"referral reason {data['referral_reason']}")
    if data.get("location"):
        parts.append(f"location {data['location']}")
    if data.get("notes"):
        parts.append(str(data["notes"]))
    return " | ".join(parts) if parts else "patient record"


def _gemini_embed(text: str, api_key: str) -> list[float] | None:
    """Call Gemini text-embedding-004. Returns None on failure (caller falls back)."""
    try:
        resp = requests.post(
            _EMBED_URL,
            params={"key": api_key},
            json={
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": VECTOR_DIM_GEMINI,
            },
            timeout=_EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        values = resp.json().get("embedding", {}).get("values")
        if isinstance(values, list) and len(values) == VECTOR_DIM_GEMINI:
            return values
        logger.warning("qdrant.embed gemini returned unexpected shape len=%s", len(values) if values else 0)
        return None
    except Exception as e:
        logger.warning("qdrant.embed gemini_failed err=%s — using fallback", e)
        return None


def _fallback_embedding(text: str) -> list[float]:
    """SHA256-based deterministic pseudo-vector (no API call needed)."""
    dim = _active_vector_dim
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out: list[float] = []
    while len(out) < dim:
        for b in h:
            out.append((b / 255.0) * 2.0 - 1.0)
            if len(out) >= dim:
                break
        h = hashlib.sha256(h).digest()
    return out


def embed(data_or_text: dict[str, Any] | str) -> list[float]:
    """
    Generate a vector for patient data or search query.
    Uses Gemini text-embedding-004 when available, otherwise SHA256 fallback.
    """
    text = data_or_text if isinstance(data_or_text, str) else _patient_text(data_or_text)
    key = _gemini_key()
    if key:
        vec = _gemini_embed(text, key)
        if vec:
            return vec
    return _fallback_embedding(text)


def embedding_mode() -> str:
    return "gemini" if _gemini_key() else "fallback"


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def create_collection() -> None:
    global _active_vector_dim
    client = _client()

    _active_vector_dim = VECTOR_DIM_GEMINI if _gemini_key() else VECTOR_DIM_FALLBACK
    logger.info("qdrant.collection target dim=%s embedding=%s", _active_vector_dim, embedding_mode())

    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        info = client.get_collection(COLLECTION)
        current_dim = info.config.params.vectors.size  # type: ignore[union-attr]
        if current_dim == _active_vector_dim:
            logger.info("qdrant.collection exists name=%s dim=%s — ok", COLLECTION, current_dim)
            _create_payload_indexes(client)
            return
        logger.warning(
            "qdrant.collection dim_mismatch current=%s target=%s — recreating",
            current_dim, _active_vector_dim,
        )
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=_active_vector_dim, distance=models.Distance.COSINE),
    )
    logger.info("qdrant.collection created name=%s dim=%s", COLLECTION, _active_vector_dim)
    _create_payload_indexes(client)


def _create_payload_indexes(client: QdrantClient) -> None:
    """Create payload indexes for efficient filtering (idempotent)."""
    indexes = [
        ("risk_level", models.PayloadSchemaType.KEYWORD),
        ("patient_name", models.PayloadSchemaType.KEYWORD),
        ("visit_type", models.PayloadSchemaType.KEYWORD),
        ("location", models.PayloadSchemaType.KEYWORD),
        ("seeded_demo", models.PayloadSchemaType.BOOL),
        # Qdrant Cloud requires an index before filtering on this field (get_emergencies).
        ("emergency.is_emergency", models.PayloadSchemaType.BOOL),
    ]
    for field, schema in indexes:
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        except Exception as e:
            # already exists, or local in-memory engine may ignore some index types
            logger.debug("qdrant.payload_index skip field=%s err=%s", field, e)


# ---------------------------------------------------------------------------
# Patient deduplication & upsert
# ---------------------------------------------------------------------------

def _normalize_name(name: str | None) -> str:
    if not name:
        return ""
    return " ".join(name.strip().lower().split())


def _find_existing_patient(name: str, age: Any) -> tuple[str, dict[str, Any]] | None:
    """Find an existing patient by exact name match + age. Returns (point_id, payload) or None."""
    norm = _normalize_name(name)
    if not norm:
        return None

    client = _client()
    try:
        results, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="patient_name",
                        match=models.MatchValue(value=name.strip()),
                    ),
                ]
            ),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.debug("qdrant.find_existing scroll failed err=%s", e)
        return None

    for point in results:
        if not point.payload:
            continue
        existing_age = point.payload.get("age_years")
        if age is not None and existing_age is not None:
            try:
                if abs(float(age) - float(existing_age)) > 2:
                    continue
            except (TypeError, ValueError):
                pass
        pid = str(point.id)
        return pid, dict(point.payload)

    return None


def store_patient(data: dict[str, Any]) -> str:
    """
    Smart upsert: if a patient with the same name+age exists, update that record
    and append to visit_history. Otherwise create a new point.
    """
    client = _client()
    name = data.get("patient_name")
    age = data.get("age_years")
    now = datetime.now(timezone.utc).isoformat()

    existing = _find_existing_patient(name, age) if name else None

    if existing:
        pid, old_payload = existing
        history: list[dict[str, Any]] = old_payload.get("visit_history", [])
        snapshot = {k: v for k, v in old_payload.items() if k not in ("visit_history", "patient_id")}
        snapshot["recorded_at"] = old_payload.get("last_updated", now)
        history.append(snapshot)

        merged = dict(old_payload)
        if old_payload.get("seeded_demo"):
            merged["seeded_demo"] = True
        # Preserve who originally registered the patient (first writer wins).
        # Later visits should not overwrite registration ownership.
        _registration_keys = {
            "registered_by_anm_id",
            "registered_by_anm_name",
            "registered_by_anm_whatsapp",
        }
        for k, v in data.items():
            if v is None:
                continue
            if k in _registration_keys:
                existing_v = merged.get(k)
                if existing_v is not None and str(existing_v).strip() != "":
                    continue
            merged[k] = v
        merged["visit_history"] = history
        merged["visit_count"] = len(history) + 1
        merged["last_updated"] = now
        merged["patient_id"] = pid

        vector = embed(merged)
        client.upsert(
            collection_name=COLLECTION,
            points=[models.PointStruct(id=pid, vector=vector, payload=merged)],
        )
        logger.info(
            "qdrant.update ok patient_id=%s name=%s visits=%s risk=%s",
            pid, merged.get("patient_name"), merged.get("visit_count"), merged.get("risk_level"),
        )
        return pid

    pid = str(uuid.uuid4())
    payload = dict(data)
    payload["patient_id"] = pid
    payload["visit_count"] = 1
    payload["visit_history"] = []
    payload["last_updated"] = now

    vector = embed(payload)
    client.upsert(
        collection_name=COLLECTION,
        points=[models.PointStruct(id=pid, vector=vector, payload=payload)],
    )
    logger.info(
        "qdrant.store ok patient_id=%s name=%s risk=%s embedding=%s",
        pid, payload.get("patient_name"), payload.get("risk_level"), embedding_mode(),
    )
    return pid


def update_patient_fields(patient_id: str, updates: dict[str, Any]) -> bool:
    """Merge `updates` into an existing point by id and re-upsert (same vector id)."""
    row = get_patient_by_id(patient_id)
    if row is None:
        return False
    merged = dict(row)
    for k, v in updates.items():
        if v is not None:
            merged[k] = v
    merged["patient_id"] = patient_id
    client = _client()
    vector = embed(merged)
    client.upsert(
        collection_name=COLLECTION,
        points=[models.PointStruct(id=patient_id, vector=vector, payload=merged)],
    )
    logger.info("qdrant.patch ok patient_id=%s keys=%s", patient_id, list(updates.keys()))
    return True


# ---------------------------------------------------------------------------
# Retrieval with Qdrant-native filtering
# ---------------------------------------------------------------------------

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
    if not found or not found[0].payload:
        return None
    return dict(found[0].payload)


def _demo_exclude_condition() -> models.FieldCondition:
    return models.FieldCondition(key="seeded_demo", match=models.MatchValue(value=True))


def _scroll_all(
    filt: models.Filter | None = None,
    *,
    include_demo: bool = False,
) -> list[dict[str, Any]]:
    """Paginated scroll with optional Qdrant-native filter.

    When *include_demo* is False (default), patients with
    ``seeded_demo=True`` are excluded at the DB level.
    """
    client = _client()

    if not include_demo:
        exclude = _demo_exclude_condition()
        if filt is None:
            filt = models.Filter(must_not=[exclude])
        else:
            existing_must_not = list(filt.must_not or [])
            existing_must_not.append(exclude)
            filt = models.Filter(
                must=filt.must,
                should=filt.should,
                must_not=existing_must_not,
            )

    rows: list[dict[str, Any]] = []
    offset: str | int | None = None
    while True:
        kwargs: dict[str, Any] = {
            "collection_name": COLLECTION,
            "limit": 128,
            "offset": offset,
            "with_payload": True,
            "with_vectors": False,
        }
        if filt:
            kwargs["scroll_filter"] = filt
        batch, offset = client.scroll(**kwargs)
        for p in batch:
            if p.payload:
                rows.append(dict(p.payload))
        if offset is None:
            break
    return rows


def get_all_patients(*, include_demo: bool = False) -> list[dict[str, Any]]:
    return _scroll_all(include_demo=include_demo)


def get_patients_by_risk(level: str, *, include_demo: bool = False) -> list[dict[str, Any]]:
    """DB-level filter by risk_level (no Python filtering)."""
    return _scroll_all(
        models.Filter(must=[
            models.FieldCondition(key="risk_level", match=models.MatchValue(value=level)),
        ]),
        include_demo=include_demo,
    )


def get_emergencies(*, include_demo: bool = False) -> list[dict[str, Any]]:
    """All patients where emergency.is_emergency is true (filtered at DB level)."""
    results = _scroll_all(
        models.Filter(must=[
            models.FieldCondition(
                key="emergency.is_emergency",
                match=models.MatchValue(value=True),
            ),
        ]),
        include_demo=include_demo,
    )
    results.sort(key=lambda x: 0 if (x.get("emergency") or {}).get("severity") == "critical" else 1)
    return results


def get_patients_by_visit_type(visit_type: str) -> list[dict[str, Any]]:
    return _scroll_all(
        models.Filter(must=[
            models.FieldCondition(key="visit_type", match=models.MatchValue(value=visit_type)),
        ])
    )


def get_patients_by_location(location: str) -> list[dict[str, Any]]:
    return _scroll_all(
        models.Filter(must=[
            models.FieldCondition(key="location", match=models.MatchValue(value=location)),
        ])
    )


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[\w\u0900-\u0fff']+", re.UNICODE)


def _search_query_for_embedding(query: str) -> str:
    """
    Enrich the raw user phrase so the embedding model aligns with how patient
    cards are vectorized (symptoms, vitals, visit context).
    """
    q = (query or "").strip()
    if not q:
        return "patient health record"
    return (
        "Match ANM/OPD health records: patient name, vitals, symptoms, diagnosis, "
        f"emergency signs, pregnancy, immunization, location. Search query: {q}"
    )


def _lexical_relevance(query: str, row: dict[str, Any]) -> float:
    """Token overlap between query and the same text shape used for patient vectors."""
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    q_tokens = set(t for t in _TOKEN_RE.findall(q) if len(t) > 1)
    if not q_tokens:
        return 0.0
    doc = _patient_text(row).lower()
    doc_tokens = set(t for t in _TOKEN_RE.findall(doc) if len(t) > 1)
    if not doc_tokens:
        return 0.0
    inter = q_tokens & doc_tokens
    return len(inter) / max(len(q_tokens), 1)


def _vector_score_norm(raw: float) -> float:
    """Qdrant cosine scores are typically in [0,1]; keep bounded."""
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if x < 0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _combined_rank_score(vscore: float, lscore: float) -> float:
    return 0.58 * vscore + 0.42 * lscore


def delete_seeded_demo_points() -> int:
    """
    Remove all points with ``seeded_demo`` flag (from Load demo data).
    Returns number of points deleted (best-effort; 0 on empty / error).
    """
    client = _client()
    flt = models.Filter(
        must=[models.FieldCondition(key="seeded_demo", match=models.MatchValue(value=True))],
    )
    try:
        before, _ = client.scroll(collection_name=COLLECTION, scroll_filter=flt, limit=5000, with_payload=False)
        n = len(before)
        if n == 0:
            return 0
        client.delete(collection_name=COLLECTION, points_selector=flt, wait=True)
        logger.info("qdrant.delete_seeded_demo removed=%s", n)
        return n
    except Exception as e:
        logger.warning("qdrant.delete_seeded_demo failed err=%s", e)
        return 0


def search_similar(query: str, limit: int = 10, *, include_demo: bool = False) -> list[dict[str, Any]]:
    """
    Hybrid semantic search: dense vectors (Gemini) + lexical rerank on patient text.
    Pulls a wider HNSW candidate set, then reorders for phrase/token alignment.
    """
    client = _client()
    qraw = (query or "").strip()
    embed_text = _search_query_for_embedding(qraw) if qraw else ""
    vector = embed(embed_text or "patient")
    fetch_max = min(100, max(limit * 4, max(16, limit)))
    query_filter: models.Filter | None = None
    if not include_demo:
        query_filter = models.Filter(must_not=[_demo_exclude_condition()])
    try:
        qp_kwargs: dict[str, Any] = {
            "collection_name": COLLECTION,
            "query": vector,
            "limit": fetch_max,
            "with_payload": True,
            "search_params": models.SearchParams(hnsw_ef=max(64, min(200, limit * 12))),
        }
        if query_filter:
            qp_kwargs["query_filter"] = query_filter
        resp = client.query_points(**qp_kwargs)
        hits = resp.points if hasattr(resp, "points") else resp
    except Exception as e:
        logger.warning("qdrant.search failed err=%s", e)
        return []

    ranked: list[tuple[float, float, float, dict[str, Any]]] = []
    for hit in hits:
        if not hit.payload:
            continue
        row = dict(hit.payload)
        vsc = _vector_score_norm(getattr(hit, "score", 0) or 0)
        lsc = _lexical_relevance(qraw, row) if qraw else 0.0
        comb = _combined_rank_score(vsc, lsc)
        ranked.append((comb, vsc, lsc, row))
    ranked.sort(key=lambda t: t[0], reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for comb, vsc, lsc, row in ranked:
        if len(out) >= limit:
            break
        pid = str(row.get("patient_id") or "")
        if pid and pid in seen:
            continue
        if pid:
            seen.add(pid)
        row["_similarity_score"] = round(comb, 4)
        row["_vector_score"] = round(vsc, 4)
        row["_lexical_score"] = round(lsc, 4)
        out.append(row)
    return out
