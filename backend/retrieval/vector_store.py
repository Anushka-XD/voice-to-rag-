"""
Qdrant-backed vector store.

Local embedded Qdrant (path) for development; optional URL + API key for a server.
Distance defaults to Cosine because E5 embeddings are L2-normalized.
Vector size is taken from the embedding model, never hard-coded.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.config import VectorStoreConfig
from backend.ingestion.schemas import ChunkRecord

logger = logging.getLogger(__name__)

# Stable namespace so chunk_id → UUID is deterministic across runs.
_CHUNK_ID_NS = uuid.UUID("7c3f1a90-4d2e-5b68-a1c4-9e0d8f7b6a52")


def point_id_for_chunk(chunk_id: str) -> str:
    """Qdrant requires UUID or unsigned int; we hash the existing chunk_id."""
    return str(uuid.uuid5(_CHUNK_ID_NS, chunk_id))


def chunk_payload(chunk: ChunkRecord) -> dict[str, Any]:
    """Only fields that exist on ChunkRecord / verified IR."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "passage_id": chunk.passage_id,
        "query_id": chunk.query_id,
        "language": chunk.language,
        "language_flores": chunk.language_flores,
        "chunk_strategy": chunk.chunk_strategy,
        "chunk_index": chunk.chunk_index,
        "start_sentence": chunk.start_sentence,
        "end_sentence": chunk.end_sentence,
        "text": chunk.text,
        "source": chunk.source,
        "passage_index": chunk.passage_index,
        "passage_source": chunk.passage_source,
        "is_selected": chunk.is_selected,
        "adaptive_reason": chunk.adaptive_reason,
    }


class QdrantVectorStore:
    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        client: QdrantClient | None = None,
        *,
        in_memory: bool = False,
    ) -> None:
        self.config = config or VectorStoreConfig.from_env()
        if client is not None:
            self.client = client
        elif in_memory:
            self.client = QdrantClient(":memory:")
        elif self.config.url:
            self.client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                prefer_grpc=self.config.prefer_grpc,
            )
        else:
            self.config.path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.config.path))

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection_name))

    def create_collection(self, vector_size: int, *, recreate: bool = False) -> None:
        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection_name)
        if self.collection_exists():
            info = self.client.get_collection(self.collection_name)
            existing = _collection_vector_size(info)
            if existing is not None and int(existing) != int(vector_size):
                raise ValueError(
                    f"Collection {self.collection_name!r} has dim={existing}, "
                    f"embedder has dim={vector_size}. Rebuild with --mode rebuild."
                )
            return
        distance = _distance(self.config.distance)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=int(vector_size), distance=distance),
        )
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="language",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create language payload index: %s", exc)
        logger.info(
            "Created collection %s dim=%s distance=%s",
            self.collection_name,
            vector_size,
            self.config.distance,
        )

    def delete_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

    def upsert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: np.ndarray,
        *,
        batch_size: int | None = None,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if len(chunks) == 0:
            return 0
        bs = batch_size or self.config.upsert_batch_size
        n = 0
        for start in range(0, len(chunks), bs):
            batch_c = chunks[start : start + bs]
            batch_v = vectors[start : start + bs]
            points = [
                qmodels.PointStruct(
                    id=point_id_for_chunk(c.chunk_id),
                    vector=batch_v[i].astype(np.float32, copy=False).tolist(),
                    payload=chunk_payload(c),
                )
                for i, c in enumerate(batch_c)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            n += len(points)
        return n

    def search(
        self,
        query_vector: np.ndarray | Sequence[float],
        *,
        top_k: int = 10,
        language_filter: str | Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        vec = np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist()
        query_filter = _language_filter(language_filter)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            limit=int(top_k),
            query_filter=query_filter,
            with_payload=True,
        )
        hits = getattr(response, "points", response)
        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            results.append(
                {
                    "id": hit.id,
                    "score": float(hit.score),
                    "payload": payload,
                }
            )
        return results

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return int(self.client.count(self.collection_name, exact=True).count)

    def stats(self) -> dict[str, Any]:
        if not self.collection_exists():
            return {"exists": False, "collection": self.collection_name, "points": 0}
        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "collection": self.collection_name,
            "points": self.count(),
            "vector_size": _collection_vector_size(info),
            "distance": self.config.distance,
        }


def _distance(name: str) -> qmodels.Distance:
    mapping = {
        "cosine": qmodels.Distance.COSINE,
        "dot": qmodels.Distance.DOT,
        "euclid": qmodels.Distance.EUCLID,
        "euclidean": qmodels.Distance.EUCLID,
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown distance {name!r}")
    return mapping[key]


def _collection_vector_size(info: Any) -> int | None:
    cfg = getattr(info, "config", None)
    params = getattr(cfg, "params", None) if cfg else None
    vectors = getattr(params, "vectors", None) if params else None
    size = getattr(vectors, "size", None)
    if size is not None:
        return int(size)
    if isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        return int(getattr(first, "size", 0) or 0)
    return None


def _language_filter(language_filter: str | Sequence[str] | None) -> qmodels.Filter | None:
    if language_filter is None:
        return None
    langs = [language_filter] if isinstance(language_filter, str) else list(language_filter)
    langs = [str(x) for x in langs if x]
    if not langs:
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="language",
                match=qmodels.MatchAny(any=langs) if len(langs) > 1 else qmodels.MatchValue(value=langs[0]),
            )
        ]
    )
