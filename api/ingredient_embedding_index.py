from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from api.config import ApiSettings


logger = logging.getLogger(__name__)


class IngredientEmbeddingIndex:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._records: Optional[List[dict]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._model = None
        self._model_load_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.settings.ingredient_embedding_enabled

    def _resolve_metadata_path(self) -> Optional[Path]:
        preferred = self.settings.ingredient_embedding_metadata_path
        if preferred.exists():
            return preferred
        fallback = self.settings.ingredient_embedding_documents_path
        if fallback.exists():
            return fallback
        return None

    def _load_records(self) -> List[dict]:
        if self._records is not None:
            return self._records
        metadata_path = self._resolve_metadata_path()
        if not metadata_path:
            logger.warning("Ingredient embedding metadata source not found")
            self._records = []
            return self._records

        records: List[dict] = []
        if metadata_path.suffix.lower() == ".pkl":
            with metadata_path.open("rb") as handle:
                loaded = pickle.load(handle)
            if isinstance(loaded, list):
                for item in loaded:
                    if isinstance(item, dict):
                        records.append(dict(item))
        else:
            frame = pd.read_csv(metadata_path, encoding="utf-8-sig", low_memory=False).fillna("")
            records = frame.to_dict(orient="records")

        normalized: List[dict] = []
        for item in records:
            standard_name = str(item.get("standard_name", "") or "").strip()
            if not standard_name:
                continue
            normalized.append(
                {
                    "id": str(item.get("id", "") or f"functional_ingredient::{standard_name}"),
                    "standard_name": standard_name,
                    "synonyms_joined": str(item.get("synonyms_joined", "") or ""),
                    "search_text": str(item.get("search_text", "") or ""),
                    "sources_joined": str(item.get("sources_joined", "") or ""),
                    "function_text": str(item.get("function_text", "") or ""),
                    "caution_text": str(item.get("caution_text", "") or ""),
                    "embedding_text": str(item.get("embedding_text", "") or item.get("search_text", "") or ""),
                    "canonical_priority": float(item.get("canonical_priority", 0.0) or 0.0),
                    "specific_penalty": float(item.get("specific_penalty", 0.0) or 0.0),
                }
            )
        self._records = normalized
        return self._records

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._model_load_error:
            raise RuntimeError(self._model_load_error)
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - dependency/runtime issue
            self._model_load_error = f"sentence-transformers unavailable: {exc}"
            raise RuntimeError(self._model_load_error) from exc
        self._model = SentenceTransformer(self.settings.ingredient_embedding_model_name)
        return self._model

    def _cache_key(self, source_path: Path) -> str:
        parts = [
            self.settings.ingredient_embedding_model_name,
            str(source_path),
            str(source_path.stat().st_mtime_ns),
            str(source_path.stat().st_size),
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def _cache_file_path(self, source_path: Path) -> Path:
        configured = self.settings.ingredient_embedding_cache_path
        if configured.suffix.lower() == ".npz":
            return configured
        return configured / f"functional_ingredient_embeddings_{self._cache_key(source_path)}.npz"

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _ensure_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        if not self.enabled:
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            return
        records = self._load_records()
        source_path = self._resolve_metadata_path()
        if not records or not source_path:
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            return

        cache_path = self._cache_file_path(source_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                vectors = np.asarray(cached["vectors"], dtype=np.float32)
                if vectors.shape[0] == len(records):
                    self._embeddings = vectors
                    return
            except Exception as exc:
                logger.warning("Failed to load ingredient embedding cache: %s", exc)

        texts = [str(item.get("embedding_text", "") or item.get("search_text", "") or item.get("standard_name", "")) for item in records]
        vectors = self._compute_embeddings(texts)
        np.savez_compressed(cache_path, vectors=vectors)
        self._embeddings = vectors

    def search(self, query_text: str, top_k: int = 8) -> List[dict]:
        text = str(query_text or "").strip()
        if not text or not self.enabled:
            return []
        self._ensure_embeddings()
        records = self._load_records()
        if self._embeddings is None or self._embeddings.size == 0 or not records:
            return []

        query_vector = self._compute_embeddings([text])
        if query_vector.size == 0:
            return []
        scores = np.dot(self._embeddings, query_vector[0])
        top_indices = np.argsort(scores)[::-1][: max(1, int(top_k))]

        results: List[dict] = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            row = dict(records[int(index)])
            row["embedding_score"] = round(score, 6)
            row["retrieval_score"] = round(score, 6)
            row["retrieval_source"] = "embedding"
            results.append(row)
        return results
