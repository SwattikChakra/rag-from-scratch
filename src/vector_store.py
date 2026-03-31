"""
vector_store.py
───────────────
FAISS-backed vector store with persistence.

Design:
  • IndexFlatIP — exact nearest-neighbour using inner product.
    Works as cosine similarity when vectors are L2-normalised (default).
  • Optional IVF index for large corpora (>100k chunks).
  • Stores document objects alongside FAISS index so search returns
    full Document metadata, not just IDs.
  • Save/load via numpy + pickle for portability.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

from .ingestion import Document


class VectorStore:
    """
    Stores document embeddings in a FAISS index and retrieves the top-k
    most similar documents for a query vector.

    Parameters
    ----------
    dim         : Embedding dimension. Must match the EmbeddingModel used.
    use_ivf     : Use IVF index (faster for >100k docs). Requires training.
    n_lists     : Number of IVF cells (only used if use_ivf=True).
    """

    def __init__(
        self,
        dim: int,
        use_ivf: bool = False,
        n_lists: int = 100,
    ) -> None:
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required.\n"
                "Install with: pip install faiss-cpu"
            )
        self.dim = dim
        self.use_ivf = use_ivf
        self.n_lists = n_lists
        self._documents: List[Document] = []
        self._index: Optional[faiss.Index] = None
        self._build_index()

    # ── public API ─────────────────────────────

    def add(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """
        Add documents and their pre-computed embeddings to the store.

        Parameters
        ----------
        documents  : List of Document objects (must match embedding order).
        embeddings : Float32 array of shape (N, dim).
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents ({len(documents)}) and embeddings ({len(embeddings)}) must have equal length."
            )
        embeddings = self._ensure_float32(embeddings)

        if self.use_ivf and not self._index.is_trained:
            print("[VectorStore] Training IVF index ...")
            self._index.train(embeddings)

        self._index.add(embeddings)
        self._documents.extend(documents)
        print(f"[VectorStore] Added {len(documents)} chunks. Total: {len(self._documents)}")

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Return top-k most similar documents with their similarity scores.

        Parameters
        ----------
        query_vector : Shape (1, dim) or (dim,), float32.
        top_k        : Number of results to return.

        Returns
        -------
        List of (Document, score) tuples, sorted by descending score.
        """
        if len(self._documents) == 0:
            return []

        query_vector = self._ensure_float32(query_vector).reshape(1, -1)
        k = min(top_k, len(self._documents))

        scores, indices = self._index.search(query_vector, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue  # FAISS padding for unfilled slots
            results.append((self._documents[idx], float(score)))

        return results

    def search_mmr(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[Tuple[Document, float]]:
        """
        Maximal Marginal Relevance retrieval — balances relevance vs diversity.

        Parameters
        ----------
        top_k      : Final number of results.
        fetch_k    : Candidate pool size before MMR re-ranking.
        lambda_mult: 0 = max diversity, 1 = max relevance.
        """
        candidates = self.search(query_vector, top_k=fetch_k)
        if not candidates:
            return []

        query_vector = self._ensure_float32(query_vector).reshape(-1)
        selected: List[Tuple[Document, float]] = []
        candidate_docs = list(candidates)

        # Retrieve raw embedding vectors for MMR scoring
        all_embeddings = self._get_embeddings_for_indices(
            [self._documents.index(doc) for doc, _ in candidate_docs]
        )

        selected_indices: List[int] = []

        while len(selected) < top_k and candidate_docs:
            best_idx = -1
            best_score = -np.inf

            for i, (doc, rel_score) in enumerate(candidate_docs):
                if i in selected_indices:
                    continue
                emb = all_embeddings[i]
                relevance = np.dot(query_vector, emb)

                if not selected_indices:
                    redundancy = 0.0
                else:
                    sims = [
                        np.dot(emb, all_embeddings[j]) for j in selected_indices
                    ]
                    redundancy = max(sims)

                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx == -1:
                break

            selected.append(candidate_docs[best_idx])
            selected_indices.append(best_idx)

        return selected

    @property
    def size(self) -> int:
        return len(self._documents)

    # ── persistence ────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Persist index and document metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))

        with open(directory / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

        print(f"[VectorStore] Saved {self.size} documents to {directory}/")

    @classmethod
    def load(cls, directory: str | Path, dim: int) -> "VectorStore":
        """Load a previously saved VectorStore."""
        directory = Path(directory)

        store = cls.__new__(cls)
        store.dim = dim
        store.use_ivf = False

        store._index = faiss.read_index(str(directory / "index.faiss"))

        with open(directory / "documents.pkl", "rb") as f:
            store._documents = pickle.load(f)

        print(f"[VectorStore] Loaded {store.size} documents from {directory}/")
        return store

    # ── private ────────────────────────────────

    def _build_index(self) -> None:
        if self.use_ivf:
            quantizer = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIVFFlat(quantizer, self.dim, self.n_lists, faiss.METRIC_INNER_PRODUCT)
        else:
            self._index = faiss.IndexFlatIP(self.dim)

    @staticmethod
    def _ensure_float32(arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    def _get_embeddings_for_indices(self, indices: List[int]) -> np.ndarray:
        """Reconstruct stored vectors from the FAISS index."""
        vectors = np.zeros((len(indices), self.dim), dtype=np.float32)
        for i, idx in enumerate(indices):
            self._index.reconstruct(idx, vectors[i])
        return vectors
