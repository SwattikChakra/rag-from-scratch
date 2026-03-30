"""
embeddings.py
─────────────
Embedding generation with sentence-transformers.

Design choices:
  • Default model: BAAI/bge-small-en-v1.5 — best quality/size trade-off
    for retrieval tasks (beats OpenAI ada-002 on BEIR in many benchmarks).
  • Batched inference with configurable batch size.
  • L2 normalisation applied so cosine similarity == dot product (FAISS
    IndexFlatIP is faster than IndexFlatL2 for normalised vectors).
  • Optional GPU support via device parameter.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

from .ingestion import Document


DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Alternatives ranked by quality vs speed:
#   "BAAI/bge-base-en-v1.5"      — larger, better, slower
#   "BAAI/bge-large-en-v1.5"     — best quality, needs ~1.3 GB RAM
#   "all-MiniLM-L6-v2"           — fast, decent, widely used in tutorials
#   "thenlper/gte-small"          — good for multilingual edge cases


class EmbeddingModel:
    """
    Wraps a SentenceTransformer model and exposes encode() for both
    raw strings and Document objects.

    Parameters
    ----------
    model_name  : HuggingFace model identifier.
    device      : 'cpu', 'cuda', 'mps', or None (auto-detect).
    batch_size  : How many texts to embed in one forward pass.
    normalize   : L2-normalise output vectors (recommended for cosine retrieval).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        print(f"[EmbeddingModel] Loading '{model_name}' ...")
        self._model = SentenceTransformer(model_name, device=device)
        self.dim: int = self._model.get_sentence_embedding_dimension()
        print(f"[EmbeddingModel] Ready — embedding dim: {self.dim}")

    # ── public API ─────────────────────────────

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray of shape (N, dim), dtype float32.
        """
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 64,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Embed a list of Document objects by their .text field.

        BGE models benefit from a prefix on passage-side text:
        "Represent this sentence: " — already handled inside the model weights,
        but we prepend the instruction for bge-* variants to match their
        training setup.
        """
        texts = [self._passage_text(doc.text) for doc in documents]
        return self.encode_texts(texts)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns
        -------
        np.ndarray of shape (1, dim), dtype float32.
        """
        query_text = self._query_text(query)
        return self.encode_texts([query_text])

    # ── private ────────────────────────────────

    def _passage_text(self, text: str) -> str:
        """Prepend BGE passage instruction if using a bge model."""
        if "bge" in self.model_name.lower():
            return f"Represent this passage for retrieval: {text}"
        return text

    def _query_text(self, text: str) -> str:
        """Prepend BGE query instruction if using a bge model."""
        if "bge" in self.model_name.lower():
            return f"Represent this query for retrieving relevant passages: {text}"
        return text
