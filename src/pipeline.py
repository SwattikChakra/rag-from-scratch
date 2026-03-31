"""
pipeline.py
───────────
End-to-end RAG pipeline — the single entry point for most use-cases.

Orchestrates:
  DocumentLoader → EmbeddingModel → VectorStore → RAGGenerator

Usage
-----
    from src.pipeline import RAGPipeline

    rag = RAGPipeline.from_documents(
        paths=["docs/annual_report.pdf"],
        provider="anthropic",       # or "openai"
    )

    result = rag.query("What was the revenue in Q3?")
    print(result)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .embeddings import EmbeddingModel
from .generator import GenerationResult, RAGGenerator
from .ingestion import Document, DocumentLoader
from .vector_store import VectorStore


class RAGPipeline:
    """
    Full retrieval-augmented generation pipeline.

    Parameters
    ----------
    embedding_model : Pre-built EmbeddingModel instance.
    vector_store    : Pre-built VectorStore instance.
    generator       : Pre-built RAGGenerator instance.
    top_k           : Default number of chunks to retrieve per query.
    use_mmr         : Use MMR re-ranking for diverse retrieval.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        generator: RAGGenerator,
        top_k: int = 5,
        use_mmr: bool = False,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.generator = generator
        self.top_k = top_k
        self.use_mmr = use_mmr

    # ── factory methods ────────────────────────

    @classmethod
    def from_documents(
        cls,
        paths: List[Union[str, Path]],
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        provider: str = "anthropic",
        llm_model: Optional[str] = None,
        top_k: int = 5,
        use_mmr: bool = False,
        index_dir: Optional[str] = None,
    ) -> "RAGPipeline":
        """
        Build a pipeline end-to-end from file paths.

        If index_dir is given and already exists, the saved index is loaded
        instead of re-ingesting and re-embedding (fast cold start).
        """
        embed_model = EmbeddingModel(model_name=embed_model_name)

        # ── try to load saved index ─────────────
        if index_dir and Path(index_dir).exists():
            print(f"[RAGPipeline] Loading saved index from {index_dir} ...")
            vs = VectorStore.load(index_dir, dim=embed_model.dim)
        else:
            # ── ingest & embed from scratch ─────
            print("[RAGPipeline] Ingesting documents ...")
            t0 = time.time()
            loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = loader.load(paths)
            print(f"[RAGPipeline] {len(documents)} chunks in {time.time()-t0:.1f}s")

            print("[RAGPipeline] Embedding chunks ...")
            t0 = time.time()
            embeddings = embed_model.encode_documents(documents)
            print(f"[RAGPipeline] Embeddings done in {time.time()-t0:.1f}s")

            vs = VectorStore(dim=embed_model.dim)
            vs.add(documents, embeddings)

            if index_dir:
                vs.save(index_dir)

        gen = RAGGenerator(provider=provider, model=llm_model)

        return cls(
            embedding_model=embed_model,
            vector_store=vs,
            generator=gen,
            top_k=top_k,
            use_mmr=use_mmr,
        )

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        **kwargs,
    ) -> "RAGPipeline":
        """Load all PDFs and TXTs from a directory."""
        directory = Path(directory)
        paths = list(directory.glob("**/*.pdf")) + list(directory.glob("**/*.txt"))
        if not paths:
            raise FileNotFoundError(f"No supported files found in {directory}")
        return cls.from_documents(paths=paths, **kwargs)

    # ── query API ──────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_mmr: Optional[bool] = None,
    ) -> GenerationResult:
        """
        Full RAG query: embed → retrieve → generate.

        Parameters
        ----------
        question : Natural language question.
        top_k    : Override instance-level top_k for this call.
        use_mmr  : Override instance-level use_mmr for this call.

        Returns
        -------
        GenerationResult with answer, sources, and token usage.
        """
        k = top_k or self.top_k
        mmr = use_mmr if use_mmr is not None else self.use_mmr

        # 1. Embed the query
        query_vec = self.embedding_model.encode_query(question)

        # 2. Retrieve relevant chunks
        if mmr:
            context = self.vector_store.search_mmr(query_vec, top_k=k, fetch_k=k * 4)
        else:
            context = self.vector_store.search(query_vec, top_k=k)

        if not context:
            from .generator import GenerationResult
            return GenerationResult(
                answer="No documents have been indexed. Please add documents first.",
                sources=[],
                model=self.generator.model,
            )

        # 3. Generate answer
        return self.generator.generate(question, context)

    def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Streaming variant of query() — yields text tokens.

        Useful for Gradio / Streamlit live rendering.
        """
        k = top_k or self.top_k
        query_vec = self.embedding_model.encode_query(question)
        context = self.vector_store.search(query_vec, top_k=k)
        yield from self.generator.stream(question, context)

    def add_documents(self, paths: List[Union[str, Path]], chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        """
        Add new documents to an existing pipeline (hot update — no rebuild).
        """
        loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = loader.load(paths)
        embeddings = self.embedding_model.encode_documents(documents)
        self.vector_store.add(documents, embeddings)

    # ── utility ────────────────────────────────

    @property
    def num_chunks(self) -> int:
        return self.vector_store.size

    def retrieve_only(
        self, question: str, top_k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """Return raw retrieved chunks without generation — useful for debugging."""
        k = top_k or self.top_k
        query_vec = self.embedding_model.encode_query(question)
        return self.vector_store.search(query_vec, top_k=k)
