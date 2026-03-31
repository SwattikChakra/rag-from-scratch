"""
generator.py
────────────
LLM-backed answer generation with multi-provider support.

Supported providers (configured via .env):
  • Anthropic Claude (default)
  • OpenAI / OpenAI-compatible (Together AI, Groq, Ollama, etc.)

Features:
  • Structured prompt building with retrieved context injection
  • Source citation in model output
  • Streaming support
  • Token usage logging
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional, Tuple

from .ingestion import Document


# ─────────────────────────────────────────────
# Response model
# ─────────────────────────────────────────────

@dataclass
class GenerationResult:
    answer: str
    sources: List[Tuple[str, int]]   # list of (source_filename, page_num)
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __str__(self) -> str:
        src_str = ", ".join(f"{s}:p{p}" for s, p in self.sources) or "none"
        return (
            f"Answer:\n{self.answer}\n\n"
            f"Sources: {src_str}\n"
            f"Tokens: {self.total_tokens} ({self.model})"
        )


# ─────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the context passages provided below.
If the context doesn't contain enough information to answer, say:
"I don't have enough information in the provided context to answer this."

Rules:
- Be factual and concise.
- Cite the source filename and page number in square brackets, e.g. [report.pdf, p.3].
- Do not hallucinate or add information not present in the context.
"""

def build_rag_prompt(query: str, context_docs: List[Tuple[Document, float]]) -> str:
    """Format retrieved documents into a numbered context block."""
    context_blocks = []
    for i, (doc, score) in enumerate(context_docs, start=1):
        block = (
            f"[{i}] Source: {doc.source}, Page: {doc.page} (relevance: {score:.3f})\n"
            f"{doc.text}"
        )
        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""CONTEXT PASSAGES:
{context_str}

---

USER QUESTION: {query}

ANSWER:"""


# ─────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────

class RAGGenerator:
    """
    Wraps an LLM API to generate grounded answers from retrieved context.

    Parameters
    ----------
    provider    : 'anthropic' | 'openai' | 'openai_compatible'
    model       : Model identifier string.
    temperature : Sampling temperature (0 = deterministic).
    max_tokens  : Max output tokens.
    base_url    : Override base URL for OpenAI-compatible providers
                  (e.g. 'http://localhost:11434/v1' for Ollama).
    """

    PROVIDER_DEFAULTS = {
        "anthropic": "claude-3-haiku-20240307",
        "openai": "gpt-4o-mini",
        "openai_compatible": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model or self.PROVIDER_DEFAULTS.get(self.provider, "claude-3-haiku-20240307")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._client = self._init_client()

    # ── public API ─────────────────────────────

    def generate(
        self,
        query: str,
        context_docs: List[Tuple[Document, float]],
    ) -> GenerationResult:
        """
        Generate a grounded answer for the query using retrieved documents.

        Returns
        -------
        GenerationResult with answer, sources, and token counts.
        """
        user_prompt = build_rag_prompt(query, context_docs)
        answer, usage = self._call_llm(user_prompt)

        sources = list({(doc.source, doc.page) for doc, _ in context_docs})

        return GenerationResult(
            answer=answer,
            sources=sources,
            model=self.model,
            prompt_tokens=usage.get("input", 0),
            completion_tokens=usage.get("output", 0),
        )

    def stream(
        self,
        query: str,
        context_docs: List[Tuple[Document, float]],
    ) -> Iterator[str]:
        """
        Streaming generation — yields text tokens as they arrive.
        Useful for Gradio / Streamlit live rendering.
        """
        user_prompt = build_rag_prompt(query, context_docs)
        yield from self._stream_llm(user_prompt)

    # ── private ────────────────────────────────

    def _init_client(self):
        if self.provider == "anthropic":
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")
                return anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("pip install anthropic")

        elif self.provider in ("openai", "openai_compatible"):
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY", "ollama")  # ollama ignores key
                kwargs = {"api_key": api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                return OpenAI(**kwargs)
            except ImportError:
                raise ImportError("pip install openai")

        else:
            raise ValueError(f"Unknown provider: {self.provider!r}. Choose 'anthropic' or 'openai'.")

    def _call_llm(self, user_prompt: str) -> Tuple[str, dict]:
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=RAG_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text
            usage = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
            return text, usage

        else:  # openai / openai_compatible
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = response.choices[0].message.content
            usage = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            }
            return text, usage

    def _stream_llm(self, user_prompt: str) -> Iterator[str]:
        if self.provider == "anthropic":
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=RAG_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text

        else:
            stream = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
