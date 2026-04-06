"""
ingestion.py
────────────
PDF loading and text chunking — zero LangChain dependency.

Supports:
  • Multi-page PDF parsing via PyMuPDF (fitz)
  • Recursive character text splitter with configurable overlap
  • Page-level metadata preservation
  • Plain-text fallback for .txt files
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class Document:
    """
    Represents a single text chunk with source metadata.

    Attributes
    ----------
    text       : The chunk content.
    source     : File name / URI the chunk was extracted from.
    page       : Page number within the source (1-indexed; -1 if not applicable).
    chunk_id   : Sequential chunk index across all ingested documents.
    metadata   : Arbitrary key-value store for downstream filtering.
    """
    text: str
    source: str
    page: int = -1
    chunk_id: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(source={self.source!r}, page={self.page}, chunk={self.chunk_id}, text={preview!r}...)"


# ─────────────────────────────────────────────
# Text splitter
# ─────────────────────────────────────────────

class RecursiveCharacterSplitter:
    """
    Splits text into overlapping chunks using a priority list of separators.
    Mirrors the spirit of LangChain's RecursiveCharacterTextSplitter but is
    self-contained and easier to audit.

    Parameters
    ----------
    chunk_size    : Target maximum characters per chunk.
    chunk_overlap : Number of trailing characters carried into the next chunk.
    separators    : Ordered list of split tokens tried from coarsest to finest.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str) -> List[str]:
        """Return a list of text chunks."""
        text = self._normalise(text)
        return self._recursive_split(text, self.separators)

    # ── private ────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)          # CRLF → LF
        text = re.sub(r"\n{3,}", "\n\n", text)       # collapse excess blank lines
        text = re.sub(r"[ \t]+", " ", text)           # collapse horizontal whitespace
        return text.strip()

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0]
        fallback_seps = separators[1:]

        if sep == "":
            # Hard split — last resort
            return self._hard_split(text)

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    # carry overlap forward
                    overlap_text = current[-self.chunk_overlap :]
                    current = (overlap_text + sep + part).lstrip(sep) if overlap_text else part
                else:
                    # Single part exceeds chunk_size — recurse with finer separator
                    if fallback_seps:
                        chunks.extend(self._recursive_split(part, fallback_seps))
                    else:
                        chunks.extend(self._hard_split(part))
                    current = ""

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    def _hard_split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────

class DocumentLoader:
    """
    Loads one or more PDF / TXT files and converts them into a flat list of
    Document chunks.

    Parameters
    ----------
    chunk_size    : Passed through to RecursiveCharacterSplitter.
    chunk_overlap : Passed through to RecursiveCharacterSplitter.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._chunk_counter = 0

    def load(self, paths: List[str | Path]) -> List[Document]:
        """Load multiple files and return all chunks."""
        all_docs: List[Document] = []
        for p in paths:
            all_docs.extend(self._load_single(Path(p)))
        return all_docs

    def load_directory(self, directory: str | Path, glob: str = "**/*.pdf") -> List[Document]:
        """Recursively load all matching files from a directory."""
        directory = Path(directory)
        paths = list(directory.glob(glob)) + list(directory.glob("**/*.txt"))
        if not paths:
            raise FileNotFoundError(f"No supported files found in {directory}")
        return self.load(paths)

    # ── private ────────────────────────────────

    def _load_single(self, path: Path) -> List[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            raw_pages = self._extract_pdf(path)
        elif suffix == ".txt":
            raw_pages = [(path.read_text(encoding="utf-8", errors="replace"), -1)]
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        docs: List[Document] = []
        for text, page_num in raw_pages:
            for chunk_text in self.splitter.split(text):
                docs.append(
                    Document(
                        text=chunk_text,
                        source=path.name,
                        page=page_num,
                        chunk_id=self._chunk_counter,
                    )
                )
                self._chunk_counter += 1

        return docs

    @staticmethod
    def _extract_pdf(path: Path) -> List[tuple[str, int]]:
        """Return list of (page_text, page_number) tuples."""
        if fitz is None:
            raise ImportError(
                "PyMuPDF is required for PDF loading. Install it with:\n"
                "  pip install pymupdf"
            )
        pages = []
        with fitz.open(str(path)) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append((text, i))
        return pages
