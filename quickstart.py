"""
quickstart.py
─────────────
Minimal working example — run this first to verify your setup.

    python quickstart.py --pdf path/to/your.pdf --question "What is this doc about?"
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG From Scratch — Quick Start")
    parser.add_argument("--pdf", required=True, help="Path to a PDF file")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mmr", action="store_true", help="Use MMR retrieval")
    parser.add_argument("--debug", action="store_true", help="Print retrieved chunks")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  RAG From Scratch")
    print(f"  PDF:      {args.pdf}")
    print(f"  Question: {args.question}")
    print(f"  Provider: {args.provider}")
    print(f"{'='*60}\n")

    # Build pipeline
    rag = RAGPipeline.from_documents(
        paths=[args.pdf],
        provider=args.provider,
        top_k=args.top_k,
        use_mmr=args.mmr,
        index_dir=".cache/index",   # save index so subsequent runs are instant
    )

    print(f"\n📄 Indexed {rag.num_chunks} chunks\n")

    # Optional: show what was retrieved
    if args.debug:
        print("── Retrieved Chunks ──────────────────────────────")
        for i, (doc, score) in enumerate(rag.retrieve_only(args.question), 1):
            print(f"\n[{i}] {doc.source} p.{doc.page} (score={score:.4f})")
            print(doc.text[:200] + "...")
        print()

    # Generate answer
    print("── Answer ────────────────────────────────────────")
    result = rag.query(args.question)
    print(result.answer)
    print(f"\n── Sources: {result.sources}")
    print(f"── Tokens:  {result.prompt_tokens} in / {result.completion_tokens} out")


if __name__ == "__main__":
    main()
