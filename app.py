"""
app.py
──────
Gradio-based demo UI for the RAG pipeline.

Run:
    python app.py

Then open http://localhost:7860 in your browser.

Set at least one of:
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional

import gradio as gr

from src.pipeline import RAGPipeline

# ─────────────────────────────────────────────
# Global pipeline state (lazy-initialised)
# ─────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None
_uploaded_files: List[str] = []


def _detect_provider() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise EnvironmentError(
        "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."
    )


# ─────────────────────────────────────────────
# Handler functions
# ─────────────────────────────────────────────

def upload_and_index(files) -> str:
    global _pipeline, _uploaded_files

    if not files:
        return "⚠️ No files uploaded."

    file_paths = [f.name for f in files]
    _uploaded_files = file_paths
    names = [Path(p).name for p in file_paths]

    try:
        provider = _detect_provider()
        _pipeline = RAGPipeline.from_documents(
            paths=file_paths,
            provider=provider,
            top_k=5,
            use_mmr=True,
        )
        return (
            f"✅ Indexed {_pipeline.num_chunks} chunks from: {', '.join(names)}\n"
            f"Provider: {provider} | Model: {_pipeline.generator.model}"
        )
    except Exception as e:
        return f"❌ Error during indexing: {e}"


def answer_query(question: str, history: list) -> Iterator[str]:
    global _pipeline

    if not question.strip():
        yield "Please enter a question."
        return

    if _pipeline is None:
        yield "⚠️ No documents indexed yet. Please upload PDFs first."
        return

    history = history or []
    partial = ""

    try:
        for token in _pipeline.stream_query(question, top_k=5):
            partial += token
            yield partial
    except Exception as e:
        yield f"❌ Generation error: {e}"


def show_retrieved_chunks(question: str) -> str:
    if not question.strip():
        return "Enter a question above first."
    if _pipeline is None:
        return "No documents indexed."

    chunks = _pipeline.retrieve_only(question, top_k=5)
    lines = []
    for i, (doc, score) in enumerate(chunks, 1):
        lines.append(
            f"**[{i}] {doc.source} — Page {doc.page}** (score: {score:.4f})\n"
            f"{doc.text[:300]}{'...' if len(doc.text) > 300 else ''}"
        )
    return "\n\n---\n\n".join(lines)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

css = """
.gradio-container { max-width: 900px !important; }
.status-box { font-family: monospace; font-size: 0.85em; }
"""

with gr.Blocks(
    title="RAG From Scratch",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=css,
) as demo:

    gr.Markdown(
        """
        # 📚 RAG From Scratch
        **No LangChain. No magic.** Just embeddings + FAISS + an LLM.

        1. Upload one or more PDFs
        2. Hit **Index Documents**
        3. Ask questions
        """
    )

    with gr.Tab("💬 Ask Questions"):
        with gr.Row():
            with gr.Column(scale=2):
                upload_btn = gr.File(
                    label="Upload PDFs",
                    file_types=[".pdf", ".txt"],
                    file_count="multiple",
                )
                index_btn = gr.Button("⚡ Index Documents", variant="primary")
                index_status = gr.Textbox(
                    label="Indexing Status",
                    interactive=False,
                    elem_classes="status-box",
                    lines=3,
                )

            with gr.Column(scale=3):
                question_box = gr.Textbox(
                    label="Your Question",
                    placeholder="What does the document say about...?",
                    lines=2,
                )
                ask_btn = gr.Button("🔍 Ask", variant="primary")
                answer_box = gr.Textbox(
                    label="Answer",
                    interactive=False,
                    lines=10,
                )

        index_btn.click(fn=upload_and_index, inputs=[upload_btn], outputs=[index_status])
        ask_btn.click(fn=answer_query, inputs=[question_box, gr.State([])], outputs=[answer_box])
        question_box.submit(fn=answer_query, inputs=[question_box, gr.State([])], outputs=[answer_box])

    with gr.Tab("🔬 Inspect Retrieval"):
        gr.Markdown("Debug view — see exactly which chunks were retrieved for your query.")
        debug_question = gr.Textbox(label="Question", placeholder="Same question as above...")
        debug_btn = gr.Button("Show Retrieved Chunks")
        chunks_display = gr.Markdown()
        debug_btn.click(fn=show_retrieved_chunks, inputs=[debug_question], outputs=[chunks_display])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
