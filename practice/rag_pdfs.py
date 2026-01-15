import os
import re
from pathlib import Path

import ollama
from pypdf import PdfReader

# ======================
# CONFIG
# ======================
PDF_FOLDER = r"C:\th\phd\mod-data\Seminar (RAG)\_full texts"

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL  = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

# Chunking settings (good starting point for academic PDFs)
CHUNK_SIZE = 1200          # characters
CHUNK_OVERLAP = 200        # characters
MAX_CHUNKS_PER_DOC = 200   # safety cap per PDF (adjust if needed)

# NEW: hard cap to avoid "input length exceeds the context length" during embedding
MAX_CHARS_PER_EMBED = 3000

# Each element: (chunk_text, embedding_vector, source_id)
VECTOR_DB = []

# ======================
# PDF -> text
# ======================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using pypdf. Works for normal (non-scanned) PDFs."""
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)

def normalize_text(text: str) -> str:
    """Light cleanup: remove excessive whitespace. Keep it conservative."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ======================
# Chunking
# ======================
def chunk_text(text: str, chunk_size: int, overlap: int):
    """Split text into overlapping character chunks."""
    if chunk_size <= overlap:
        raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks

# ======================
# Embeddings + retrieval
# ======================
def add_chunk_to_database(chunk_text: str, source_id: str):
    """
    Embed and store a chunk safely.
    Fixes: "the input length exceeds the context length (status code: 400)"
    by truncating oversized chunks before calling ollama.embed().
    """
    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        return

    # Hard cap to prevent embed context overflow
    if len(chunk_text) > MAX_CHARS_PER_EMBED:
        chunk_text = chunk_text[:MAX_CHARS_PER_EMBED]

    try:
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk_text)["embeddings"][0]
        VECTOR_DB.append((chunk_text, embedding, source_id))
    except Exception as e:
        # If something still slips through, don't crash the whole run—skip this chunk.
        msg = str(e).lower()
        if "exceeds the context length" in msg:
            # Last-resort: truncate harder and retry once
            shorter = chunk_text[: max(500, MAX_CHARS_PER_EMBED // 2)]
            try:
                embedding = ollama.embed(model=EMBEDDING_MODEL, input=shorter)["embeddings"][0]
                VECTOR_DB.append((shorter, embedding, source_id))
                return
            except Exception:
                pass
        # Log and skip
        print(f"    - Skipped a chunk from [{source_id}] due to embed error: {e}")

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def retrieve(query: str, top_n: int = 5):
    query = (query or "").strip()
    if not query:
        return []

    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    sims = []
    for chunk_text, embedding, source_id in VECTOR_DB:
        sim = cosine_similarity(query_embedding, embedding)
        sims.append((chunk_text, sim, source_id))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]

# ======================
# Build corpus from PDFs
# ======================
def build_vector_db_from_pdfs(pdf_folder: str):
    pdf_paths = sorted([str(p) for p in Path(pdf_folder).rglob("*.pdf")])

    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in: {pdf_folder}")

    print(f"Found {len(pdf_paths)} PDF files.")
    total_chunks_added = 0

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        filename = os.path.basename(pdf_path)
        source_id = filename  # you can also use full path if you prefer

        print(f"\n[{idx}/{len(pdf_paths)}] Reading: {filename}")
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            text = normalize_text(raw_text)

            if len(text) < 200:
                print("  - Skipping (too little extractable text). Possibly scanned PDF.")
                continue

            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            chunks = chunks[:MAX_CHUNKS_PER_DOC]

            before = len(VECTOR_DB)
            for ch in chunks:
                add_chunk_to_database(ch, source_id)
            added_now = len(VECTOR_DB) - before

            total_chunks_added += added_now
            print(f"  - Added {added_now}/{len(chunks)} chunks from this PDF. Total chunks so far: {total_chunks_added}")

        except Exception as e:
            print(f"  - ERROR processing {filename}: {e}")

    print(f"\nDONE. VECTOR_DB size = {len(VECTOR_DB)} chunks")

# ======================
# Main
# ======================
if __name__ == "__main__":
    build_vector_db_from_pdfs(PDF_FOLDER)

    while True:
        input_query = input("\nAsk me a question (or type 'exit'): ").strip()
        if not input_query or input_query.lower() == "exit":
            break

        retrieved = retrieve(input_query, top_n=5)

        print("\nRetrieved knowledge:")
        for chunk_text, sim, source_id in retrieved:
            preview = chunk_text[:180].replace("\n", " ")
            print(f" - ({sim:.2f}) [{source_id}] {preview}...")

        # Build the instruction prompt with explicit sources
        context_lines = []
        for chunk_text, sim, source_id in retrieved:
            context_lines.append(f"[SOURCE: {source_id} | sim={sim:.2f}]\n{chunk_text}")

        instruction_prompt = (
            "You are a helpful chatbot.\n"
            "Use ONLY the context below to answer the question.\n"
            "If the answer is not contained in the context, say you don't know.\n\n"
            "CONTEXT:\n"
            + "\n\n---\n\n".join(context_lines)
        )

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": input_query},
            ],
            stream=True,
        )

        print("\nChatbot response:")
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print()
