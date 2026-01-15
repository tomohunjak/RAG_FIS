import os
import re
import csv
import uuid
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
import ollama

from datetime import datetime
import os

LOG_PATH = os.path.join("logs", "rag_results.txt")

def append_log(question_key: str, question_text: str, answer_text: str, sources=None) -> None:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n=== {ts} | {question_key} ===\n")
        f.write(f"Q: {question_text}\n\n")
        f.write(f"A: {answer_text.strip()}\n")

        # Optional: log sources if your pipeline returns them
        if sources:
            f.write("\nSOURCES:\n")
            if isinstance(sources, (list, tuple)):
                for s in sources:
                    f.write(f"- {s}\n")
            else:
                f.write(str(sources).strip() + "\n")


def load_questions(path="questions.txt"):
    questions = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, text = line.split("=", 1)
            questions[key.strip()] = text.strip()
    return questions


@dataclass
class RAGConfig:
    persist_dir: str = "rag_store"
    collection_name: str = "slr_rag"
    embed_model: str = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest"
    gen_model: str = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
    chunk_chars: int = 750
    chunk_overlap: int = 150
    top_k: int = 5
    quote_max_chars: int = 400
    embed_max_chars: int = 1200
    embed_min_chars: int = 200



def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_doc_id(path_or_key: str) -> str:
    base = os.path.basename(path_or_key)
    base = re.sub(r"\W+", "_", base).strip("_").lower()
    return base if base else str(uuid.uuid4())


def sanitize_metadata(meta: Dict) -> Dict:
    return {k: v for k, v in meta.items() if v is not None}


def embed_text(cfg: RAGConfig, text: str) -> List[float]:
    t = text.strip()
    max_chars = cfg.embed_max_chars

    while True:
        cut = t if len(t) <= max_chars else t[:max_chars]
        try:
            resp = ollama.embeddings(model=cfg.embed_model, prompt=cut)
            return resp["embedding"]
        except Exception as e:
            msg = str(e).lower()
            if "exceeds the context length" in msg and max_chars > cfg.embed_min_chars:
                max_chars = max(cfg.embed_min_chars, max_chars // 2)
                continue
            raise



def extract_pdf_text_by_page(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(clean_text(t))
    return pages


def chunk_fulltext(pages: List[str], cfg: RAGConfig) -> List[Dict]:
    joined = []
    for i, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        joined.append(f"\n\n[PAGE {i}]\n\n{page_text}")
    full = clean_text("".join(joined))

    chunks = []
    n = len(full)
    start = 0
    while start < n:
        end = min(start + cfg.chunk_chars, n)
        chunk = full[start:end]

        if end < n:
            boundary = max(chunk.rfind(". "), chunk.rfind("\n\n"), chunk.rfind("; "))
            if boundary > int(cfg.chunk_chars * 0.6):
                end = start + boundary + 1
                chunk = full[start:end]

        chunk = clean_text(chunk)
        if chunk:
            pages_in_chunk = re.findall(r"\[PAGE (\d+)\]", chunk)
            if pages_in_chunk:
                pmin = int(pages_in_chunk[0])
                pmax = int(pages_in_chunk[-1])
            else:
                pmin = None
                pmax = None

            chunks.append({"text": chunk, "page_start": pmin, "page_end": pmax})

        if end == n:
            break
        start = max(0, end - cfg.chunk_overlap)

    return chunks


def load_abstracts_csv(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            dialect = csv.excel
            dialect.delimiter = ";"

        reader = csv.DictReader(f, dialect=dialect)

        for r in reader:
            abstract = (r.get("Abstract") or r.get("abstract") or "").strip()
            if not abstract:
                continue

            rows.append({
                "id": (r.get("ID") or r.get("id") or "").strip(),
                "title": (r.get("Title") or r.get("title") or "").strip(),
                "year": (r.get("Year") or r.get("year") or "").strip(),
                "authors": (r.get("Authors") or r.get("authors") or "").strip(),
                "doi": (r.get("DOI") or r.get("doi") or "").strip(),
                "abstract": clean_text(abstract)
            })
    return rows



def get_chroma_client(cfg: RAGConfig):
    os.makedirs(cfg.persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=cfg.persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )


def get_collection(cfg: RAGConfig):
    client = get_chroma_client(cfg)
    return client.get_or_create_collection(
        name=cfg.collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_pdfs(cfg: RAGConfig, pdf_dir: str):
    col = get_collection(cfg)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in: {pdf_dir}")
        return

    for fname in sorted(pdf_files):
        path = os.path.join(pdf_dir, fname)
        doc_id = stable_doc_id(fname)
        print(f"Ingesting PDF: {fname} -> doc_id={doc_id}")

        pages = extract_pdf_text_by_page(path)
        chunks = chunk_fulltext(pages, cfg)

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict] = []
        embs: List[List[float]] = []

        for i, ch in enumerate(chunks):
            chunk_id = f"{doc_id}__full__{i}"
            text = ch["text"]
            emb = embed_text(cfg, text)

            ids.append(chunk_id)
            docs.append(text)
            embs.append(emb)

            meta = {
                "doc_id": doc_id,
                "source_type": "fulltext",
                "filename": fname,
                "title": fname,
                "page_start": ch["page_start"],
                "page_end": ch["page_end"],
                "chunk_index": i
            }
            metas.append(sanitize_metadata(meta))

        if ids:
            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            print(f"  Added {len(ids)} chunks.")
        else:
            print("  No text chunks extracted (PDF may be scanned or empty).")


def ingest_abstracts(cfg: RAGConfig, abstracts_csv: str):
    col = get_collection(cfg)
    rows = load_abstracts_csv(abstracts_csv)
    if not rows:
        print(f"No abstracts found in: {abstracts_csv}")
        return

    print(f"Ingesting abstracts from CSV: {abstracts_csv} ({len(rows)} rows)")

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []
    embs: List[List[float]] = []

    for idx, r in enumerate(rows):
        base = r["id"] if r["id"] else f"abstract_{idx}"
        doc_id = stable_doc_id(base)

        text = r["abstract"]
        emb = embed_text(cfg, text)

        chunk_id = f"{doc_id}__abstract__0"

        ids.append(chunk_id)
        docs.append(text)
        embs.append(emb)

        meta = {
            "doc_id": doc_id,
            "source_type": "abstract",
            "filename": os.path.basename(abstracts_csv),
            "title": r["title"] or doc_id,
            "year": r["year"],
            "authors": r["authors"],
            "doi": r["doi"],
            "chunk_index": 0
        }
        metas.append(sanitize_metadata(meta))

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"  Added {len(ids)} abstracts (each as a single chunk).")


def retrieve(cfg: RAGConfig, query: str, top_k: Optional[int] = None) -> List[Tuple[str, Dict, float, str]]:
    col = get_collection(cfg)
    k = top_k or cfg.top_k
    q_emb = embed_text(cfg, query)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    out: List[Tuple[str, Dict, float, str]] = []
    for doc, meta, dist, cid in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
        res["ids"][0]
    ):
        out.append((cid, meta, float(dist), doc))
    return out


def build_traceable_context(cfg: RAGConfig, retrieved: List[Tuple[str, Dict, float, str]], query: str) -> str:
    blocks = []
    for cid, meta, dist, doc in retrieved:
        excerpt = doc[:cfg.quote_max_chars]
        title = meta.get("title") or meta.get("filename") or meta.get("doc_id")
        src_type = meta.get("source_type", "unknown")
        pages = ""
        if meta.get("page_start") is not None and meta.get("page_end") is not None:
            pages = f", pages {meta['page_start']}-{meta['page_end']}"
        blocks.append(
            f"[SOURCE {cid}] ({src_type}; {title}{pages})\n"
            f"EXCERPT:\n{excerpt}\n"
        )

    context = "\n---\n".join(blocks)

    prompt = f"""You are a helpful research assistant.
Use ONLY the information in the SOURCES below. If the sources do not contain the answer, say so.

IMPORTANT:
- Do NOT include any citations, author names, or years (e.g., do not write "Ivanov et al., 2016").
- Do NOT mention papers by name. Summarize ideas only.

Question: {query}

SOURCES:
{context}

Answer in a concise academic style (no citations).
"""

    return prompt


def answer_query(cfg: RAGConfig, query: str, top_k: Optional[int] = None) -> Dict:
    retrieved = retrieve(cfg, query, top_k=top_k)
    prompt = build_traceable_context(cfg, retrieved, query)

    resp = ollama.generate(model=cfg.gen_model, prompt=prompt)
    text = resp["response"]

    evidence = []
    for cid, meta, dist, doc in retrieved:
        evidence.append({
            "source_id": cid,
            "distance": dist,
            "title": meta.get("title"),
            "filename": meta.get("filename"),
            "source_type": meta.get("source_type"),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "excerpt": doc[:cfg.quote_max_chars]
        })

    return {"answer": text, "evidence": evidence}

def ingest(cfg: RAGConfig, pdf_dir: str, abstracts_csv: str = ""):
    # Ingest PDFs if folder exists
    if pdf_dir and os.path.isdir(pdf_dir):
        ingest_pdfs(cfg, pdf_dir)
    else:
        print(f"PDF folder not found: {pdf_dir}")

    # Ingest abstracts if CSV path is provided
    if abstracts_csv:
        if os.path.isfile(abstracts_csv):
            ingest_abstracts(cfg, abstracts_csv)
        else:
            print(f"Abstracts CSV not found: {abstracts_csv}")

def main():
    parser = argparse.ArgumentParser(description="Local SLR-grounded RAG pipeline with explicit traceability.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDFs and/or abstracts into the vector store.")
    p_ingest.add_argument("--pdf_dir", default="data/pdfs", help="Folder with PDFs.")
    p_ingest.add_argument("--abstracts_csv", default="", help="Optional CSV with abstracts (one per row).")

    p_ask = sub.add_parser("ask", help="Ask a question and get an answer with sources.")
    p_ask.add_argument("query", help="Your question")
    p_ask.add_argument("--top_k", type=int, default=5, help="Number of retrieved chunks")

    args = parser.parse_args()
    cfg = RAGConfig()

    if args.cmd == "ingest":
        ingest(cfg, pdf_dir=args.pdf_dir, abstracts_csv=args.abstracts_csv)

    elif args.cmd == "ask":
        questions = load_questions()

        if args.query in questions:
            question_key = args.query
            question_text = questions[args.query]
            print(f"(Using preset question {question_key})\n")
        else:
            question_key = "custom"
            question_text = args.query

        result = answer_query(cfg, question_text, top_k=args.top_k)

        print("\n=== ANSWER ===\n")
        print(result["answer"].strip())

        sources = result.get("evidence") if isinstance(result, dict) else None
        append_log(question_key, question_text, result["answer"], sources)
        print(f"\n(Logged to {LOG_PATH})")

if __name__ == "__main__":
    main()

