"""
Unified Index Pipeline
Extract → Clean → Chunk → Embed → Insert to Milvus
Uses OpenAI text-embedding-3-small (1536 dim) with enhanced schema.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

# Import our modules
from scripts.text_cleaner import clean_and_detect, detect_doc_type, detect_chunk_type
from scripts.unified_extractor import extract_all_documents, ExtractedDocument
from scripts.semantic_chunker import HybridChunker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = os.environ.get(
    "MILVUS_COLLECTION_RAG",
    os.environ.get("MILVUS_COLLECTION", "documents"),
)
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small
BATCH_SIZE = 50  # Embedding batch size
EMBEDDING_BATCH_SIZE = 100  # OpenAI API batch limit

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")


# ---------------------------------------------------------------------------
# Milvus Collection Setup (Enhanced Schema)
# ---------------------------------------------------------------------------


def setup_milvus_collection(drop_existing: bool = True) -> Collection:
    """Create Milvus collection with enhanced metadata schema."""
    print(f"[MILVUS] Connecting to {MILVUS_HOST}:{MILVUS_PORT}...")

    if MILVUS_TOKEN:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            token=MILVUS_TOKEN,
            secure=True,
            timeout=30,
        )
    else:
        connections.connect(
            alias="default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=30
        )
    print("[MILVUS] Connected")

    if utility.has_collection(COLLECTION_NAME):
        if drop_existing:
            print(f"[MILVUS] Dropping existing collection: {COLLECTION_NAME}")
            utility.drop_collection(COLLECTION_NAME)
        else:
            print(f"[MILVUS] Using existing collection: {COLLECTION_NAME}")
            return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page", dtype=DataType.INT32),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="chunk_index", dtype=DataType.INT32),
        FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]

    schema = CollectionSchema(
        fields, description="Dhara RAG - Document embeddings with metadata"
    )
    collection = Collection(COLLECTION_NAME, schema)

    # HNSW index for fast retrieval (Phase 6)
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,  # Number of bi-directional links
            "efConstruction": 256,  # Build-time accuracy
        },
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

    print(
        f"[MILVUS] Collection created: {COLLECTION_NAME} (HNSW, {EMBEDDING_DIM}-dim COSINE)"
    )
    return collection


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_document(doc: ExtractedDocument) -> List[Dict[str, Any]]:
    """
    Chunk a document into pieces with metadata.
    Returns list of dicts with text, source, page, language, doc_type, etc.
    """
    chunker = HybridChunker(semantic_max=1200, fallback_size=800, overlap=150)
    all_chunks = []

    full_text = doc.full_text
    if not full_text or len(full_text.strip()) < 50:
        return []

    # Detect doc type from filename + first page
    doc_type = detect_doc_type(full_text[:2000], doc.filename)

    # Process each page
    page_offset = 0
    for page in doc.pages:
        if not page.text or len(page.text.strip()) < 30:
            continue

        # Clean and detect language
        cleaned_text, language = clean_and_detect(page.text)
        if len(cleaned_text) < 30:
            continue

        # Chunk the page
        chunks = chunker.chunk_text(cleaned_text, source=doc.relative_path)

        for chunk_idx, (chunk_text, metadata) in enumerate(chunks):
            if len(chunk_text.strip()) < 50:
                continue

            # Detect chunk type
            chunk_type = detect_chunk_type(chunk_text)

            # Clean again (in case chunking introduced issues)
            chunk_text = chunk_text.strip()

            all_chunks.append(
                {
                    "text": chunk_text,
                    "source": doc.relative_path,
                    "page": page.page_num,
                    "language": language,
                    "doc_type": doc_type,
                    "chunk_type": chunk_type,
                    "chunk_index": page_offset + chunk_idx,
                    "file_hash": doc.file_hash,
                }
            )

        page_offset += len(chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def get_embeddings_client():
    """Get OpenAI embeddings client."""
    from langchain_openai import OpenAIEmbeddings

    api_key = os.environ.get("OPENAI_API_KEY", "")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )


def embed_batch(texts: List[str], embeddings_client) -> List[List[float]]:
    """Embed a batch of texts."""
    return embeddings_client.embed_documents(texts)


# ---------------------------------------------------------------------------
# Insert to Milvus
# ---------------------------------------------------------------------------


def insert_chunks_to_milvus(
    collection: Collection,
    chunks: List[Dict[str, Any]],
    embeddings_client,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed and insert chunks into Milvus in batches."""
    total_inserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        try:
            vectors = embed_batch(texts, embeddings_client)

            entities = [
                [c["text"] for c in batch],
                [c["source"] for c in batch],
                [c["page"] for c in batch],
                [c["language"] for c in batch],
                [c["doc_type"] for c in batch],
                [c["chunk_type"] for c in batch],
                [c["chunk_index"] for c in batch],
                [c["file_hash"] for c in batch],
                vectors,
            ]

            collection.insert(entities)
            total_inserted += len(batch)

            if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(chunks):
                collection.flush()
                print(
                    f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks"
                )

        except Exception as e:
            print(f"  Batch error at {i}: {e}")
            continue

    collection.flush()
    return total_inserted


# ---------------------------------------------------------------------------
# Progress tracking (resume support)
# ---------------------------------------------------------------------------

PROGRESS_FILE = Path("data/indexing_progress.json")


def load_progress() -> Dict[str, str]:
    """Load indexed file hashes for resume support."""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_progress(progress: Dict[str, str]):
    """Save progress."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    docs_dir: str = "data/docs",
    drop_existing: bool = True,
    max_extract_workers: int = 4,
    skip_ocr: bool = False,
    resume: bool = False,
):
    """
    Full pipeline: Extract → Clean → Chunk → Embed → Insert

    Args:
        docs_dir: Path to documents directory
        drop_existing: Drop and recreate collection
        max_extract_workers: Parallel extraction threads
        skip_ocr: Skip OCR for scanned documents (faster but less content)
        resume: Resume from last progress
    """
    print("=" * 70)
    print("DHARA RAG - UNIFIED INDEX PIPELINE")
    print("=" * 70)
    print(f"  Documents: {docs_dir}")
    print(f"  Embedding: OpenAI text-embedding-3-small ({EMBEDDING_DIM} dim)")
    print(f"  Milvus:    {MILVUS_HOST}:{MILVUS_PORT}")
    print("  Index:     HNSW (M=16, efConstruction=256)")
    print(
        "  Schema:    text, source, page, language, doc_type, chunk_type, chunk_index, file_hash"
    )
    print("=" * 70)

    overall_start = time.time()

    # Load progress for resume
    progress = load_progress() if resume else {}
    if progress:
        print(f"[RESUME] Found {len(progress)} previously indexed files")

    # Phase 1: Extract all documents
    print("\n--- Phase 1: Document Extraction ---")
    documents = extract_all_documents(docs_dir, max_workers=max_extract_workers)

    # Filter out empty/failed docs
    valid_docs = [d for d in documents if d.total_chars > 0]
    if resume:
        valid_docs = [d for d in valid_docs if d.file_hash not in progress]
    print(f"\n  Valid documents to process: {len(valid_docs)}")

    if not valid_docs:
        print("No documents to index!")
        return

    # Phase 2: Chunk all documents
    print("\n--- Phase 2: Chunking ---")
    chunk_start = time.time()
    all_chunks = []

    for i, doc in enumerate(valid_docs):
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        if (i + 1) % 20 == 0:
            print(
                f"  Chunked {i + 1}/{len(valid_docs)} docs ({len(all_chunks)} chunks)"
            )

    chunk_elapsed = time.time() - chunk_start
    print(
        f"  Total chunks: {len(all_chunks)} (from {len(valid_docs)} docs in {chunk_elapsed:.1f}s)"
    )

    if not all_chunks:
        print("No chunks created!")
        return

    # Stats
    lang_stats = {}
    dtype_stats = {}
    for c in all_chunks:
        lang_stats[c["language"]] = lang_stats.get(c["language"], 0) + 1
        dtype_stats[c["doc_type"]] = dtype_stats.get(c["doc_type"], 0) + 1

    print("\n  Language distribution:")
    for lang, count in sorted(lang_stats.items(), key=lambda x: -x[1]):
        print(f"    {lang}: {count} chunks")

    print("\n  Document type distribution:")
    for dtype, count in sorted(dtype_stats.items(), key=lambda x: -x[1]):
        print(f"    {dtype}: {count} chunks")

    # Phase 3: Setup Milvus
    print("\n--- Phase 3: Milvus Setup ---")
    collection = setup_milvus_collection(drop_existing=drop_existing)

    # Phase 4: Embed and Insert
    print("\n--- Phase 4: Embedding & Indexing ---")
    embeddings_client = get_embeddings_client()

    # Test embedding
    test_vec = embeddings_client.embed_query("test")
    print(f"  Embedding dimension: {len(test_vec)} (expected: {EMBEDDING_DIM})")

    index_start = time.time()
    total_inserted = insert_chunks_to_milvus(collection, all_chunks, embeddings_client)
    index_elapsed = time.time() - index_start

    # Save progress
    for doc in valid_docs:
        progress[doc.file_hash] = doc.filename
    save_progress(progress)

    # Final stats
    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)
    print(f"  Documents processed: {len(valid_docs)}")
    print(f"  Chunks indexed:      {total_inserted}")
    print(f"  Extraction time:     {chunk_elapsed:.1f}s")
    print(f"  Indexing time:       {index_elapsed:.1f}s")
    print(f"  Total time:          {total_elapsed:.1f}s")
    print("  Index type:          HNSW (M=16, efConstruction=256)")
    print(f"  Embedding model:     OpenAI text-embedding-3-small ({EMBEDDING_DIM}d)")
    print("=" * 70)

    # Verify collection
    collection.load()
    print(f"\n  Collection count: {collection.num_entities} entities")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dhara RAG Index Pipeline")
    parser.add_argument("--docs-dir", default="data/docs", help="Documents directory")
    parser.add_argument(
        "--no-drop", action="store_true", help="Don't drop existing collection"
    )
    parser.add_argument("--workers", type=int, default=4, help="Extraction workers")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last progress"
    )
    args = parser.parse_args()

    run_pipeline(
        docs_dir=args.docs_dir,
        drop_existing=not args.no_drop,
        max_extract_workers=args.workers,
        skip_ocr=args.skip_ocr,
        resume=args.resume,
    )
