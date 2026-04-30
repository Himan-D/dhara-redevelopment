"""
Document Indexer using Semantic Chunking
Rebuilds the vector store with semantic chunks for better retrieval.
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from semantic_chunker import SemanticChunker
from services.milvus_utils import get_collection, setup_local_milvus
from langchain_openai import OpenAIEmbeddings


DOCS_DIR = Path(__file__).parent.parent / "docs"
COLLECTION_NAME = os.environ.get(
    "MILVUS_COLLECTION_RAG",
    os.environ.get("MILVUS_COLLECTION", "dcpr_knowledge"),
)
BATCH_SIZE = 100


def extract_text_from_file(filepath: Path) -> str:
    """Extract text from various file formats"""
    text = ""
    method = "unknown"

    try:
        if filepath.suffix.lower() == ".pdf":
            try:
                import pypdf

                reader = pypdf.PdfReader(str(filepath))
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                method = "pypdf"
            except Exception as e1:
                try:
                    import fitz

                    doc = fitz.open(str(filepath))
                    for page in doc:
                        text += page.get_text() + "\n"
                    method = "PyMuPDF"
                except Exception as e2:
                    print(f"    PDF error: pypdf={e1}, PyMuPDF={e2}")
        elif filepath.suffix.lower() == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            method = "text"
        elif filepath.suffix.lower() == ".docx":
            try:
                from docx import Document

                doc = Document(str(filepath))
                text = "\n".join([p.text for p in doc.paragraphs])
                method = "python-docx"
            except Exception as e:
                print(f"    DOCX error: {e}")
    except Exception as e:
        print(f"    Error extracting {filepath.name}: {e}")

    if not text or len(text) < 50:
        print(f"    Skipping (too little text: {len(text) if text else 0} chars)")
        return None

    print(f"    Extracted {len(text)} chars via {method}")
    return text


def semantic_chunk_text(text: str, source: str) -> List[Tuple[str, str]]:
    """Chunk text using semantic approach"""
    chunker = SemanticChunker(
        min_chunk_size=100,
        max_chunk_size=1200,
        overlap=150,
    )
    chunks = chunker.chunk_text(text, source)
    print(f"    Created {len(chunks)} semantic chunks")
    return chunks


def find_all_files() -> List[Path]:
    """Find all document files to process"""
    extensions = [".pdf", ".txt", ".docx"]
    files = []
    skip_files = {
        "README.md",
        "INTEGRATION_IMPLEMENTATION.md",
        "GOVERNMENT_INTEGRATION.md",
    }

    if not DOCS_DIR.exists():
        print(f"Warning: {DOCS_DIR} not found, using current directory")
        doc_dir = Path(".")
    else:
        doc_dir = DOCS_DIR

    for root, dirs, filenames in os.walk(doc_dir):
        for f in filenames:
            if (
                any(f.lower().endswith(ext) for ext in extensions)
                and f not in skip_files
            ):
                files.append(Path(root) / f)

    print(f"Found {len(files)} files to process")
    return files


def process_file(filepath: Path) -> List[Tuple[str, str]]:
    """Process a single file and return chunks"""
    print(f"\nProcessing: {filepath.name}")
    text = extract_text_from_file(filepath)
    if not text:
        return []

    # Use semantic chunking
    source = str(filepath.relative_to(filepath.parent))
    chunks = semantic_chunk_text(text, source)

    return chunks


def index_chunks(chunks: List[Tuple[str, str]], collection, embeddings):
    """Index chunks into Milvus"""
    total = len(chunks)
    indexed = 0

    print(f"\nIndexing {total} chunks to Milvus...")

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        chunk_texts = [c[0] for c in batch]
        metadatas = [c[1] for c in batch]

        try:
            vectors = embeddings.embed_documents(chunk_texts)

            # Prepare data for insertion
            ids = list(range(i, i + len(batch)))
            data = [ids, chunk_texts, metadatas, vectors]

            collection.insert(data)
            indexed += len(batch)
            print(
                f"  Progress: {min(i + BATCH_SIZE, total)}/{total} ({indexed} indexed)"
            )

        except Exception as e:
            print(f"  Error indexing batch {i}: {e}")

    return indexed


def main():
    print("=" * 60)
    print("DOCUMENT INDEXER - SEMANTIC CHUNKING")
    print("=" * 60)

    # Setup Milvus
    print("\n[1/4] Setting up Milvus...")
    setup_local_milvus(COLLECTION_NAME)
    collection = get_collection(COLLECTION_NAME)

    if collection.num_entities > 0:
        print(f"  Collection already has {collection.num_entities} entities")
        response = input("  Clear and rebuild? (y/n): ")
        if response.lower() == "y":
            collection.drop()
            collection = get_collection(COLLECTION_NAME)
            print("  Collection cleared")
        else:
            print("  Using existing collection")

    # Setup embeddings
    print("\n[2/4] Initializing embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("  Using: text-embedding-3-small")

    # Find and process files
    print("\n[3/4] Processing documents...")
    files = find_all_files()

    if not files:
        print("No files found!")
        return

    all_chunks = []

    # Process files in parallel
    print("  Using parallel processing...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in as_completed(futures):
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  Error: {e}")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks created!")
        return

    # Index chunks
    print("\n[4/4] Indexing to Milvus...")
    indexed = index_chunks(all_chunks, collection, embeddings)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE - Indexed {indexed} chunks")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
