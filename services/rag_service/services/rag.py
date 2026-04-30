#!/usr/bin/env python3
"""
Production RAG System with LangGraph, Tools, Excel & Math Capabilities
"""

import os
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import requests

# LangGraph
try:
    from langchain_ollama import ChatOllama
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    ChatOllama = None

# Milvus imports
try:
    from pymilvus import (
        connections,
        Collection,
        FieldSchema,
        CollectionSchema,
        DataType,
        utility,
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

# Constants
DEFAULT_MODEL = os.environ.get("MODEL", "qwen2.5:7b")
DEFAULT_EMBEDDING = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
MILVUS_URI = os.environ.get("MILVUS_URI", "")

# Clean host for Zilliz (remove https:// if present)
if MILVUS_HOST.startswith("https://"):
    MILVUS_HOST = MILVUS_HOST.replace("https://", "")

# Data dirs
DATA_DIR = Path("data")
VECTORS_DIR = DATA_DIR / "vectors"
DOCS_DIR = DATA_DIR / "docs"
FEEDBACK_FILE = DATA_DIR / "feedback.json"
TRAINING_FILE = DATA_DIR / "training_data.json"

DATA_DIR.mkdir(exist_ok=True)
VECTORS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


# ==================== TOOLS ====================


class Tool:
    def __init__(self, name: str, description: str, function: callable):
        self.name = name
        self.description = description
        self.function = function

    def execute(self, **kwargs):
        return self.function(**kwargs)


class DocumentSearchTool:
    """Search documents for information"""

    def __init__(self, vectorstore):
        self.vs = vectorstore

    def search(self, query: str, k: int = 10) -> str:
        results = self.vs.search(query, k=k)
        if not results:
            return "No relevant information found."

        output = f"Found {len(results)} relevant sections:\n\n"
        for i, (text, score) in enumerate(results, 1):
            output += f"{i}. [Relevance: {score:.2f}]\n{text[:800]}\n\n"
        return output


class MathCalculatorTool:
    """Perform mathematical calculations"""

    def calculate(self, expression: str) -> str:
        try:
            # Handle Excel-like formulas
            expr = expression.upper().strip()

            # Extract numbers
            numbers = re.findall(r"-?\d+\.?\d*", expr)

            if not numbers:
                return "No numbers found in expression."

            nums = [float(n) for n in numbers]

            # Check operation type
            if "SUM" in expr or "ADD" in expr or "+" in expr:
                result = sum(nums)
                return f"SUM: {result}"
            elif "AVERAGE" in expr or "AVG" in expr:
                result = sum(nums) / len(nums)
                return f"AVERAGE: {result}"
            elif "MAX" in expr:
                return f"MAX: {max(nums)}"
            elif "MIN" in expr:
                return f"MIN: {min(nums)}"
            elif "MULTIPLY" in expr or "*" in expr:
                result = 1
                for n in nums:
                    result *= n
                return f"PRODUCT: {result}"
            elif "DIVIDE" in expr or "/" in expr:
                if len(nums) >= 2:
                    result = nums[0] / nums[1]
                    return f"DIVIDE: {result}"
            elif "PERCENTAGE" in expr or "%" in expr:
                if len(nums) >= 2:
                    result = (nums[0] / nums[1]) * 100
                    return f"PERCENTAGE: {result:.2f}%"
            elif "SUBTRACT" in expr or "-" in expr:
                result = nums[0] - sum(nums[1:])
                return f"SUBTRACT: {result}"
            else:
                # Basic calculation
                return f"Result: {eval(expr) if expr.isdigit() else 'Expression not recognized'}"

        except Exception as e:
            return f"Calculation error: {str(e)}"

    def statistics(self, numbers: str) -> str:
        try:
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", numbers)]
            if not nums:
                return "No numbers found."

            import statistics

            return f"""Statistics:
- Count: {len(nums)}
- Sum: {sum(nums)}
- Mean: {statistics.mean(nums):.2f}
- Median: {statistics.median(nums):.2f}
- Min: {min(nums)}
- Max: {max(nums)}
- Range: {max(nums) - min(nums)}"""
        except Exception as e:
            return f"Error: {str(e)}"


class ExcelTool:
    """Excel/Google Sheets operations"""

    def __init__(self):
        self.excel_available = False
        try:
            import pandas as pd

            self.pd = pd
            self.excel_available = True
        except:
            pass

    def read_excel(self, filepath: str, sheet: str = None) -> str:
        if not self.excel_available:
            return "Excel support not available. Install pandas and openpyxl."

        try:
            xl = self.pd.ExcelFile(filepath)
            if sheet:
                df = self.pd.read_excel(xl, sheet_name=sheet)
                return f"Sheet '{sheet}':\n{self.pd.DataFrame(df).head(20).to_string()}"

            output = f"File: {filepath}\nSheets: {xl.sheet_names}\n\n"
            for s in xl.sheet_names[:3]:
                df = self.pd.read_excel(xl, s)
                output += (
                    f"=== {s} ===\n{self.pd.DataFrame(df).head(5).to_string()}\n\n"
                )
            return output
        except Exception as e:
            return f"Error: {str(e)}"

    def analyze_data(self, filepath: str) -> str:
        if not self.excel_available:
            return "Excel not available."

        try:
            df = (
                self.pd.read_csv(filepath)
                if filepath.endswith(".csv")
                else self.pd.read_excel(filepath)
            )

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            output = (
                f"Data Analysis:\n- Rows: {len(df)}\n- Columns: {len(df.columns)}\n\n"
            )

            if numeric_cols:
                output += "Numeric Columns:\n"
                for col in numeric_cols[:5]:
                    output += f"  {col}: mean={df[col].mean():.2f}, min={df[col].min()}, max={df[col].max()}\n"

            return output
        except Exception as e:
            return f"Error: {str(e)}"

    def create_spreadsheet(self, data: str, output: str = "output.xlsx") -> str:
        if not self.excel_available:
            return "Excel not available."

        try:
            # Parse simple data
            lines = data.strip().split("\n")
            if not lines:
                return "No data provided."

            # Try to create DataFrame
            rows = []
            for line in lines:
                cols = [c.strip() for c in line.split(",")]
                rows.append(cols)

            df = self.pd.DataFrame(rows[1:], columns=rows[0] if rows else None)
            df.to_excel(output, index=False)

            return f"Created {output} with {len(df)} rows"
        except Exception as e:
            return f"Error: {str(e)}"


class WebSearchTool:
    """Real web search capability using Serper API"""

    def __init__(self):
        self.api_key = os.environ.get("SERPER_API_KEY")
        self.url = "https://google.serper.dev/search"

    def search(self, query: str) -> str:
        if not self.api_key:
            return f"[Web search simulation] Would search for: {query}\nNote: Set SERPER_API_KEY in .env for actual search."

        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = json.dumps({"q": query})

        try:
            response = requests.post(self.url, headers=headers, data=payload)
            response.raise_for_status()
            results = response.json()

            snippets = []
            if "organic" in results:
                for result in results["organic"][:5]:
                    snippets.append(
                        f"Title: {result.get('title')}\nSnippet: {result.get('snippet')}\nLink: {result.get('link')}\n"
                    )

            if not snippets:
                return "No web results found."

            return "\n".join(snippets)
        except Exception as e:
            return f"Web search error: {str(e)}"


# ==================== RAG CORE ====================


class MilvusVectorStore:
    """Milvus vector store"""

    def __init__(self, embeddings, collection_name: str = "documents"):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.collection = None
        self._connect()

    def _connect(self):
        """Connect to Milvus/Zilliz — prefers cloud URI if configured."""
        try:
            if MILVUS_URI and MILVUS_TOKEN:
                # Zilliz Cloud via URI + token
                uri = MILVUS_URI if MILVUS_URI.startswith("https://") else f"https://{MILVUS_URI}"
                print(f"Connecting to Zilliz Cloud at {uri}...")
                connections.connect(
                    alias="default",
                    uri=uri,
                    token=MILVUS_TOKEN,
                    timeout=15,
                )
            elif MILVUS_TOKEN:
                # Cloud via host + token (legacy)
                print(f"Connecting to Zilliz Cloud at {MILVUS_HOST}...")
                connections.connect(
                    alias="default",
                    host=MILVUS_HOST,
                    port=MILVUS_PORT,
                    token=MILVUS_TOKEN,
                    secure=True,
                    timeout=10,
                )
            else:
                # Local Milvus
                connections.connect(
                    alias="default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=5
                )
            print(f"[OK] Connected to Milvus/Zilliz")
            self._setup_collection()
        except Exception as e:
            print(f"[FAIL] Milvus connection failed: {e}")
            raise

    def _setup_collection(self):
        """Create or load collection with enhanced metadata schema."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"[OK] Loaded collection: {self.collection_name}")
        else:
            dim = int(
                os.environ.get("EMBEDDING_DIM", "1536")
            )  # OpenAI text-embedding-3-small
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
                ),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="page", dtype=DataType.INT32),
                FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),
                FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="chunk_index", dtype=DataType.INT32),
                FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(
                fields, description="Dhara RAG document embeddings with metadata"
            )
            self.collection = Collection(self.collection_name, schema)
            # HNSW index for fast retrieval
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 256},
            }
            self.collection.create_index(
                field_name="embedding", index_params=index_params
            )
            print(
                f"[OK] Created collection: {self.collection_name} (HNSW, {dim}d COSINE)"
            )

    def add_documents(self, documents: List) -> int:
        """Add documents to Milvus with metadata support."""
        texts = [doc.page_content for doc in documents]
        print(f"Embedding {len(texts)} docs...")

        batch_size = 10
        all_texts = []
        all_sources = []
        all_pages = []
        all_languages = []
        all_doc_types = []
        all_chunk_types = []
        all_chunk_indices = []
        all_file_hashes = []
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.embeddings.embed_documents(batch)
            all_texts.extend(batch)
            all_vectors.extend(vectors)

            # Extract metadata from documents if available
            for j, doc in enumerate(documents[i : i + batch_size]):
                meta = getattr(doc, "metadata", {}) or {}
                all_sources.append(meta.get("source", ""))
                all_pages.append(meta.get("page", 0))
                all_languages.append(meta.get("language", "en"))
                all_doc_types.append(meta.get("doc_type", "other"))
                all_chunk_types.append(meta.get("chunk_type", "paragraph"))
                all_chunk_indices.append(meta.get("chunk_index", 0))
                all_file_hashes.append(meta.get("file_hash", ""))

        # Pad metadata if documents don't have it
        while len(all_sources) < len(all_texts):
            all_sources.append("")
            all_pages.append(0)
            all_languages.append("en")
            all_doc_types.append("other")
            all_chunk_types.append("paragraph")
            all_chunk_indices.append(0)
            all_file_hashes.append("")

        entities = [
            all_texts,
            all_sources,
            all_pages,
            all_languages,
            all_doc_types,
            all_chunk_types,
            all_chunk_indices,
            all_file_hashes,
            all_vectors,
        ]
        self.collection.insert(entities)
        self.collection.flush()
        print(f"[OK] Added {len(texts)} docs to Milvus")
        return len(texts)

    def search(self, query: str, k: int = 10) -> List[tuple]:
        """Search documents with HNSW params and metadata."""
        try:
            if self.collection:
                self.collection.load()

            query_vec = self.embeddings.embed_query(query)
            # HNSW search params
            search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

            results = self.collection.search(
                data=[query_vec],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=[
                    "text",
                    "source",
                    "page",
                    "language",
                    "doc_type",
                    "chunk_type",
                ],
            )

            output = []
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    output.append(
                        (
                            hit.distance,
                            entity.get("text", ""),
                            entity.get("source", ""),
                        )
                    )
            return output
        except Exception as e:
            print(f"Search error: {e}")
            return []


class SimpleVectorStore:
    """Vector storage"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
        self.loaded_doc = None

    def load_from_file(self, filepath: Path):
        if filepath.exists():
            data = json.loads(filepath.read_text())
            self.documents = data.get("documents", [])
            self.vectors = data.get("vectors", [])
            self.loaded_doc = filepath.stem
            print(f"[OK] Loaded {len(self.documents)} docs")

    def save_to_file(self, filepath: Path):
        data = {"documents": self.documents, "vectors": self.vectors}
        filepath.write_text(json.dumps(data))

    def add_documents(self, documents: List) -> int:
        texts = [doc.page_content for doc in documents]
        print(f"Embedding {len(texts)} docs...")

        batch_size = 10
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.embeddings.embed_documents(batch)
            all_vectors.extend(vectors)

        for t, v in zip(texts, all_vectors):
            self.documents.append({"text": t})
            self.vectors.append(v)

        return len(texts)

    def search(self, query: str, k: int = 10) -> List[tuple]:
        if not self.documents:
            return []

        try:
            query_vec = self.embeddings.embed_query(query)

            results = []
            for i, vec in enumerate(self.vectors):
                if i >= len(self.documents):
                    break
                # Cosine similarity
                dot = sum(a * b for a, b in zip(query_vec, vec))
                norm1 = sum(x * x for x in query_vec) ** 0.5
                norm2 = sum(x * x for x in vec) ** 0.5
                score = dot / (norm1 * norm2 + 0.0001)
                results.append((score, self.documents[i]["text"]))

            results.sort(reverse=True)
            return results[:k]
        except Exception as e:
            print(f"Search error: {e}")
            return []


# ==================== DOCUMENT LOADER ====================


class DocumentLoader:
    @staticmethod
    def load_pdf(filepath: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        strategy: str = "semantic",
    ) -> List:
        """
        Chunk text using semantic or fixed-size approach.

        Args:
            text: Input text to chunk
            chunk_size: Max chunk size (for fixed strategy)
            overlap: Overlap between chunks
            strategy: "semantic" or "fixed"
        """
        if strategy == "semantic":
            try:
                from scripts.semantic_chunker import SemanticChunker

                chunker = SemanticChunker(
                    min_chunk_size=100,
                    max_chunk_size=chunk_size,
                    overlap=overlap,
                )
                chunks = chunker.chunk_text(text, "")
                return [
                    Document(page_content=c[0], metadata={"type": c[1]}) for c in chunks
                ]
            except ImportError:
                print("Warning: SemanticChunker not found, falling back to fixed")

        # Default to langchain's RecursiveCharacterTextSplitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.create_documents([text])


# ==================== AGENT STATE ====================


class AgentState(dict):
    """State for LangGraph agent"""

    pass


# ==================== RAG AGENT WITH LANGGRAPH ====================


class RAGAgent:
    """Main RAG Agent with LangGraph and Tools"""

    def __init__(self, model: str = DEFAULT_MODEL, use_milvus: bool = True):
        print("Initializing RAG Agent with Tools...")

        from langchain_openai import OpenAIEmbeddings

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        if LANGGRAPH_AVAILABLE and ChatOllama is not None:
            self.llm = ChatOllama(model=model)
        else:
            from langchain_openai import OpenAI

            print("LangChain Ollama not available; using OpenAI fallback.")
            self.llm = OpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

        # Use Milvus if available
        if use_milvus and MILVUS_AVAILABLE:
            try:
                collection_name = os.environ.get(
                    "MILVUS_COLLECTION_RAG",
                    os.environ.get("MILVUS_COLLECTION", "documents"),
                )
                self.vectorstore = MilvusVectorStore(self.embeddings, collection_name=collection_name)
                print(f"Using Milvus vector store on collection '{collection_name}'")
            except Exception as e:
                print(f"Milvus failed: {e}, using simple store")
                self.vectorstore = SimpleVectorStore(self.embeddings)
        else:
            self.vectorstore = SimpleVectorStore(self.embeddings)
            print("Using simple vector store")

        # Initialize tools
        self.math_tool = MathCalculatorTool()
        self.excel_tool = ExcelTool()

        # Tool registry
        self.tools = {
            "search": Tool(
                "search",
                "Search documents for information",
                lambda q, k=10: self._search(q, k),
            ),
            "calculate": Tool(
                "calculate",
                "Perform math calculations",
                lambda e: self.math_tool.calculate(e),
            ),
            "statistics": Tool(
                "statistics",
                "Calculate statistics on numbers",
                lambda n: self.math_tool.statistics(n),
            ),
            "excel_read": Tool(
                "excel_read",
                "Read Excel files",
                lambda f, s=None: self.excel_tool.read_excel(f, s),
            ),
            "excel_analyze": Tool(
                "excel_analyze",
                "Analyze Excel/CSV data",
                lambda f: self.excel_tool.analyze_data(f),
            ),
            "excel_create": Tool(
                "excel_create",
                "Create Excel spreadsheet",
                lambda d, o="output.xlsx": self.excel_tool.create_spreadsheet(d, o),
            ),
        }

        # Feedback
        self.feedback = []
        if FEEDBACK_FILE.exists():
            self.feedback = json.loads(FEEDBACK_FILE.read_text())

        self.current_doc = None
        self.query_count = 0

        print(f"[OK] Agent ready with {len(self.tools)} tools")

    def _search(self, query: str, k: int = 10) -> str:
        results = self.vectorstore.search(query, k=k)
        if not results:
            return "No relevant information found."

        output = f"Found {len(results)} relevant sections:\n\n"
        for i, (score, text) in enumerate(results, 1):
            output += f"{i}. [Score: {score:.2f}]\n{text[:500]}\n\n"
        return output

    def load_document(self, filepath: Path, force: bool = False) -> int:
        cache = VECTORS_DIR / f"{filepath.stem}.json"

        if cache.exists() and not force:
            self.vectorstore.load_from_file(cache)
            self.current_doc = filepath.stem
            return len(self.vectorstore.documents)

        print(f"Loading: {filepath.name}")
        text = DocumentLoader.load_pdf(filepath)
        print(f"Text: {len(text):,} chars")

        # Save text
        (DOCS_DIR / f"{filepath.stem}.txt").write_text(text)

        # Chunk
        docs = DocumentLoader.chunk_text(text)
        print(f"Chunks: {len(docs)}")

        # Index
        self.vectorstore.add_documents(docs)
        self.vectorstore.save_to_file(cache)
        self.current_doc = filepath.stem

        return len(docs)

    def analyze_question(self, question: str) -> Dict:
        """Analyze what tools to use"""
        question_lower = question.lower()

        tools_to_use = []

        # Check for math keywords
        math_keywords = [
            "calculate",
            "sum",
            "average",
            "total",
            "percentage",
            "multiply",
            "divide",
            "add",
            "subtract",
            "statistics",
        ]
        if any(k in question_lower for k in math_keywords):
            tools_to_use.append("calculate")

        # Check for Excel keywords
        excel_keywords = ["excel", "spreadsheet", "sheet", "csv", "data", "analyze"]
        if any(k in question_lower for k in excel_keywords):
            tools_to_use.append("excel_analyze")

        # Check for document search
        search_keywords = [
            "what",
            "explain",
            "describe",
            "define",
            "tell",
            "find",
            "list",
        ]
        if any(k in question_lower for k in search_keywords):
            tools_to_use.append("search")

        # Default: always search document
        if "search" not in tools_to_use:
            tools_to_use.insert(0, "search")

        return {
            "question": question,
            "tools": tools_to_use,
            "needs_full_analysis": any(
                k in question_lower
                for k in ["analyze", "comprehensive", "detailed", "explain everything"]
            ),
        }

    def execute_tools(self, tool_names: List[str], question: str) -> Dict[str, str]:
        """Execute multiple tools"""
        results = {}

        for tool_name in tool_names:
            if tool_name == "search":
                results["search"] = self._search(question, k=10)

            elif tool_name == "calculate":
                # Extract numbers from question
                numbers = re.findall(r"\d+\.?\d*", question)
                if numbers:
                    results["calculate"] = self.math_tool.calculate(",".join(numbers))
                else:
                    results["calculate"] = (
                        "No numbers found in question for calculation."
                    )

            elif tool_name == "excel_analyze":
                # Would need filepath
                results["excel_analyze"] = "Excel analysis - provide file path"

        return results

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer question with tool calling"""
        self.query_count += 1

        # Step 1: Analyze question
        analysis = self.analyze_question(question)

        # Step 2: Execute tools
        tool_results = self.execute_tools(analysis["tools"], question)

        # Step 3: Generate answer with context
        context = "\n\n".join(tool_results.values())

        prompt = f"""You are an expert assistant with access to various tools.

Question: {question}

Tool Results:
{context}

Document Loaded: {self.current_doc or "None"}

Instructions:
1. Answer using the tool results above
2. If information is not available, state that clearly
3. Be specific and provide details
4. If calculations were performed, show the results

Answer:"""

        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "tool_results": tool_results,
            "analysis": analysis,
            "sources": [t[:200] for t in tool_results.values() if t],
            "query_count": self.query_count,
        }

    def add_feedback(self, question: str, answer: str, rating: int, feedback: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "rating": rating,
            "feedback": feedback,
            "approved": rating >= 4,
        }
        self.feedback.append(entry)
        FEEDBACK_FILE.write_text(json.dumps(self.feedback, indent=2))
        return entry

    def get_stats(self) -> Dict:
        return {
            "queries": self.query_count,
            "documents": len(self.vectorstore.documents),
            "current_doc": self.current_doc,
            "feedback": len(self.feedback),
            "approved": sum(1 for f in self.feedback if f.get("approved")),
        }


# ==================== CLI ====================


def main():
    parser = argparse.ArgumentParser(description="RAG Agent with Tools")

    subparsers = parser.add_subparsers(dest="cmd")

    # Query
    q_parser = subparsers.add_parser("query", help="Query")
    q_parser.add_argument("pdf", help="PDF path")
    q_parser.add_argument("question", help="Question")
    q_parser.add_argument("--rebuild", action="store_true")

    # Interactive
    i_parser = subparsers.add_parser("interactive", help="Interactive")
    i_parser.add_argument("pdf", help="PDF path")
    i_parser.add_argument("--rebuild", action="store_true")

    # Calculate
    calc_parser = subparsers.add_parser("calculate", help="Calculate")
    calc_parser.add_argument("expression", help="Expression")

    # Excel
    x_parser = subparsers.add_parser("excel", help="Excel operations")
    x_parser.add_argument("file", help="Excel file")
    x_parser.add_argument("--analyze", action="store_true")
    x_parser.add_argument("--create", help="Create from data")

    # Stats
    subparsers.add_parser("stats", help="Stats")

    args = parser.parse_args()

    if args.cmd == "stats":
        agent = RAGAgent()
        s = agent.get_stats()
        print("\n=== STATS ===")
        print(f"Queries: {s['queries']}")
        print(f"Document: {s['current_doc']}")
        print(f"Chunks: {s['documents']}")
        print(f"Feedback: {s['feedback']} ({s['approved']} approved)")

    elif args.cmd == "calculate":
        agent = RAGAgent()
        result = agent.math_tool.calculate(args.expression)
        print(f"Result: {result}")

    elif args.cmd == "excel":
        agent = RAGAgent()
        if args.analyze:
            result = agent.excel_tool.analyze_data(args.file)
        else:
            result = agent.excel_tool.read_excel(args.file)
        print(result)

    elif args.cmd in ["query", "interactive"]:
        agent = RAGAgent()

        pdf = Path(args.pdf)
        if not pdf.exists():
            print(f"Error: {pdf} not found")
            return

        # Load doc
        if agent.current_doc != pdf.stem or args.rebuild:
            agent.load_document(pdf, force=args.rebuild)

        if args.cmd == "query":
            result = agent.answer(args.question)
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\n[Query: {result['query_count']}]")

        else:
            print("\n=== Interactive Mode ===")
            print(f"Document: {agent.current_doc}")
            print("Commands: quit, stats\n")

            while True:
                try:
                    q = input("Question: ").strip()
                    if q.lower() in ["quit", "exit"]:
                        break
                    if q.lower() == "stats":
                        s = agent.get_stats()
                        print(f"Queries: {s['queries']}, Chunks: {s['documents']}")
                        continue

                    result = agent.answer(q)
                    print(f"\nAnswer:\n{result['answer']}\n")
                except KeyboardInterrupt:
                    break

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
