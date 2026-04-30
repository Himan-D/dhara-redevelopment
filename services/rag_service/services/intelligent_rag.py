#!/usr/bin/env python3
"""
Enhanced Intelligent RAG System with Dynamic Knowledge Graph
Builds knowledge graph from DCPR document parsing
"""

import os
import re
import json
import uuid
import warnings
import time
import threading
import hashlib
import dataclasses
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict, OrderedDict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from fastapi import logger

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Optional imports with proper error handling
try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    from pymilvus import connections, Collection
except ImportError:
    connections = None
    Collection = None

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().strip().split("\n"):
        if "=" in line:
            key, val = line.split("=", 1)
            os.environ[key] = val


@dataclass
class QueryContext:
    original_query: str
    intent: str
    entities: List[str]
    clauses: List[str]
    topic: str
    subtopics: List[str]
    is_compound: bool
    compound_parts: List[str]
    location: Optional[str] = None
    units: Dict[str, str] = field(default_factory=dict)
    needs_market_data: bool = False
    needs_regulatory_data: bool = True
    is_technical: bool = False


@dataclass
class SearchResult:
    query: str
    text: str
    score: float
    clauses: List[str]
    tables: List[str]
    relevance_tags: List[str]
    source: str = ""
    page: int = 0
    language: str = "en"
    doc_type: str = "other"
    result_type: str = "local"  # "local" or "web"
    web_url: str = ""


@dataclass
class ClauseInfo:
    clause_id: str
    text: str
    related_clauses: Set[str] = field(default_factory=set)
    tables: Set[str] = field(default_factory=set)
    topics: Set[str] = field(default_factory=set)


class DynamicKnowledgeGraph:
    """Dynamically builds knowledge graph from DCPR document"""

    def __init__(self):
        self.clauses: Dict[str, ClauseInfo] = {}
        self.tables: Dict[str, str] = {}
        self.topic_to_clauses: Dict[str, Set[str]] = defaultdict(set)
        self.clause_relationships: Dict[str, List[str]] = defaultdict(list)
        self._initialized = False

    def build_from_chunks(self, chunks: List[str]):
        """Extract knowledge graph from ALL document chunks"""
        if self._initialized:
            return

        clause_pattern = re.compile(
            r"(?:Clause|Regulation|Rule|Sub-section|धारा|नियम|कलम|उपधारा)\s*(\d+(?:\([\w]+\))?)",
            re.IGNORECASE,
        )
        table_pattern = re.compile(
            r"(?:Table|तालिका)\s*(?:No\.?\s*)?(\d+[a-zA-Z]?)", re.IGNORECASE
        )

        cross_ref_pattern = re.compile(
            r"(?:Clause|Regulation|Table|धारा|नियम|तालिका)\s*(\d+(?:\([\w]+\))?)[,;\s]*(?:and|&|,|आणि|तथा)\s*(?:Clause|Regulation|Table|धारा|नियम|तालिका)?\s*(\d+(?:\([\w]+\))?)",
            re.IGNORECASE,
        )

        # Process ALL chunks
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            clauses_in_chunk = clause_pattern.findall(chunk)
            tables_in_chunk = table_pattern.findall(chunk)
            cross_refs = cross_ref_pattern.findall(chunk)

            for clause in clauses_in_chunk:
                clause_key = (
                    f"Clause {clause}" if not clause.startswith("Clause") else clause
                )
                if clause_key not in self.clauses:
                    self.clauses[clause_key] = ClauseInfo(
                        clause_id=clause_key, text=chunk[:500]
                    )

                self.clauses[clause_key].tables.update(
                    [f"Table {t}" for t in tables_in_chunk]
                )

                for ref1, ref2 in cross_refs:
                    ref1_key = (
                        f"Clause {ref1}" if not ref1.startswith("Clause") else ref1
                    )
                    ref2_key = (
                        f"Clause {ref2}" if not ref2.startswith("Clause") else ref2
                    )
                    self.clauses[clause_key].related_clauses.add(ref1_key)
                    self.clauses[clause_key].related_clauses.add(ref2_key)
                    self.clause_relationships[ref1_key].append(ref2_key)
                    self.clause_relationships[ref2_key].append(ref1_key)

            for table in tables_in_chunk:
                table_key = f"Table {table}"
                if table_key not in self.tables:
                    self.tables[table_key] = chunk[:500]

            if (i + 1) % 500 == 0:
                print(
                    f"  KG Progress: {i + 1}/{total} chunks, {len(self.clauses)} clauses"
                )

        self._extract_topics(chunks)
        self._initialized = True
        print(f"  KG Complete: {len(self.clauses)} clauses, {len(self.tables)} tables")

    def _extract_topics(self, chunks: List[str]):
        """Extract topics from chunks using keyword clustering"""
        topic_keywords = {
            "FSI": [
                "fsi",
                "floor space index",
                "built-up area",
                "built up",
                "फ्लोर स्पेस इंडेक्स",
                "एफएसआय",
                "एफएसआई",
            ],
            "Parking": [
                "parking",
                "vehicle",
                "car park",
                "parking space",
                "वाहनतळ",
                "पार्किंग",
            ],
            "Open Space": [
                "open space",
                "marginal",
                "setback",
                "osr",
                "मोकळी जागा",
                "अंतर",
            ],
            "Redevelopment": [
                "redevelopment",
                "reconstruction",
                "rebuilding",
                "पुनर्विकास",
                "पुनर्बांधकाम",
            ],
            "Premium": [
                "premium",
                "premium charge",
                "additional fsi",
                "प्रीमियम",
                "अतिरिक्त एफएसआय",
            ],
            "Height": ["height", "storey", "stories", "stilt", "उंची", "मजला"],
            "Rehabilitation": [
                "rehabilitation",
                "tenant",
                "occupier",
                "rehousing",
                "पुनर्वसन",
                "भाडेकरू",
            ],
            "TDR": ["tdr", "transferable", "development rights", "हस्तांतरणीय विकास हक्क"],
            "Margins": ["marginal distance", "margin", "setback", "अंतर", "सेटबॅक"],
            "Loading": ["loading", "unloading", "bay", "लोडिंग", "उतरणी"],
        }

        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            for clause in list(self.clauses.keys()):
                if clause in chunk:
                    for topic, keywords in topic_keywords.items():
                        if any(kw in chunk_lower for kw in keywords):
                            self.clauses[clause].topics.add(topic)
                            self.topic_to_clauses[topic].add(clause)

    def get_related(self, clause_or_term: str) -> List[str]:
        """Get related clauses for a given clause or term"""
        clause_key = (
            clause_or_term
            if clause_or_term.startswith("Clause")
            else f"Clause {clause_or_term}"
        )
        related = set()

        if clause_key in self.clauses:
            related.update(self.clauses[clause_key].related_clauses)
            for topic in self.clauses[clause_key].topics:
                related.update(self.topic_to_clauses.get(topic, set()))

        related.discard(clause_key)
        return list(related)[:15]

    def get_clauses_by_topic(self, topic: str) -> List[str]:
        """Get all clauses related to a topic"""
        return list(self.topic_to_clauses.get(topic, set()))

    def get_clause_info(self, clause_id: str) -> Optional[ClauseInfo]:
        return self.clauses.get(
            clause_id if clause_id.startswith("Clause") else f"Clause {clause_id}"
        )


class ConversationMemory:
    """Simple conversation memory with optional Redis persistence."""

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self.messages: List[Dict] = []
        self.query_count = 0
        # Try to restore from Redis if session_id is provided
        if session_id:
            self._load_from_redis()

    def _redis_key(self) -> str:
        return f"mem:{self.session_id}"

    def _load_from_redis(self):
        try:
            cached = SessionRAG._get_from_cache(self._redis_key())
            if cached and isinstance(cached, list):
                self.messages = cached
        except Exception:
            pass

    def _persist_to_redis(self):
        if self.session_id:
            try:
                SessionRAG._set_in_cache(self._redis_key(), self.messages, ttl=86400)
            except Exception:
                pass

    def add(self, role: str, content: str):
        self.messages.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )
        self._persist_to_redis()

    def get_history(self, last_k: int = 10) -> List[Dict]:
        return self.messages[-last_k * 2 :]

    def get_context_for_query(self, current_query: str) -> str:
        if len(self.messages) < 2:
            return ""
        recent = self.messages[-6:]
        parts = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content'][:150]}...")
        return "\n".join(parts)

    def clear(self):
        self.messages = []
        self._persist_to_redis()


class SessionManager:
    """Manage multiple RAG sessions"""

    _sessions: OrderedDict = OrderedDict()
    _lock = threading.Lock()
    _knowledge_graph: Optional[DynamicKnowledgeGraph] = None
    _chunks: List[str] = []
    _MAX_SESSIONS: int = int(os.environ.get("MAX_RAG_SESSIONS", "200"))

    @classmethod
    def initialize_knowledge_graph(cls, chunks: List[str]):
        """Initialize knowledge graph from document chunks"""
        if cls._knowledge_graph is None:
            cls._knowledge_graph = DynamicKnowledgeGraph()
            cls._chunks = chunks
            cls._knowledge_graph.build_from_chunks(chunks)
            print(
                f"[OK] Knowledge graph built: {len(cls._knowledge_graph.clauses)} clauses, {len(cls._knowledge_graph.tables)} tables"
            )

    @classmethod
    def get_knowledge_graph(cls) -> DynamicKnowledgeGraph:
        if cls._knowledge_graph is None:
            # Initialize it first so we never return None
            cls._knowledge_graph = DynamicKnowledgeGraph()

            try:
                # Read texts from vector cache for KG building
                vector_dir = Path("data/vectors")
                if vector_dir.exists():
                    caches = list(vector_dir.glob("*.json"))
                    if caches:
                        cache = caches[0]
                        print(f"Loading knowledge graph from {cache.name}...")
                        data = json.loads(cache.read_text())
                        if isinstance(data, list):
                            texts = [d.get("text", "") for d in data if d.get("text")]
                            cls.initialize_knowledge_graph(texts)
                    else:
                        print("No vector cache found for knowledge graph.")
            except Exception as e:
                print(f"Error loading knowledge graph: {e}")

        return cls._knowledge_graph

    @classmethod
    def get_or_create_session(
        cls, session_id: Optional[str] = None
    ) -> Tuple["SessionRAG", str]:
        with cls._lock:
            if session_id and session_id in cls._sessions:
                # LRU touch: move to most-recently-used end
                cls._sessions.move_to_end(session_id)
                return cls._sessions[session_id], session_id

            new_id = session_id or str(uuid.uuid4())[:8]
            if new_id not in cls._sessions:
                # Evict least-recently-used session if at capacity
                if len(cls._sessions) >= cls._MAX_SESSIONS:
                    cls._sessions.popitem(last=False)
                cls._sessions[new_id] = SessionRAG(session_id=new_id)
            return cls._sessions[new_id], new_id

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        with cls._lock:
            if session_id in cls._sessions:
                del cls._sessions[session_id]
                return True
            return False

    @classmethod
    def list_sessions(cls) -> List[Dict]:
        with cls._lock:
            return [
                {"session_id": sid, "query_count": rag.query_count}
                for sid, rag in cls._sessions.items()
            ]


class SessionRAG:
    """RAG instance for a single session"""

    _reranker = None
    _reranker_checked = False
    _vs_failed = False
    _llm_warmed = False

    # Shared vectorstore and embeddings — one instance for all sessions
    _vectorstore = None
    _embeddings = None
    _vs_lock = threading.Lock()

    # In-process LRU embedding cache
    _embed_cache: OrderedDict = OrderedDict()
    _embed_cache_lock = threading.Lock()
    _EMBED_CACHE_MAX = 512

    # Redis cache configuration
    _redis_client = None
    _CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))  # Default 1 hour

    @classmethod
    def _get_redis_client(cls):
        """Get or create Redis client."""
        if cls._redis_client is None:
            redis_url = os.environ.get("REDIS_URL", "")
            if redis_url:
                try:
                    import redis

                    cls._redis_client = redis.from_url(redis_url, decode_responses=True)
                    # Test connection
                    cls._redis_client.ping()
                    print("[Redis] Connected to Redis cache")
                except Exception as e:
                    print(f"[Redis] Connection failed: {e}")
                    cls._redis_client = False  # Mark as failed to avoid retry
            else:
                cls._redis_client = False
        return cls._redis_client if cls._redis_client else None

    @classmethod
    def _get_from_cache(cls, key: str):
        """Get value from Redis cache."""
        client = cls._get_redis_client()
        if client:
            try:
                import json

                data = client.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                pass
        return None

    @classmethod
    def _set_in_cache(cls, key: str, value, ttl: int = None):
        """Set value in Redis cache with optional TTL override."""
        client = cls._get_redis_client()
        if client:
            try:
                import json

                client.setex(
                    key, ttl if ttl is not None else cls._CACHE_TTL, json.dumps(value)
                )
            except Exception:
                pass

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.query_count = 0
        self._client = None
        self._llm = None
        self._memory = ConversationMemory(session_id=session_id)
        self._knowledge_graph = SessionManager.get_knowledge_graph()
        self._init_llm()
        self._init_embeddings()  # Pre-warm shared embeddings if not yet done
        if os.environ.get("ENABLE_RERANKER", "true").lower() == "true":
            self._init_reranker()

    @classmethod
    def _init_embeddings(cls):
        """Pre-warm shared embeddings on startup for faster first query."""
        if cls._embeddings is not None:
            return

        if OpenAIEmbeddings is None:
            print("[Embeddings] langchain-openai not installed")
            return

        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key.startswith("sk-"):
                cls._embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small", api_key=api_key
                )
                cls._embeddings.embed_query("test")
                print("[OK] Shared embeddings pre-warmed")
        except Exception as e:
            print(f"[Embeddings] Pre-warm failed: {e}")

    def _init_llm(self):
        """Initialize LLM - Ollama primary, OpenAI fallback"""
        api_key = os.environ.get("OPENAI_API_KEY", "")

        # Use Ollama if API key is empty or placeholder (not valid)
        if not api_key or api_key in ["", "sk-placeholder", "sk-proj-"]:
            self._model = os.environ.get("MODEL", "glm4:latest")
            self._use_openai = False
            print(f"[OK] LLM initialized: {self._model} (Ollama)")
            return

        # Try OpenAI if key is provided and valid
        if OpenAI is None:
            self._model = os.environ.get("MODEL", "glm4:latest")
            self._use_openai = False
            print(
                f"[OK] LLM initialized: {self._model} (Ollama fallback - OpenAI package missing)"
            )
            return

        try:
            base_url = os.environ.get("OPENAI_BASE_URL", None)
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            test_client = OpenAI(**kwargs)
            self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self._use_openai = True
            print(f"[OK] LLM initialized: {self._model} (OpenAI-compatible)")
        except Exception as e:
            # Fallback to Ollama on any error
            self._model = os.environ.get("MODEL", "glm4:latest")
            self._use_openai = False
            print(f"[OK] LLM initialized: {self._model} (Ollama fallback - {e})")

    @classmethod
    def _init_reranker(cls):
        """Initialize multilingual reranker model (supports en/mr/hi)."""
        if cls._reranker is None and not cls._reranker_checked:
            cls._reranker_checked = True
            if CrossEncoder is None:
                print("Reranker warning: sentence-transformers package missing")
                return
            try:
                print("Loading multilingual reranker (BAAI/bge-reranker-v2-m3)...")
                # Try multilingual reranker first (best for en/mr/hi)
                try:
                    cls._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
                    print("[OK] Multilingual reranker loaded (bge-reranker-v2-m3)")
                except Exception:
                    # Fallback to English-only
                    cls._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    print("[OK] Reranker loaded (ms-marco fallback)")
            except Exception as e:
                print(f"Reranker warning (could be memory or CPU limit): {e}")
                cls._reranker = None

    @property
    def client(self):
        if self._client is None:
            if self._use_openai:
                if OpenAI is None:
                    raise ImportError("OpenAI package is required for OpenAI models")
                api_key = os.environ.get("OPENAI_API_KEY", "")
                kwargs = {"api_key": api_key}
                base_url = os.environ.get("OPENAI_BASE_URL", "")
                if base_url:
                    kwargs["base_url"] = base_url
                self._client = OpenAI(**kwargs)
            else:
                from langchain_ollama import ChatOllama

                self._client = ChatOllama(model=self._model)
        return self._client

    def _search_web(self, query: str) -> tuple:
        """Search web using DuckDuckGo + SerpApi in parallel. Returns (text, sources_list)."""
        # Always check environment variable first - failsafe to prevent any web search when disabled
        if os.environ.get("ENABLE_WEB_SEARCH", "true").lower() != "true":
            return "", []

        api_key = os.environ.get("SERP_API_KEY", "") or os.environ.get(
            "SERPER_API_KEY", ""
        )

        # Run DuckDuckGo and SerpApi in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            ddg_future = executor.submit(self._search_duckduckgo, query)
            serp_future = (
                executor.submit(self._search_serpapi, query, api_key)
                if api_key
                else None
            )

            # Get DuckDuckGo results
            ddg_text, ddg_sources = ddg_future.result() or ("", [])

            # Get SerpApi results
            serp_text, serp_sources = ("", [])
            if serp_future:
                try:
                    serp_text, serp_sources = serp_future.result(timeout=5)
                except TimeoutError:
                    pass

        # Combine results - use SerpApi as primary (better quality), DuckDuckGo as extra sources
        combined_text = serp_text or ddg_text
        combined_sources = serp_sources + ddg_sources

        return combined_text, combined_sources[:10]  # Return top 10 sources

    def _search_serpapi(self, query: str, api_key: str) -> tuple:
        """Search using SerpApi Google AI Mode. Returns (text, sources_list)."""
        try:
            from serpapi import Client

            client = Client(api_key=api_key)
            results = client.search({"engine": "google_ai_mode", "q": query})

            snippets = []
            sources = []

            if "text_blocks" in results:
                for block in results["text_blocks"][:5]:
                    text = block.get("text", "")
                    if text:
                        snippets.append(text[:500])
                        sources.append(
                            {"title": "Google AI", "url": "", "snippet": text[:200]}
                        )

            if "reconstructed_markdown" in results:
                markdown = results["reconstructed_markdown"]
                if markdown and not snippets:
                    snippets.append(markdown[:1000])

            return "\n\n".join(snippets), sources

        except ImportError:
            print("[Web search] SerpApi library not installed")
            return "", []
        except Exception as e:
            print(f"[Web search] SerpApi error: {e}")
            return "", []

    def _search_duckduckgo(self, query: str) -> tuple:
        """Search using DuckDuckGo Instant Answer API (free). Returns (text, sources_list)."""
        try:
            import requests

            # Use DuckDuckGo HTML search (more reliable than instant answer API)
            params = {
                "q": query,
                "kl": "in-en",  # India region
                "kd": "-1",
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            # Try the lite version which is easier to parse
            response = requests.get(
                "https://lite.duckduckgo.com/lite/",
                params=params,
                headers=headers,
                timeout=15,
            )

            if response.status_code != 200:
                print(f"[Web search] DuckDuckGo returned {response.status_code}")
                return "", []

            # Parse results from HTML
            from html.parser import HTMLParser

            class DDLiteParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results = []
                    self.current = {}
                    self.capture = None
                    self.depth = 0

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == "a" and "href" in attrs_dict:
                        href = attrs_dict["href"]
                        if href.startswith("http") and "duckduckgo" not in href:
                            self.current["url"] = href
                            self.capture = "title"
                    elif (
                        tag == "td" and attrs_dict.get("class", "") == "result-snippet"
                    ):
                        self.capture = "snippet"

                def handle_data(self, data):
                    data = data.strip()
                    if data and self.capture == "title":
                        self.current["title"] = data
                        self.capture = None
                    elif data and self.capture == "snippet":
                        self.current["snippet"] = data
                        if "url" in self.current:
                            self.results.append(self.current.copy())
                        self.current = {}
                        self.capture = None

            parser = DDLiteParser()
            parser.feed(response.text)

            snippets = []
            sources = []
            for r in parser.results[:5]:
                title = r.get("title", "Unknown")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                snippets.append(f"[{title}] {url}\n{snippet}")
                sources.append({"title": title, "url": url, "snippet": snippet[:200]})

            return "\n\n".join(snippets), sources

        except Exception as e:
            print(f"[Web search] DuckDuckGo error: {e}")
            return "", []

    @property
    def vectorstore(self):
        # Fast path: already initialized
        if SessionRAG._vectorstore is not None:
            return SessionRAG._vectorstore
        if SessionRAG._vs_failed:
            return None
        # Slow path: initialize once with double-checked locking
        with SessionRAG._vs_lock:
            if SessionRAG._vectorstore is not None:
                return SessionRAG._vectorstore
            if SessionRAG._vs_failed:
                return None
            if connections is None or Collection is None:
                print("Vectorstore warning: pymilvus package missing")
                SessionRAG._vs_failed = True
                return None
            try:
                milvus_host = os.environ.get("MILVUS_HOST", "localhost")
                milvus_port = os.environ.get("MILVUS_PORT", "19530")
                milvus_token = os.environ.get("MILVUS_TOKEN", "")
                milvus_uri = os.environ.get("MILVUS_URI", "")

                if milvus_uri and milvus_token:
                    # Zilliz Cloud via URI
                    uri = (
                        milvus_uri
                        if milvus_uri.startswith("https://")
                        else f"https://{milvus_uri}"
                    )
                    print(f"Connecting to Zilliz Cloud at {uri}...")
                    connections.connect(
                        alias="default",
                        uri=uri,
                        token=milvus_token,
                        timeout=15,
                    )
                elif milvus_token:
                    connections.connect(
                        alias="default",
                        host=milvus_host,
                        port=milvus_port,
                        token=milvus_token,
                        secure=True,
                        timeout=15,
                    )
                else:
                    connections.connect(
                        alias="default", host=milvus_host, port=milvus_port, timeout=15
                    )
                coll_name = os.environ.get(
                    "MILVUS_COLLECTION_RAG",
                    os.environ.get("MILVUS_COLLECTION", "documents"),
                )
                vs = Collection(coll_name)
                vs.load()
                SessionRAG._vectorstore = vs
                print(
                    f"[OK] Connected to Milvus/Zilliz — collection '{coll_name}' loaded"
                )
            except Exception as e:
                print(f"Vectorstore connection failed: {e}")
                SessionRAG._vs_failed = True
        return SessionRAG._vectorstore

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector with LRU in-process cache + Redis overflow."""
        key = hashlib.sha256(text.encode()).hexdigest()[:16]

        # 1. In-process LRU cache
        with SessionRAG._embed_cache_lock:
            if key in SessionRAG._embed_cache:
                SessionRAG._embed_cache.move_to_end(key)
                return list(SessionRAG._embed_cache[key])

        # 2. Redis cache
        cached = self._get_from_cache(f"emb:{key}")
        if cached:
            with SessionRAG._embed_cache_lock:
                if len(SessionRAG._embed_cache) >= SessionRAG._EMBED_CACHE_MAX:
                    SessionRAG._embed_cache.popitem(last=False)
                SessionRAG._embed_cache[key] = cached
            return cached

        # 3. Compute via API (cache miss)
        if SessionRAG._embeddings is None:
            raise RuntimeError("Embeddings not initialized")
        vec = SessionRAG._embeddings.embed_query(text)
        self._set_in_cache(f"emb:{key}", vec, ttl=86400)  # 24h TTL
        with SessionRAG._embed_cache_lock:
            if len(SessionRAG._embed_cache) >= SessionRAG._EMBED_CACHE_MAX:
                SessionRAG._embed_cache.popitem(last=False)
            SessionRAG._embed_cache[key] = vec
        return vec

    def _search_local(
        self,
        query: str,
        k: int = 10,
        doc_type_filter: str = None,
        precomputed_vector: List[float] = None,
    ):
        """Search local Milvus with full metadata schema."""
        try:
            if not self.vectorstore:
                return []

            query_vec = (
                precomputed_vector
                if precomputed_vector is not None
                else self._get_embedding(query)
            )

            # HNSW search params
            search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

            # Filter by doc_type if specified
            expr = None
            if doc_type_filter:
                expr = f'doc_type == "{doc_type_filter}"'

            # For local Milvus - text only
            results = self.vectorstore.search(
                data=[query_vec],
                anns_field="embedding",
                param=search_params,
                limit=k,
                expr=expr,
                output_fields=["text"],
            )

            output = []
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    output.append(
                        SearchResult(
                            query=query,
                            text=entity.get("text", ""),
                            score=hit.distance,
                            clauses=[],
                            tables=[],
                            relevance_tags=[],
                            source=entity.get("source", ""),
                            page=entity.get("page", 0),
                            language=entity.get("language", "en"),
                            doc_type=entity.get("doc_type", "other"),
                            result_type="local",
                        )
                    )
            return output
        except Exception as e:
            err_msg = str(e)
            if "dimension" in err_msg.lower() or "size(byte)" in err_msg:
                return []
            if "not exist" in err_msg.lower():
                print("[SEARCH] Schema error: collection may need reindexing")
                return []
            print(f"[SEARCH] Error: {err_msg[:100]}")
            return []

    def _analyze_query(self, question: str, history_context: str = "") -> QueryContext:
        """Analyze query to extract intent, entities, topic, location and units"""
        # Cache context-free analyses (deterministic for same question + no history)
        if not history_context:
            qctx_key = f"qctx:{hashlib.sha256(question.lower().strip().encode()).hexdigest()[:16]}"
            cached = self._get_from_cache(qctx_key)
            if cached:
                try:
                    return QueryContext(**cached)
                except Exception:
                    pass  # fall through to LLM call if cached data is stale

        prompt = f"""Analyze this query about urban planning/real estate in India:

Query: {question}
History: {history_context}

Return JSON with:
- "intent": "technical_lookup" (for margins, FSI, parking, height, or specific rules), "feasibility_analysis" (for ROI, investment, or general project viability), or "explain".
- "is_technical": true if the query is about specific regulations, numbers, or rules.
- "entities": Key entities mentioned - include ALL of: area/locality names (Kothrud, Hinjewadi, Wakad, etc.), FSI, TDR, Premium, Scheme numbers (33(7B), 33(11), etc.), road widths, zone types (residential, commercial, industrial).
- "topic": Main topic (e.g., "Side Margins", "FSI", "Parking", "TDR", "Premium", "Plot Feasibility", "Market Analysis").
- "subtopics": Related topics.
- "location": Specific area/locality mentioned (e.g., "Kothrud", "Prabhadevi", "Worli"). Also extract city (Pune/Mumbai). Default to "Mumbai" if Mumbai localities or "DCPR 2034" are mentioned, otherwise default to "Pune".
- "units": Extract area, building height, and road width from query. Convert sq ft to sq m.
   Example: if query has "2000 sq ft", return {{"original": "2000 sq ft", "sq_m": 185.8, "road_width": null}}
   Example: if query has "12m road", return {{"road_width": 12}}
   Example: if query has "15m height", return {{"building_height": 15}}
- "is_compound": true if multiple questions.
- "compound_parts": list of sub-questions if compound.
- "needs_market_data": true ONLY if query explicitly asks about prices, rates, ROI, investment potential, or current market conditions.
- "needs_regulatory_data": true if query asks about FSI, regulations, permissions, compliance, zoning rules.

IMPORTANT: If the query is about "side margins", "setbacks", "height limits", "FSI tables", or "redevelopment schemes (33(7), 33(20), etc.)", it is a TECHNICAL lookup.
"""
        try:
            kwargs = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            if self._use_openai:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kwargs)
            data = json.loads(response.choices[0].message.content)
            entities = data.get("entities", [])
            if isinstance(entities, bool) or entities is None:
                entities = []
            elif isinstance(entities, str):
                entities = [entities]
            elif isinstance(entities, dict):
                entities = list(entities.values())
            elif not isinstance(entities, list):
                entities = []

            subtopics = data.get("subtopics", [])
            if isinstance(subtopics, bool) or subtopics is None:
                subtopics = []
            elif isinstance(subtopics, str):
                subtopics = [subtopics]
            elif not isinstance(subtopics, list):
                subtopics = []

            compound_parts = data.get("compound_parts", [])
            if isinstance(compound_parts, bool) or compound_parts is None:
                compound_parts = []
            elif isinstance(compound_parts, str):
                compound_parts = [compound_parts]
            elif not isinstance(compound_parts, list):
                compound_parts = []

            units = data.get("units", {})
            if isinstance(units, bool) or units is None:
                units = {}
            elif isinstance(units, str):
                try:
                    units = json.loads(units)
                except:
                    units = {}
            elif not isinstance(units, dict):
                units = {}

            result = QueryContext(
                original_query=question,
                intent=str(data.get("intent", "explain"))
                if data.get("intent")
                else "explain",
                entities=entities,
                clauses=[],
                topic=str(data.get("topic", "general"))
                if data.get("topic")
                else "general",
                subtopics=subtopics,
                is_compound=bool(data.get("is_compound", False)),
                compound_parts=compound_parts,
                location=str(data.get("location", "Pune"))
                if data.get("location")
                else "Pune",
                units=units,
                needs_market_data=bool(data.get("needs_market_data", False)),
                needs_regulatory_data=bool(data.get("needs_regulatory_data", True)),
                is_technical=bool(data.get("is_technical", False)),
            )
            if not history_context:
                self._set_in_cache(qctx_key, dataclasses.asdict(result), ttl=7200)
            return result
        except Exception as e:
            print(f"Query analysis error: {e}")
            return QueryContext(
                original_query=question,
                intent="explain",
                entities=[],
                clauses=[],
                topic="general",
                subtopics=[],
                is_compound=False,
                compound_parts=[],
                location="Pune",
                units={},
                needs_market_data=False,
                needs_regulatory_data=True,
                is_technical=False,
            )

    def _expand_queries(self, context: QueryContext) -> List[str]:
        """Expand query for better retrieval using location, units, and language."""
        queries = [context.original_query]

        location = context.location or "Pune"
        reg_type = "UDCPR" if str(location).lower() == "pune" else "DCPR 2034"

        # Detect if query is in Devanagari script
        query_str = str(context.original_query) if context.original_query else ""
        has_devanagari = any("\u0900" <= c <= "\u097f" for c in query_str)

        # If Devanagari query, also add English equivalent terms
        if has_devanagari:
            devanagari_to_english = {
                "एफएसआय": "FSI",
                "एफएसआई": "FSI",
                "फ्लोर स्पेस इंडेक्स": "FSI",
                "पार्किंग": "parking",
                "वाहनतळ": "parking",
                "प्रीमियम": "premium",
                "अतिरिक्त": "additional",
                "पुनर्विकास": "redevelopment",
                "हस्तांतरणीय": "transferable",
                "उंची": "height",
                "मजला": "storey",
                "अंतर": "marginal distance",
                "सेटबॅक": "setback",
                "तालिका": "table",
                "धारा": "clause",
                "नियम": "regulation",
                "कलम": "clause",
                "निवासी": "residential",
                "व्यावसायिक": "commercial",
            }
            eng_terms = []
            for dev, eng in devanagari_to_english.items():
                query_str = (
                    str(context.original_query).lower()
                    if context.original_query
                    else ""
                )
                if dev in query_str:
                    eng_terms.append(eng)
            if eng_terms:
                queries.append(f"{' '.join(eng_terms)} {reg_type}")
            # Also add the query with just English terms
            queries.append(f"{reg_type} {context.topic} regulations")

        # Extract locality names from entities (e.g., Kothrud, Hinjewadi)
        entity_list = context.entities if isinstance(context.entities, list) else []
        localities = [
            str(e)
            for e in entity_list
            if e and isinstance(e, str) and len(e) > 2 and e[0].isupper()
        ]

        # Core FSI/Parking queries that always apply
        core_terms = [
            "FSI",
            "parking",
            "marginal distance",
            "setback",
            "height",
            "TDR",
            "premium",
        ]
        for term in core_terms[:3]:
            queries.append(f"{term} {reg_type} Table 12")
            if localities:
                queries.append(f"{term} {localities[0]} {reg_type}")

        # Topic-specific queries
        if context.topic:
            queries.append(f"{context.topic} {reg_type} regulations")
            if localities:
                queries.append(f"{context.topic} {localities[0]}")

        # Entity queries
        if entity_list:
            for entity in entity_list[:3]:
                if isinstance(entity, str):
                    queries.append(f"{entity} {reg_type}")

        # Unit-based query (CRITICAL for FSI)
        if context.units and "sq_m" in context.units:
            sq_m = context.units["sq_m"]
            queries.append(f"FSI for plot area {sq_m} sq m {reg_type}")
            queries.append(f"FSI table for plots up to {sq_m} sq m")

        # Road width queries
        if context.units and "road_width" in context.units:
            rw = context.units["road_width"]
            queries.append(f"FSI road width {rw} meters {reg_type}")

        # Mumbai Redevelopment Schemes (Specific for Mumbai context)
        if (
            reg_type == "DCPR 2034"
            or "redevelop" in str(context.original_query).lower()
        ):
            mumbai_schemes = ["33(7)(B)", "33(20)(B)", "33(9)", "30(A)", "33(12)(B)"]
            for scheme in mumbai_schemes:
                queries.append(
                    f"Regulation {scheme} Mumbai DCPR 2034 eligibility requirements"
                )
            queries.append("Mumbai DCPR 2034 Redevelopment Scheme Eligibility Table")

        # Keep only unique queries while preserving order
        unique_queries = []
        seen = set()
        for q in queries:
            q_normalized = str(q).strip()
            if q_normalized and q_normalized not in seen:
                unique_queries.append(q_normalized)
                seen.add(q_normalized)

        return unique_queries[:15]

    def _multi_search(
        self, queries: List[str], context: QueryContext, k: int
    ) -> List[SearchResult]:
        """Search multiple queries with batched embedding and combine results."""
        all_results = []
        seen_texts = set()

        unique_queries = list(dict.fromkeys(queries))  # preserve order, deduplicate

        # Pre-compute embeddings: check cache per query, batch-embed all misses in one API call
        query_vecs: Dict[str, List[float]] = {}
        misses: List[str] = []
        for q in unique_queries:
            key = hashlib.sha256(q.encode()).hexdigest()[:16]
            with SessionRAG._embed_cache_lock:
                if key in SessionRAG._embed_cache:
                    SessionRAG._embed_cache.move_to_end(key)
                    query_vecs[q] = list(SessionRAG._embed_cache[key])
                    continue
            cached = self._get_from_cache(f"emb:{key}")
            if cached:
                query_vecs[q] = cached
                with SessionRAG._embed_cache_lock:
                    if len(SessionRAG._embed_cache) >= SessionRAG._EMBED_CACHE_MAX:
                        SessionRAG._embed_cache.popitem(last=False)
                    SessionRAG._embed_cache[key] = cached
            else:
                misses.append(q)

        if misses and SessionRAG._embeddings is not None:
            try:
                vecs = SessionRAG._embeddings.embed_documents(misses)
                for q, vec in zip(misses, vecs):
                    query_vecs[q] = vec
                    key = hashlib.sha256(q.encode()).hexdigest()[:16]
                    self._set_in_cache(f"emb:{key}", vec, ttl=86400)
                    with SessionRAG._embed_cache_lock:
                        if len(SessionRAG._embed_cache) >= SessionRAG._EMBED_CACHE_MAX:
                            SessionRAG._embed_cache.popitem(last=False)
                        SessionRAG._embed_cache[key] = vec
            except Exception as e:
                print(f"[Embed] Batch embed error: {e}")

        for query in queries:
            try:
                results = self._search_local(
                    query, k=k, precomputed_vector=query_vecs.get(query)
                )
                for r in results:
                    if r.text and r.text not in seen_texts:
                        seen_texts.add(r.text)
                        r.relevance_tags = [context.topic]
                        all_results.append(r)
            except Exception as e:
                print(f"Search error for '{query}': {e}")

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[: k * 2]

    def _extract_applicable_regulations(self, results: List[SearchResult]) -> List[str]:
        """Detect known DCPR/UDCPR scheme references from local retrievals."""
        import re

        schemes = set()
        pattern = re.compile(
            r"\b(?:30\(A\)|33\(\d+\)(?:\(\w+\))?|33\(\d+\w*\))\b"
        )
        for result in results:
            for match in pattern.findall(result.text or ""):
                normalized = match.replace("33(7B)", "33(7)(B)").replace(
                    "33(20B)", "33(20)(B)"
                )
                schemes.add(normalized)
        return sorted(schemes)

    def _rerank_results(
        self, question: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Re-rank results using cross-encoder with pre-filtering and timeout guard."""
        if not self._reranker or not results:
            return results

        # Pre-filter: only rerank candidates above cosine threshold, cap at 15
        candidates = [r for r in results if r.score > 0.25][:15]
        if not candidates:
            candidates = results[:15]  # keep top-15 if all below threshold

        try:
            pairs = [(question, r.text) for r in candidates]
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(self._reranker.predict, pairs)
                try:
                    scores = future.result(timeout=2.0)
                except TimeoutError:
                    print("[Rerank] Timeout — falling back to cosine scores")
                    return results  # return original order on timeout
            for i, r in enumerate(candidates):
                r.score = float(scores[i])
            candidates.sort(key=lambda x: x.score, reverse=True)
            # Append any results that were filtered out, keeping their original order
            filtered_out = [r for r in results if r not in candidates]
            return candidates + filtered_out
        except Exception as e:
            print(f"Rerank error: {e}")
            return results

    def query(self, question: str, k: int = 10) -> Dict:
        """Main query method with parallel RAG + web search retrieval."""
        question = str(question)
        start_time = time.time()
        self.query_count += 1
        thought_process = []

        print(f"[QUERY] Processing: {question}")

        # Check cache first (Redis or in-memory fallback)
        cache_key = (
            f"query:{hashlib.sha256(question.lower().strip().encode()).hexdigest()}"
        )

        # Try Redis cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            print(f"[CACHE HIT] Returning cached response for: {question}")
            return cached_result

        thought_process.append(f"Received query: '{question}'")

        history_context = self._memory.get_context_for_query(question)
        if history_context:
            thought_process.append("Retrieved conversation history for context.")

        # 1. Analyze Query
        thought_process.append("Analyzing query intent and extracting entities...")
        context = self._analyze_query(question, history_context)
        thought_process.append(
            f"Detected intent: {context.intent}, Topic: {context.topic}"
        )

        # 2. Parallel Retrieval: local RAG retrieval using DCPR documents only
        search_queries = self._expand_queries(context)
        web_enabled = False
        if context.needs_market_data and os.environ.get("ENABLE_WEB_SEARCH", "true").lower() == "true":
            web_enabled = True

        if web_enabled:
            thought_process.append("Executing parallel retrieval (RAG + Web Search)...")
        else:
            thought_process.append(
                "Executing local RAG retrieval using DCPR documents only..."
            )

        # Web search timeout (seconds)
        WEB_SEARCH_TIMEOUT = int(os.environ.get("WEB_SEARCH_TIMEOUT", "5"))

        with ThreadPoolExecutor(max_workers=2) as executor:
            rag_future = executor.submit(self._multi_search, search_queries, context, k)
            web_future = (
                executor.submit(self._search_web, question) if web_enabled else None
            )

            local_results = rag_future.result()
            thought_process.append(
                f"Retrieved {len(local_results)} local candidate chunks."
            )

            web_context = ""
            web_sources = []
            if web_future:
                try:
                    web_context, web_sources = web_future.result(
                        timeout=WEB_SEARCH_TIMEOUT
                    )
                    print(f"\n[Web Search] Found {len(web_sources)} sources:")
                    for i, src in enumerate(web_sources[:3], 1):
                        print(f"  {i}. {src.get('title', 'N/A')}")
                        print(f"     {src.get('snippet', 'N/A')[:100]}...")
                    print()
                    thought_process.append(
                        f"Web search completed. Found {len(web_sources)} sources."
                    )
                except TimeoutError:
                    print(
                        f"\n[Web Search] Timed out after {WEB_SEARCH_TIMEOUT}s - proceeding with RAG only"
                    )
                    thought_process.append(
                        "Web search timed out - using local DCPR knowledge only."
                    )

        # Enhanced web search for feasibility/market queries
        # Do not use web search for DCPR-focused RAG answers by default.
        if web_enabled and context.needs_market_data and context.location:
            enhanced_web = self._search_web(
                f"{context.location} Pune property rates price per sqft 2025 2026 investment commercial IT parks yields"
            )
            if enhanced_web[0]:
                web_context = enhanced_web[0] + "\n\n" + web_context
                web_sources.extend(enhanced_web[1])

            # Additional search for specific yields and companies
            if (
                context.topic in ["Market Analysis", "Investment", "Commercial"]
                or "investment" in question.lower()
            ):
                yield_search = self._search_web(
                    "Pune commercial property rental yields percentage Hinjewadi Kharadi Kalyani Nagar IT companies 2025"
                )
                if yield_search[0]:
                    web_context = yield_search[0] + "\n\n" + web_context
                    web_sources.extend(yield_search[1])

            thought_process.append("Enhanced web search for market data.")

        # 3. Re-rank combined results (local + web)
        if self._reranker and (local_results or web_context):
            thought_process.append("Re-ranking combined results...")
            all_for_rerank = list(local_results)
            if web_context and web_sources:
                # Add web results as SearchResult for reranking
                for ws in web_sources[:3]:
                    all_for_rerank.append(
                        SearchResult(
                            query=question,
                            text=ws.get("snippet", ""),
                            score=0.5,
                            clauses=[],
                            tables=[],
                            relevance_tags=["web"],
                            result_type="web",
                            web_url=ws.get("url", ""),
                        )
                    )
            reranked = self._rerank_results(question, all_for_rerank)
            # Split back: local ones keep their metadata, web ones stay at bottom
            local_results = [r for r in reranked if r.result_type == "local"]
            web_reranked = [r for r in reranked if r.result_type == "web"]

        # 4. Synthesis & Comparison
        thought_process.append("Synthesizing information from sources...")
        synthesis = self._synthesize_all(question, local_results, web_context, context)

        # 5. Generate Dynamic Answer with citations
        thought_process.append("Generating final response...")
        answer = self._generate_dynamic_answer(
            question, local_results, web_context, synthesis, context
        )

        confidence = self._calculate_confidence(local_results, synthesis)
        all_clauses = list(set([c for r in local_results for c in r.clauses]))
        all_tables = list(set([t for r in local_results for t in r.tables]))
        suggestions = self._generate_suggestions(question, context, all_clauses)

        self._memory.add("user", question)
        self._memory.add("assistant", answer)

        total_time = time.time() - start_time
        print(f"[QUERY COMPLETE] Total time: {total_time:.2f}s")

        return {
            "query": question,
            "answer": answer,
            "session_id": self.session_id,
            "thought_process": thought_process,
            "sources": [
                {
                    "text": r.text[:300],
                    "score": r.score,
                    "source": r.source,
                    "page": r.page,
                    "doc_type": r.doc_type,
                    "result_type": r.result_type,
                    "web_url": r.web_url,
                }
                for r in local_results[:5]
            ],
            "web_sources": web_sources[:5] if web_sources else [],
            "clauses_found": all_clauses[:15],
            "tables_found": all_tables[:10],
            "suggestions": suggestions,
            "confidence": confidence,
            "conversation_history": self._memory.get_history(),
            "total_time": total_time,
            "retrieval_stats": {
                "local_results": len(local_results),
                "web_results": len(web_sources) if web_sources else 0,
                "queries_expanded": len(search_queries),
            },
        }

        # Store in Redis cache
        self._set_in_cache(cache_key, results)

        return results

    def _synthesize_all(
        self, question: str, local_results: List, web_context: str, context: Dict
    ) -> Dict:
        # Extract query parameters for conditional lookups
        import re

        area_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqr?\s*ft|sq\.?\s*m)", question.lower()
        )
        road_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter)", question.lower())

        query_params = {
            "area_sqft": float(area_match.group(1)) if area_match else None,
            "road_width_m": float(road_match.group(1)) if road_match else None,
            "location": "mumbai"
            if any(
                x in question.lower()
                for x in ["mumbai", "prabhadevi", "worli", "bandra", "dcpr 2034"]
            )
            else "pune",
        }

        local_text = "\n\n".join(
            [f"[Local {i + 1}] {r.text[:500]}" for i, r in enumerate(local_results[:5])]
        )

        prompt = f"""Compare and synthesize information for the question: "{question}"

Query Parameters Detected:
- Area: {query_params.get("area_sqft")} sq.ft
- Road Width: {query_params.get("road_width_m")} meters
- Location: {query_params.get("location")}

1. LOCAL RAG DOCUMENTS:
{local_text if local_text else "No local information found."}

2. WEB SEARCH:
{web_context if web_context else "No web search results."}

INSTRUCTIONS:
- If area and road_width are detected, look up the relevant FSI tables in the documents
- For Mumbai (DCPR 2034), identify ALL eligible redevelopment schemes (33(7)(B), 33(20)(B), 33(9), 30(A), etc.) based on plot area and road width.
- Identify which regulation/table applies
- Compare local vs web sources

Return JSON:
{{
  "comparison": "Brief comparison of sources",
  "key_parameters": {{
    "area": "detected or estimated from question",
    "road_width": "detected or assumed standard (9m default)",
    "zone_type": "detected or residential default",
    "location": "{query_params.get("location")}"
  }},
  "eligible_schemes": [
    {{
      "scheme": "Regulation number (e.g., 33(7)(B))",
      "status": "Eligible / Potentially Eligible / Not Eligible",
      "reason": "Brief explanation based on area/road"
    }}
  ],
  "applicable_regulations": ["list of regulation numbers that apply"],
  "fsi_data": {{
    "base_fsi": "value from tables",
    "max_fsi": "maximum allowed",
    "premium_fsi": "if applicable"
  }},
  "missing_info": ["what we need from user"]
}}
"""
        try:
            kwargs = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            if self._use_openai:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kwargs)
            res = json.loads(response.choices[0].message.content)

            # If the model did not return explicit regulations, infer them from retrieved document text.
            if not res.get("applicable_regulations"):
                res["applicable_regulations"] = self._extract_applicable_regulations(
                    local_results
                )

            if not res.get("eligible_schemes"):
                inferred_schemes = res.get("applicable_regulations", [])[:3]
                res["eligible_schemes"] = [
                    {
                        "scheme": scheme,
                        "status": "Potentially Eligible",
                        "reason": "Detected in local regulation excerpts"
                    }
                    for scheme in inferred_schemes
                ]

            return res
        except Exception as e:
            print(f"Synthesis error: {e}")
            return {
                "comparison": "Error in synthesis",
                "key_parameters": {},
                "applicable_regulations": [],
                "fsi_data": {},
                "missing_info": [],
            }

    def stream_query(self, question: str, k: int = 10):
        """Streaming version with parallel RAG + web search retrieval."""
        question = str(question)
        self.query_count += 1
        thought_steps = []
        web_enabled = os.environ.get("ENABLE_WEB_SEARCH", "true").lower() == "true"

        history_context = self._memory.get_context_for_query(question)
        context = self._analyze_query(question, history_context)
        thought_steps.append(
            f"Identified intent: {context.intent}, topic: {context.topic}"
        )
        yield json.dumps({"type": "thought_process", "steps": thought_steps}) + "\n"

        # 2. Expand queries
        search_queries = self._expand_queries(context)
        thought_steps.append(f"Expanded to {len(search_queries)} search queries")
        yield json.dumps({"type": "thought_process", "steps": thought_steps}) + "\n"

        # 3. Parallel Retrieval: RAG + Web Search
        thought_steps.append("Executing parallel retrieval (RAG + Web)...")
        yield json.dumps({"type": "thought_process", "steps": thought_steps}) + "\n"

        web_enabled = os.environ.get("ENABLE_WEB_SEARCH", "true").lower() == "true"
        WEB_SEARCH_TIMEOUT = int(os.environ.get("WEB_SEARCH_TIMEOUT", "5"))

        with ThreadPoolExecutor(max_workers=2) as executor:
            rag_future = executor.submit(self._multi_search, search_queries, context, k)
            web_future = (
                executor.submit(self._search_web, question) if web_enabled else None
            )

            local_results = rag_future.result()
            thought_steps.append(f"Found {len(local_results)} local results")

            web_context = ""
            web_sources = []
            if web_future:
                try:
                    web_context, web_sources = web_future.result(
                        timeout=WEB_SEARCH_TIMEOUT
                    )
                    thought_steps.append(f"Web search found {len(web_sources)} sources")
                except TimeoutError:
                    thought_steps.append("Web search timed out - using local knowledge")

        yield json.dumps({"type": "thought_process", "steps": thought_steps}) + "\n"

        # 4. Rerank combined results
        if self._reranker and local_results:
            thought_steps.append("Reranking results...")
            yield json.dumps({"type": "thought_process", "steps": thought_steps}) + "\n"
            local_results = self._rerank_results(question, local_results)

        # 5. Synthesize
        thought_steps.append("Synthesizing answer from sources...")
        yield json.dumps({"type": "thought_process", "steps": thought_steps}) + "\n"

        synthesis = self._synthesize_all(question, local_results, web_context, context)

        # Yield metadata
        metadata = {
            "type": "metadata",
            "session_id": self.session_id,
            "sources": [
                {
                    "text": r.text[:200],
                    "source": r.source,
                    "page": r.page,
                    "doc_type": r.doc_type,
                }
                for r in local_results[:5]
            ],
            "web_sources": web_sources[:5] if web_sources else [],
            "comparison": synthesis.get("comparison", ""),
            "thought_process": thought_steps,
        }
        yield json.dumps(metadata) + "\n"

        # Stream the Answer
        full_answer = ""
        for chunk in self._stream_dynamic_answer(
            question, local_results, web_context, synthesis, context, web_sources
        ):
            full_answer += chunk
            yield json.dumps({"type": "content", "content": chunk}) + "\n"

        # Finalize
        self._memory.add("user", question)
        self._memory.add("assistant", full_answer)

        final_data = {
            "type": "final",
            "confidence": self._calculate_confidence(local_results, synthesis),
            "suggestions": self._generate_suggestions(
                question,
                context,
                list(set([c for r in local_results for c in r.clauses])),
            ),
            "thought_process": thought_steps,
        }
        yield json.dumps(final_data) + "\n"

    def _generate_dynamic_answer(
        self,
        question: str,
        local_results: List[SearchResult],
        web_context: str,
        synthesis: Dict,
        context: QueryContext,
        web_sources: List[Dict] = None,
    ) -> str:
        """Generate answer with proper citations from source metadata."""
        # Build context with source metadata for each result
        local_text = "\n\n".join(
            [
                f"[Doc {i + 1}] Source: {r.source}, Page: {r.page}\n{r.text[:600]}"
                for i, r in enumerate(local_results[:6])
                if r.text
            ]
        )

        # Build citations with actual source info - include web titles
        citations = []
        for i, r in enumerate(local_results[:6]):
            if r.source:
                page_info = f", p.{r.page}" if r.page else ""
                citations.append(f"[Doc {i + 1}] {r.source}{page_info}")
            else:
                source_preview = r.text[:60].replace("\n", " ").strip() + "..."
                citations.append(f"[Doc {i + 1}] {source_preview}")

        # Build web source citations - use actual titles from search
        web_citations = []
        if web_sources:
            for i, s in enumerate(web_sources[:5]):
                title = s.get("title", "Unknown")[:40]
                url = s.get("url", "")
                web_citations.append(f"[Web {i + 1}] {title} - {url}")

        all_citations = citations + web_citations
        citations_text = (
            "\n".join(all_citations) if all_citations else "No sources available"
        )

        # Different prompts for different query types
        is_comparison = any(
            word in question.lower()
            for word in [
                "best",
                "top",
                "compare",
                "areas",
                "hubs",
                "list",
                "recommend",
                "vs",
                "versus",
            ]
        )
        # If it's technical, ALWAYS use the technical consultant persona, even if market data is needed
        is_market_analysis = (
            context.needs_market_data or context.intent == "feasibility_analysis"
        )

        if context.is_technical or context.intent == "technical_lookup":
            # Technical Regulatory Query (Like Google AI Search)
            consultant_location = "Mumbai" if context.location == "mumbai" else "Pune"
            prompt = f"""You are a senior Urban Planning Consultant specializing in {consultant_location} DCPR/UDCPR.
Provide a high-precision answer based strictly on the regulations.

Question: {question}
Location: {context.location or "Mumbai"}
Parameters: {context.units}

REGULATION EXCERPTS:
{local_text}

WEB CONTEXT:
{web_context[:2000] if web_context else "No web results"}

RULES:
1. STRUCTURE LIKE GOOGLE AI OVERVIEWS: Categorize by building height, plot size, or zone type.
2. MUMBAI REDEVELOPMENT: If location is Mumbai, ALWAYS include a 'Scheme Eligibility Table' comparing 33(7)(B), 33(20)(B), 33(9), 30(A), and 33(12)(B) based on the plot parameters.
3. EXTRACT CONDITIONAL RULES: (e.g., "For buildings up to 15m...", "For plots > 1000sqm...", "H/5 Rule").
4. USE TABLES: For any numerical comparison (Margins, FSI, Parking).
5. CITATIONS: Cite [Doc 1], [Doc 2], etc. inline with exact page numbers.
6. NO FILLER: Do not explain what DCPR is. Go straight to the data.
7. DIRECT ANSWER: If the user asks for a specific number, put it in the first sentence.
8. IF MISSING INFO: If plot area, road width, or zone is not specified, explicitly ask for these required parameters.
9. BE SPECIFIC: Quote exact regulation numbers (e.g., "As per Regulation 33(7)(B)...").
10. IF NO LOCAL DATA: If the regulation excerpts are empty or generic, respond: "To provide accurate feasibility analysis, please share: Plot Area (sq ft), Road Width (meters), and Zone Type (Residential/Commercial)."""
        elif is_comparison and is_market_analysis:
            prompt = f"""You are a real estate investment advisor for Pune, Maharashtra. Provide a comprehensive ranked list based on web search data.

Question: {question}

WEB SEARCH RESULTS (PRIORITY - this is live market data):
{web_context[:4000] if web_context else "No web results"}

REGULATORY DATA (from documents):
{local_text if local_text else "No local regulatory data"}

RULES FOR RANKED LIST QUERIES:
1. STRUCTURE YOUR RESPONSE LIKE A RANKED LIST - each area as a separate section with subheaders
2. For each area, include: location name, price per sq ft, key features, investment potential, rental yields
3. Use markdown tables to compare multiple areas side by side
4. Include specific names of: IT parks, companies, developers, consultants mentioned in web results
5. Mention specific infrastructure projects (metro lines, road widening, etc.)
6. Use bullet points for quick scanning
7. Cite sources as [Web 1], [Web 2] inline
8. If data is not available for a specific metric, state "Data not available" - do NOT make up numbers
9. End with a comparison table summarizing all areas

Example format:
## 1. Hinjewadi (Highest ROI)
- **Price:** ₹X/sq ft
- **Key Feature:** IT hub, Rajiv Gandhi Infotech Park
- **Yields:** X%
- **Companies:** [list]
- [Web 1]

## 2. Kharadi
...

## Summary Table
| Area | Price/sqft | Key Feature | Yields |
|------|-----------|-------------|--------|
"""
        elif context.needs_market_data or context.intent == "feasibility_analysis":
            # Feasibility/Investment query - prioritize web data, give market specifics
            prompt = f"""You are a real estate investment advisor for Pune and Mumbai, Maharashtra. Provide a comprehensive feasibility analysis using only local DCPR document excerpts.

Question: {question}
Location: {context.location or "Pune"}

REGULATORY DATA (from local documents):
{local_text if local_text else "No local regulatory data"}

RULES FOR FEASIBILITY QUERIES:
1. Use only local DCPR documents. Do not use web results or external knowledge.
2. Give feasibility analysis with regulatory numbers from the local documents
3. Mention specific infrastructure or scheme guidance from the documents
4. Then give regulatory FSI/permit information from local documents
5. Use markdown tables for comparing values
6. Cite sources as [Doc 1], [Doc 2]
7. If document data is missing, clearly state "Local DCPR information not available" rather than inventing anything
8. End with actionable next steps based on DCPR regulations
"""
        else:
            # Standard regulatory query
            prompt = f"""You are a Pune urban planning expert. Provide a comprehensive answer like Google AI Search.

Question: {question}
Parameters: Area={context.units.get("original") if context.units else "not specified"}, Road Width={context.units.get("road_width", "9m (default)")}, Location={context.location or "Pune"}

DCPR/UDCPR Regulations:
{local_text[:2500]}

Web Search:
{web_context[:3000] if web_context else "None"}

STYLE - BE LIKE GOOGLE AI SEARCH:
1. Start with direct answer in 1-2 sentences (NO heading before it)
2. Add details in flowing paragraphs without section headers
3. Include tables for data
4. Add key points naturally in paragraphs
5. Cite sources inline
6. Be thorough - not brief
7. NEVER use LaTeX math notation - use plain text
8. NO section titles like "Direct Answer", "Conclusion", etc.
"""
        try:
            answer_model = self._model
            response = self.client.chat.completions.create(
                model=answer_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Pune urban planning expert. Answer using only local DCPR document excerpts. Do not use web search or external knowledge. Cite sources as 'As per [Doc N]'. If you don't have data, say so clearly and do not hallucinate.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()

            if citations:
                citation_labels = [f"[Doc {i + 1}]" for i in range(len(citations))]
                if "Sources:" not in answer:
                    answer += "\n\n**Sources:**\n" + "\n".join(citations)
                answer += "\n\nAs per " + ", ".join(citation_labels) + "."

            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def _stream_dynamic_answer(
        self,
        question: str,
        local_results: List[SearchResult],
        web_context: str,
        synthesis: Dict,
        context: QueryContext,
        web_sources: List[Dict] = None,
    ):
        # Build context with source metadata
        local_text = "\n\n".join(
            [f"[Doc {i + 1}] {r.text[:800]}" for i, r in enumerate(local_results[:8])]
        )

        # Different prompts for feasibility vs regulatory queries
        is_market_analysis = (
            context.needs_market_data or context.intent == "feasibility_analysis"
        )

        if context.is_technical or context.intent == "technical_lookup":
            # Technical Regulatory Query (Streaming)
            fsi_data = synthesis.get("fsi_data", {})
            query_params = synthesis.get("key_parameters", {})

            area_str = query_params.get("area", "Not specified")
            road_str = query_params.get("road_width", "9m (standard default)")
            base_fsi = fsi_data.get("base_fsi", "See documents")
            consultant_location = "Mumbai" if context.location == "mumbai" else "Pune"

            prompt = f"""You are a high-precision Urban Planning Expert specializing in {consultant_location} DCPR/UDCPR. Provide a comprehensive answer like Google AI Search.

Question: {question}
Parameters: Area={area_str}, Road={road_str}, Base FSI={base_fsi}

REGULATION EXCERPTS:
{local_text}

STYLE - BE LIKE GOOGLE AI SEARCH:
1. Use only local DCPR document excerpts. Do not use web results or external knowledge.
2. Start with direct answer in 1-2 sentences (NO heading)
3. MUMBAI REDEVELOPMENT: If location is Mumbai, ALWAYS include a 'Scheme Eligibility Table' comparing 33(7)(B), 33(20)(B), 33(9), 30(A), and 33(12)(B).
4. Flowing paragraphs without section headers
5. Use tables for data
6. Include key points naturally in text
7. Cite sources inline using 'As per [Doc N]'
8. Quote exact regulation numbers (e.g., "As per Regulation 33(7)(B)...").
9. NEVER use LaTeX math notation
10. NO titles like "Direct Answer", "Conclusion", etc.
"""
        elif is_market_analysis:
            prompt = f"""You are a real estate investment advisor for Pune and Mumbai, Maharashtra. Provide a comprehensive feasibility analysis using only local DCPR documents.

Question: {question}
Location: {context.location or "Pune"}

REGULATORY DATA:
{local_text if local_text else "No local regulatory data"}

RULES FOR FEASIBILITY QUERIES:
1. Use only local DCPR documents. Do not use web results or external knowledge.
2. Give investment analysis with regulatory numbers from the local documents
3. Mention specific infrastructure or scheme guidance from the documents
4. Then give regulatory FSI/permit information based on documents
5. Use markdown tables for comparing values
6. Cite sources as [Doc 1], [Doc 2]
7. If no document data is available, state "Local DCPR information not available"
"""
        else:
            # Standard regulatory query
            fsi_data = synthesis.get("fsi_data", {})
            applicable_regs = synthesis.get("applicable_regulations", [])
            query_params = synthesis.get("key_parameters", {})

            regs_str = (
                ", ".join(applicable_regs) if applicable_regs else "See documents below"
            )
            area_str = query_params.get("area", "Not specified")
            road_str = query_params.get("road_width", "9m (standard default)")
            loc_str = query_params.get("location", "Pune (default)")
            base_fsi = fsi_data.get("base_fsi", "See documents")
            max_fsi = fsi_data.get("max_fsi", "See documents")
            prem_fsi = fsi_data.get("premium_fsi", "See documents")

            prompt = f"""You are an expert urban planning consultant for Pune, Maharashtra. Based on the DCPR/UDCPR regulation excerpts below, provide a comprehensive answer.

Question: {question}

Detected Parameters:
- Area: {area_str}
- Road Width: {road_str}
- Location: {loc_str}

Applicable Regulations: {regs_str}
FSI Data: Base={base_fsi}, Max={max_fsi}, Premium={prem_fsi}

DCPR/UDCPR Regulation Excerpts:
{local_text}

RULES:
1. Use only local DCPR documents. Do not use web search or external knowledge.
2. Start with DIRECT answer in 1-2 sentences.
3. If documents have the answer, cite [Doc X]
4. NEVER fabricate values not in sources
5. Use markdown tables for data when relevant
6. Be direct, no filler
"""
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful urban planning expert. Give direct, practical answers. Use tables when comparing values.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                stream=True,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            yield f"\n\n[Streaming Error]: {str(e)}"

    def _calculate_confidence(
        self, results: List[SearchResult], synthesis: Dict = None
    ) -> float:
        if not results:
            return 0.0

        # Top result score
        top_score = results[0].score

        # Average top 5 scores
        avg_score = sum(r.score for r in results[:5]) / min(5, len(results))

        # Count results with decent relevance
        relevant_count = len([r for r in results if r.score > 0.3])
        count_factor = min(relevant_count / 5, 1.0) * 0.15

        # Number of unique clauses found
        total_clauses = len(set(c for r in results for c in r.clauses))
        clause_factor = min(total_clauses / 10, 1.0) * 0.15

        # Number of tables found
        total_tables = len(set(t for r in results for t in r.tables))
        table_factor = min(total_tables / 3, 1.0) * 0.1

        # Check if answer has specific values
        has_values = 0
        if synthesis:
            values = synthesis.get("key_technical_data", [])
            if values:
                has_values = 0.1

        # Final weighted confidence
        confidence = (
            top_score * 0.35
            + avg_score * 0.25
            + count_factor
            + clause_factor
            + table_factor
            + has_values
        )

        return round(min(confidence, 1.0), 2)

    def _generate_suggestions(
        self, question: str, context: QueryContext, clauses: List[str]
    ) -> List[str]:
        return [
            f"What are the requirements under {clauses[0]}?"
            if clauses
            else f"Requirements for {context.topic}",
            "What premiums apply?",
            "How does this compare to other schemes?",
            "What documents are needed?",
        ][:4]

    def reset_memory(self):
        self._memory.clear()


class IntelligentRAG:
    """Main RAG class with session support"""

    def __init__(self, session_id: Optional[str] = None):
        # We still have a default session if provided
        self._default_session_id = session_id

    def query(
        self, question: str, k: int = 10, session_id: Optional[str] = None
    ) -> Dict:
        # Use provided session_id or fall back to default
        question = str(question)
        sid = session_id or self._default_session_id
        session, _ = SessionManager.get_or_create_session(sid)
        return session.query(question, k)

    def stream_query(
        self, question: str, k: int = 10, session_id: Optional[str] = None
    ):
        """Streaming version of the query method"""
        question = str(question)
        sid = session_id or self._default_session_id
        session, _ = SessionManager.get_or_create_session(sid)

        # Start the generator
        for chunk in session.stream_query(question, k):
            yield chunk

    def reset_memory(self, session_id: Optional[str] = None):
        sid = session_id or self._default_session_id
        session, _ = SessionManager.get_or_create_session(sid)
        session.reset_memory()

    @classmethod
    def list_sessions(cls) -> List[Dict]:
        return SessionManager.list_sessions()

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        return SessionManager.delete_session(session_id)

    @classmethod
    def get_knowledge_graph_stats(cls) -> Dict:
        kg = SessionManager.get_knowledge_graph()
        return {
            "clauses": len(kg.clauses) if kg else 0,
            "tables": len(kg.tables) if kg else 0,
            "topics": len(kg.topic_to_clauses) if kg else 0,
        }


def format_result(result: Dict) -> str:
    output = ["=" * 70, f"QUERY: {result['query']}", "=" * 70]
    output.append(f"**Confidence:** {result.get('confidence', 0):.0%}")
    output.append(f"**Session:** {result.get('session_id', 'N/A')}")
    output.append("\n" + "=" * 70)
    output.append("ANSWER")
    output.append("=" * 70)
    output.append(result.get("answer", ""))
    output.append("\n" + "=" * 70)
    return "\n".join(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DCPR RAG with Dynamic Knowledge Graph"
    )
    parser.add_argument("question", help="Your question")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--session", type=str, default=None)

    args = parser.parse_args()
    rag = IntelligentRAG(session_id=args.session)
    result = rag.query(args.question, k=args.k)
    print(format_result(result))
    print(f"\nKnowledge Graph: {result.get('knowledge_graph_stats', {})}")
