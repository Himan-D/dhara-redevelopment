import os
import logging
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

logger = logging.getLogger(__name__)

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_URI = os.environ.get("MILVUS_URI", "")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "documents")
RAG_COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION_RAG", COLLECTION_NAME)
EMBEDDING_DIM = 1536


def _connect_milvus():
    """Connect to Milvus — supports both local (host:port) and Zilliz Cloud (uri+token)."""
    if MILVUS_URI and MILVUS_TOKEN:
        # Zilliz Cloud / remote Milvus with token auth
        uri = MILVUS_URI if MILVUS_URI.startswith("https://") else f"https://{MILVUS_URI}"
        logger.info("Connecting to Zilliz Cloud at %s", uri)
        connections.connect(alias="default", uri=uri, token=MILVUS_TOKEN, timeout=15)
    else:
        # Local Milvus (host:port)
        logger.info("Connecting to Milvus at %s:%s", MILVUS_HOST, MILVUS_PORT)
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=10)


def setup_local_milvus(collection_name=COLLECTION_NAME, dim=EMBEDDING_DIM):
    print(f"Connecting to Milvus...")
    try:
        _connect_milvus()
        print("Connected to Milvus")
    except Exception as e:
        print(f"Failed to connect: {e}")
        raise

    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} already exists")
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Document embeddings with source")
    collection = Collection(collection_name, schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    print(f"Created collection: {collection_name}")
    return collection

def get_collection(collection_name=COLLECTION_NAME):
    try:
        _connect_milvus()
        if utility.has_collection(collection_name):
            return Collection(collection_name)
        return None
    except Exception:
        return None
