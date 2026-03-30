"""
FinSight AI — Qdrant Indexer
Stores news article embeddings in Qdrant vector database.
Enables fast semantic search — find articles by meaning, not keywords.
"""
import psycopg2
import psycopg2.extras
import logging
import uuid
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from rag.embedder import embed_batch, prepare_article_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.indexer")

QDRANT_HOST       = "localhost"
QDRANT_PORT       = 6333
COLLECTION_NAME   = "finsight_news"
EMBEDDING_DIM     = 384   # all-MiniLM-L6-v2 output size

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "finsight",
    "user":     "finsight_user",
    "password": "finsight_pass",
}


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection_exists(client: QdrantClient):
    """
    Create the Qdrant collection if it doesn't exist.
    A collection is like a table in a regular DB — stores vectors + metadata.
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE   # cosine similarity = best for text
            )
        )
        logger.info(f"✅ Created Qdrant collection: {COLLECTION_NAME}")
    else:
        logger.info(f"📦 Collection '{COLLECTION_NAME}' already exists")


def get_unindexed_articles(conn, limit: int = 100) -> list:
    """Fetch articles from TimescaleDB that haven't been indexed yet."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, symbol, title, summary, published_at, source
            FROM news_articles
            WHERE embedding_id IS NULL
            ORDER BY published_at DESC
            LIMIT %s
        """, (limit,))
        return cur.fetchall()


def mark_as_indexed(conn, article_id: int, embedding_id: str):
    """Update the article record with its Qdrant vector ID."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE news_articles
            SET embedding_id = %s
            WHERE id = %s
        """, (embedding_id, article_id))
    conn.commit()


def index_articles(batch_size: int = 50):
    """
    Main indexing pipeline:
    1. Load unindexed articles from TimescaleDB
    2. Generate embeddings
    3. Store in Qdrant with metadata
    4. Mark as indexed in TimescaleDB
    """
    logger.info("🔍 Starting article indexing pipeline")

    client = get_qdrant_client()
    ensure_collection_exists(client)

    conn = psycopg2.connect(**DB_CONFIG)
    articles = get_unindexed_articles(conn, limit=batch_size)

    if not articles:
        logger.info("✅ No new articles to index")
        conn.close()
        return 0

    logger.info(f"📄 Found {len(articles)} articles to index")

    # Prepare texts for embedding
    texts = [
        prepare_article_text(
            title=a["title"],
            summary=a.get("summary", ""),
            symbol=a.get("symbol", "")
        )
        for a in articles
    ]

    # Generate embeddings in batch
    embeddings = embed_batch(texts)

    # Build Qdrant points (vector + metadata payload)
    points = []
    for i, article in enumerate(articles):
        point_id = str(uuid.uuid4())

        points.append(PointStruct(
            id=point_id,
            vector=embeddings[i].tolist(),
            payload={
                "article_id":   article["id"],
                "symbol":       article.get("symbol", ""),
                "title":        article["title"],
                "summary":      article.get("summary", ""),
                "published_at": str(article["published_at"]),
                "source":       article.get("source", ""),
            }
        ))

        # Mark as indexed in DB
        mark_as_indexed(conn, article["id"], point_id)

    # Upload to Qdrant in one batch
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    conn.close()
    logger.info(f"✅ Indexed {len(points)} articles into Qdrant!")
    return len(points)


def get_collection_stats() -> dict:
    """Return stats about the Qdrant collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "total_vectors": info.vectors_count,
            "collection":    COLLECTION_NAME,
            "status":        info.status,
        }
    except Exception:
        return {"total_vectors": 0, "collection": COLLECTION_NAME}


if __name__ == "__main__":
    count = index_articles()
    stats = get_collection_stats()
    print(f"\n📊 Qdrant collection stats: {stats}")
