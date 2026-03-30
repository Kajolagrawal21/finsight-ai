"""
FinSight AI — Text Embedder
Converts news articles to vector embeddings using sentence-transformers.
These vectors capture the MEANING of text, enabling semantic search.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.embedder")

# This model runs locally — no API calls, no cost
# 384-dimensional embeddings, fast on Apple Silicon
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Singleton — load model once, reuse
_model = None


def get_model() -> SentenceTransformer:
    """Load embedding model (cached after first load)."""
    global _model
    if _model is None:
        logger.info(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("✅ Embedding model loaded!")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Convert a single text string to a vector embedding.
    
    Think of it like: each article gets converted to a list of 384 numbers
    that captures its meaning. Similar articles will have similar numbers.
    """
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding


def embed_batch(texts: list) -> np.ndarray:
    """
    Convert a list of texts to embeddings efficiently (batch processing).
    Much faster than embedding one by one.
    """
    if not texts:
        return np.array([])

    model = get_model()
    logger.info(f"🔢 Embedding batch of {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=len(texts) > 10
    )
    logger.info(f"✅ Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
    return embeddings


def prepare_article_text(title: str, summary: str = "", symbol: str = "") -> str:
    """
    Combine article fields into one string for embedding.
    The richer the text, the better the embedding captures meaning.
    """
    parts = []
    if symbol:
        parts.append(f"Stock: {symbol}.")
    if title:
        parts.append(title)
    if summary and summary != title:
        parts.append(summary)
    return " ".join(parts)


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "Apple reported record iPhone sales beating analyst expectations",
        "NVIDIA announces new AI chip with 2x performance improvement",
        "Federal Reserve keeps interest rates unchanged amid inflation concerns",
    ]
    embeddings = embed_batch(test_texts)
    print(f"✅ Test embeddings shape: {embeddings.shape}")
    print(f"   Each text → {embeddings.shape[1]}-dimensional vector")
