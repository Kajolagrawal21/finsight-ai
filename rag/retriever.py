"""
FinSight AI — RAG Retriever
Answers questions about stocks using retrieved news context + local LLM (Ollama).

RAG = Retrieval Augmented Generation
Think of it like: instead of asking the AI to answer from memory,
we first FIND relevant news articles, then ask the AI to answer
USING those articles as context. Much more accurate!
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from rag.embedder import embed_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.retriever")

QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
COLLECTION_NAME = "finsight_news"
OLLAMA_MODEL    = "llama3"
TOP_K           = 5   # retrieve top 5 most relevant articles


# ─── Retriever ────────────────────────────────────────────────────────────────

def retrieve_relevant_news(
    query: str,
    symbol: str = None,
    top_k: int = TOP_K
) -> list:
    client = get_qdrant_client()
    query_vector = embed_text(query).tolist()

    search_filter = None
    if symbol:
        search_filter = Filter(
            must=[FieldCondition(
                key="symbol",
                match=MatchValue(value=symbol)
            )]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    ).points

    articles = []
    for hit in results:
        articles.append({
            "title":        hit.payload.get("title", ""),
            "summary":      hit.payload.get("summary", ""),
            "symbol":       hit.payload.get("symbol", ""),
            "source":       hit.payload.get("source", ""),
            "published_at": hit.payload.get("published_at", ""),
            "score":        round(hit.score, 4),
        })

    logger.info(f"🔍 Retrieved {len(articles)} articles for query: '{query}'")
    return articles

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ─── RAG Chain ────────────────────────────────────────────────────────────────

FINANCIAL_ANALYST_PROMPT = PromptTemplate(
    input_variables=["context", "question", "symbol"],
    template="""You are FinSight AI, an expert financial analyst assistant.
You answer questions about stocks using the provided news context.
Be concise, factual, and cite the news sources when relevant.
If the context doesn't contain enough information, say so clearly.

Stock Symbol: {symbol}

Relevant News Context:
{context}

Question: {question}

Answer (be specific and reference the news where relevant):"""
)


def format_context(articles: list) -> str:
    """Format retrieved articles into a readable context string."""
    if not articles:
        return "No recent news articles found for this query."

    context_parts = []
    for i, article in enumerate(articles, 1):
        context_parts.append(
            f"[Article {i}] {article['source']} ({article['published_at'][:10]})\n"
            f"Title: {article['title']}\n"
            f"Summary: {article['summary']}\n"
            f"Relevance Score: {article['score']}"
        )
    return "\n\n".join(context_parts)


def answer_question(
    question: str,
    symbol: str = None,
    top_k: int = TOP_K
) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant news articles
    2. Format as context
    3. Ask local LLM to answer using that context
    4. Return answer + sources
    """
    logger.info(f"❓ Question: {question}")

    # Step 1: Retrieve relevant articles
    articles = retrieve_relevant_news(question, symbol=symbol, top_k=top_k)

    # Step 2: Format context
    context = format_context(articles)

    # Step 3: Ask LLM
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)
        prompt = FINANCIAL_ANALYST_PROMPT.format(
            context=context,
            question=question,
            symbol=symbol or "General Market"
        )
        answer = llm.invoke(prompt)

    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        answer = (
            f"LLM unavailable (is Ollama running?). "
            f"Here are the relevant articles I found:\n\n{context}"
        )

    return {
        "question": question,
        "symbol":   symbol,
        "answer":   answer,
        "sources":  articles,
        "num_sources": len(articles),
    }


# ─── Run Pipeline ─────────────────────────────────────────────────────────────

def run_rag_pipeline():
    """Run the complete RAG pipeline — fetch, index, then demo Q&A."""
    from rag.news_fetcher import run_news_fetcher
    from rag.indexer import index_articles, get_collection_stats

    logger.info("🚀 Starting RAG Pipeline")

    # Step 1: Fetch news
    logger.info("\n📰 Step 1: Fetching news articles...")
    run_news_fetcher()

    # Step 2: Index into Qdrant
    logger.info("\n🔍 Step 2: Indexing into Qdrant...")
    indexed = index_articles()

    # Step 3: Show stats
    stats = get_collection_stats()
    logger.info(f"\n📊 Qdrant stats: {stats}")

    # Step 4: Demo Q&A
    logger.info("\n🤖 Step 3: Testing RAG Q&A...")

    test_questions = [
        ("Why is NVDA stock moving?", "NVDA"),
        ("What are analysts saying about Apple?", "AAPL"),
        ("What is the outlook for the S&P 500?", "SPY"),
    ]

    for question, symbol in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ Q: {question}")
        result = answer_question(question, symbol=symbol)
        print(f"🤖 A: {result['answer'][:500]}...")
        print(f"📰 Sources: {result['num_sources']} articles used")

    logger.info("\n✅ RAG Pipeline complete!")


if __name__ == "__main__":
    run_rag_pipeline()
