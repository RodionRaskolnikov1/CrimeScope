import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from crimescope.config import settings
from crimescope.nlp.embeddings import get_chroma_client, get_collection
from crimescope.utils.logger import logger


# ── LLM Setup ─────────────────────────────────────────────────────

def get_llm() -> ChatGoogleGenerativeAI:
    """Initialize Gemini via LangChain."""
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=settings.google_gemini_api_key,
        temperature=0.3,                # low = more factual
        max_output_tokens=1024,
    )


# ── Retrieval ─────────────────────────────────────────────────────

def retrieve_context(query: str, n_results: int = 3) -> str:
    """
    Search ChromaDB for most relevant zone documents.
    Converts query to embedding, finds closest matches,
    returns their text as context for the LLM.
    """

    client = get_chroma_client()
    collection = get_collection(client)

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant zone data found."

    # Combine retrieved documents into one context block
    context_parts = []
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        context_parts.append(
            f"[Zone {meta['zone_id']} — Risk: {meta['risk_score']}/100]\n{doc}"
        )

    return "\n\n---\n\n".join(context_parts)


# ── RAG Chain ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CrimeScope AI, an urban safety intelligence assistant
for Chicago. You analyze crime patterns, forecasts, and risk assessments to help
users understand safety conditions across different city zones.

Always base your answers strictly on the provided context data.
Be specific with zone IDs, crime types, percentages, and times.
If the data doesn't contain enough information to answer, say so clearly.
Keep responses concise, factual, and actionable."""


def ask(query: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant zone documents from ChromaDB
    2. Build prompt with context
    3. Send to Gemini
    4. Return answer + sources

    Example:
        ask("Which zones are most dangerous on Friday nights?")
    """

    logger.info(f"Query: {query}")

    # Step 1 — Retrieve context
    context = retrieve_context(query, n_results=3)
    logger.debug(f"Retrieved context ({len(context)} chars)")

    # Step 2 — Build messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Context from CrimeScope database:
{context}

User question: {query}

Please answer based on the context above.
        """.strip())
    ]

    # Step 3 — Call Gemini
    llm = get_llm()
    response = llm.invoke(messages)
    
    content = response.content
    if isinstance(content, list):
        answer = content[0]["text"] if content else ""
    else:
        answer = content

    logger.success(f"Answer generated ({len(answer)} chars)")

    return {
        "query": query,
        "answer": answer,
        "context_used": context[:500] + "..." if len(context) > 500 else context,
    }


def interactive_chat() -> None:
    """
    Simple CLI chat loop for testing the RAG system.
    Type 'quit' to exit.
    """

    logger.info("CrimeScope Chat started. Type 'quit' to exit.")
    print("\n🔍 CrimeScope AI — Ask me anything about Chicago crime patterns\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if not query:
            continue

        result = ask(query)
        print(f"\n🤖 CrimeScope: {result['answer']}\n")