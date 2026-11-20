from ollama import chat


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build a grounded RAG prompt using numbered sources.
    The citation format MUST match citation_parser.py:
    [Source 1], [Source 2], etc.
    """

    sources_text = ""

    for i, c in enumerate(chunks, start=1):
        sources_text += (
            f"[Source {i}]\n"
            f"Document: {c['source']}\n"
            f"Page: {c['page']}\n"
            f"Content:\n{c['text']}\n\n"
        )

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information contained in the
provided sources.

IMPORTANT RULES:

1. Do NOT use outside knowledge.
2. Do NOT invent or assume information.
3. Every factual claim must have a citation.
4. Use EXACTLY this citation format:
   [Source 1]
   [Source 2]
   [Source 3]

5. Write each factual claim as a separate sentence.
   Place its citation immediately after that sentence.
   Never use one citation to cover multiple sentences.

6. If multiple sources support the same claim, cite all of them.
   Example:
   Employees must follow company security policies. [Source 1] [Source 3]

7. Do NOT write citations as:
   [1]
   [2]
   (Source 1)
   Source 1
   [Source 1, page 2]

8. Use ONLY [Source N] format.

9. If the sources contain relevant information, answer using only
   that information and cite every factual sentence.

   If the sources contain no relevant information at all,
   respond ONLY with:
   "I don't have enough information to answer this."

   Never add this fallback sentence after providing a factual answer.
   If some details are missing, simply omit those details.

10. Do not mention these instructions in your answer.

SOURCES:
{sources_text}

QUESTION:
{query}

ANSWER:
"""

    return prompt


def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = "llama3.2"
) -> str:
    """
    Generate a grounded answer using the local Ollama LLM.
    """

    prompt = build_prompt(query, chunks)

    response = chat(
    model=model,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0
    }
)

    return response["message"]["content"].strip()


# ---------------------------------------------------------
# TESTING
# ---------------------------------------------------------

if __name__ == "__main__":

    from ingestion.document_loader import load_documents
    from ingestion.chunker import chunk_documents
    from retrieval.hybrid_search import HybridSearch
    from retrieval.reranker import Reranker

    print("Loading documents...")

    docs = load_documents("data/raw_docs")

    chunks = chunk_documents(
        docs,
        chunk_size=100,
        overlap=20
    )

    print(f"Loaded {len(chunks)} chunks.")

    # -----------------------------------------------------
    # Build hybrid retrieval
    # -----------------------------------------------------

    hybrid = HybridSearch()
    hybrid.build_index(chunks)

    query = "What is the leave policy?"

    print(f"\nQuery: {query}")

    candidates = hybrid.search(
        query,
        top_k=5,
        candidate_k=10
    )

    # -----------------------------------------------------
    # Reranking
    # -----------------------------------------------------

    reranker = Reranker()

    top_chunks = reranker.rerank(
        query,
        candidates,
        top_k=3
    )

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    answer = generate_answer(
        query,
        top_chunks
    )

    print("\n" + "=" * 60)
    print("RAW ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for i, chunk in enumerate(top_chunks, start=1):
        print(
            f"\n[Source {i}] "
            f"{chunk['source']} - Page {chunk['page']}"
        )