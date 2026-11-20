import numbers
import re



def parse_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Split the answer into sentences and map citations
    to individual claims.
    """
    results = []

    citation_pattern = r"\[Source\s+(\d+)\]"

    # Keep citations attached to their preceding sentences
    sentences = re.split(
    r"(?<=\])\s+|(?<=[.!?])\s+(?!\[Source\s+\d+\])",
    answer.strip(),
    flags=re.IGNORECASE
)

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        matches = re.findall(
            citation_pattern,
            sentence,
            re.IGNORECASE
        )

        clean = re.sub(
            citation_pattern,
            "",
            sentence,
            flags=re.IGNORECASE
        ).strip()

        if not clean:
            continue

        # Skip the exact fallback message
        if clean.lower().rstrip(".") == "i don't have enough information to answer this":
            continue

        cited_chunks = []

        for source_num in matches:
            idx = int(source_num) - 1

            if 0 <= idx < len(chunks):
                if chunks[idx] not in cited_chunks:
                    cited_chunks.append(chunks[idx])

        results.append({
            "claim": clean,
            "cited_chunks": cited_chunks,
            "has_citation": bool(cited_chunks)
        })

    return results



if __name__ == "__main__":

    from ingestion.document_loader import load_documents
    from ingestion.chunker import chunk_documents
    from retrieval.hybrid_search import HybridSearch
    from retrieval.reranker import Reranker
    from models.llm_client import generate_answer

    # Load documents
    docs = load_documents(
        "data/raw_docs"
    )

    # Create chunks
    chunks = chunk_documents(
        docs,
        chunk_size=100,
        overlap=20
    )

    # Hybrid retrieval
    hybrid = HybridSearch()

    hybrid.build_index(
        chunks
    )

    query = "What is the leave policy?"

    candidates = hybrid.search(
        query,
        top_k=5,
        candidate_k=10
    )

    # Reranking
    reranker = Reranker()

    top_chunks = reranker.rerank(
        query,
        candidates,
        top_k=3
    )

    # Generate answer
    answer = generate_answer(
        query,
        top_chunks
    )

    print(
        f"Raw Answer:\n{answer}\n"
    )

    # Parse citations
    parsed = parse_citations(
        answer,
        top_chunks
    )

    print("Parsed Claims:")

    for p in parsed:

        print(
            f"Claim: {p['claim']}"
        )

        for c in p["cited_chunks"]:

            print(
                f"  -> Source: {c['source']} "
                f"(page {c['page']})"
            )

        print("---")