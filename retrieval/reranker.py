from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3
    ) -> list[dict]:

        pairs = [[query, c["text"]] for c in candidates]

        scores = self.model.predict(pairs)

        scored = [
            {**c, "rerank_score": float(score)}
            for c, score in zip(candidates, scores)
        ]

        scored = sorted(
            scored,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return scored[:top_k]
    
if __name__ == "__main__":
    from ingestion.document_loader import load_documents
    from ingestion.chunker import chunk_documents
    from retrieval.hybrid_search import HybridSearch

    docs = load_documents("data/raw_docs")
    chunks = chunk_documents(
        docs,
        chunk_size=100,
        overlap=20
    )

    hybrid = HybridSearch()
    hybrid.build_index(chunks)

    query = "What is the leave policy?"

    candidates = hybrid.search(
        query,
        top_k=5,
        candidate_k=10
    )

    reranker = Reranker()

    reranked = reranker.rerank(
        query,
        candidates,
        top_k=3
    )

    print(f"Query: {query}\n")

    for r in reranked:
        print(
            f"Rerank Score: {r['rerank_score']:.4f} | "
            f"Source: {r['source']} (page {r['page']})"
        )

        print(f"Text: {r['text'][:100]}...")
        print("---")