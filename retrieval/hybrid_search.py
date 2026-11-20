from retrieval.vector_store import VectorStore
from retrieval.bm25_search import BM25Search

def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Combines vector and BM25 ranked results using Reciprocal Rank Fusion.
    """
    scores = {}
    chunk_lookup = {}

    for rank, chunk in enumerate(vector_results, start=1):
        cid = chunk["chunk_id"]

        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

        if cid not in chunk_lookup:
            chunk_lookup[cid] = chunk

        for rank, chunk in enumerate(bm25_results, start=1):
         cid = chunk["chunk_id"]

         scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

         if cid not in chunk_lookup:
            chunk_lookup[cid] = chunk

        final_list = []

    for cid in scores:
        chunk = {
            **chunk_lookup[cid],
            "rrf_score": scores[cid]
        }
        final_list.append(chunk)

    final_list = sorted(
        final_list,
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return final_list
    
class HybridSearch:
        def __init__(self):
            self.vector_store = VectorStore()
            self.bm25_search = BM25Search()

        def build_index(self, chunks: list[dict]):
            self.vector_store.build_index(chunks)
            self.bm25_search.build_index(chunks)

        def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10
    ) -> list[dict]:

           vector_results = self.vector_store.search(
            query,
            top_k=candidate_k
        )

           bm25_results = self.bm25_search.search(
            query,
            top_k=candidate_k
        )

           fused = reciprocal_rank_fusion(
            vector_results,
            bm25_results
        )

           return fused[:top_k]
        
if __name__ == "__main__":
    from ingestion.document_loader import load_documents
    from ingestion.chunker import chunk_documents

    docs = load_documents("data/raw_docs")
    chunks = chunk_documents(docs, chunk_size=100, overlap=20)

    hybrid = HybridSearch()
    hybrid.build_index(chunks)

    query = "What is the leave policy?"
    results = hybrid.search(query, top_k=3)

    print(f"Query: {query}\n")

    for r in results:
        print(
            f"RRF Score: {r['rrf_score']:.4f} | "
            f"Source: {r['source']} (page {r['page']})"
        )
        print(f"Text: {r['text'][:100]}...")
        print("---")