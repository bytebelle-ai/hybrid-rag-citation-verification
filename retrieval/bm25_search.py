from rank_bm25 import BM25Okapi
import numpy as np

class BM25Search:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()
    
    def build_index(self, chunks: list[dict]):
        """
        chunks: list of {"chunk_id", "text", "source", "page"}
        """
        self.chunks = chunks

        tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for i in top_indices:
            results.append({
                **self.chunks[i],
                "score": float(scores[i])
            })

        return results
    
if __name__ == "__main__":
    from ingestion.document_loader import load_documents
    from ingestion.chunker import chunk_documents

    docs = load_documents("data/raw_docs")
    chunks = chunk_documents(docs, chunk_size=100, overlap=20)

    bm25_search = BM25Search()
    bm25_search.build_index(chunks)

    query = "What is the leave policy?"
    results = bm25_search.search(query, top_k=2)

    print(f"Query: {query}\n")

    for r in results:
        print(f"Score: {r['score']:.4f} | Source: {r['source']} (page {r['page']})")
        print(f"Text: {r['text'][:100]}...")
        print("---")