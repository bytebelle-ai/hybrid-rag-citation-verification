from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
         self.model = SentenceTransformer(model_name)
         self.index = None
         self.chunks = []

        def build_index(self, chunks: list[dict]):
         """
         chunks: list of {"chunk_id", "text", "source", "page"}
         """
         self.chunks = chunks
         texts = [c["text"] for c in chunks]
         embeddings = self.model.encode(texts)
         embeddings = embeddings.astype("float32")
         dimension = embeddings.shape[1]
         self.index = faiss.IndexFlatL2(dimension)
         self.index.add(embeddings)

        def search(self, query: str, top_k: int = 3) -> list[dict]:
         """
         Returns top_k chunks most similar to query, each with a 'score' key added.
         """
         top_k = min(top_k, len(self.chunks))
         
         query_embedding = self.model.encode([query])
         query_embedding = np.array(query_embedding).astype("float32")

         distances, indices = self.index.search(query_embedding, top_k)
         results = []

         for distance, index in zip(distances[0],indices[0]):
          chunk = self.chunks[index].copy()
          chunk["score"] = float(distance)
          results.append(chunk)
        
         return results

if __name__ == "__main__":
    from ingestion.document_loader import load_documents
    from ingestion.chunker import chunk_documents

    docs = load_documents("data/raw_docs")
    chunks = chunk_documents(docs, chunk_size=100, overlap=20)

    store = VectorStore()
    store.build_index(chunks)

    query = "What is the leave policy?"
    results = store.search(query, top_k=2)

    print(f"Query: {query}\n")

    for r in results:
        print(f"Score: {r['score']:.4f} | Source: {r['source']} (page {r['page']})")
        print(f"Text: {r['text'][:100]}...")
        print("---")