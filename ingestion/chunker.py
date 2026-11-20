from ingestion.document_loader import load_documents
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Splits a single text string into overlapping word-based chunks.
    """
    words = text.split()
    chunks = []
    for start in range(0, len(words), chunk_size - overlap):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            continue

        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    
    return chunks

def chunk_documents(documents: list[dict], chunk_size: int = 200, overlap: int = 50) -> list[dict]:
    """
    Takes page-level documents and produces chunk-level documents
    with unique chunk_id, preserving source and page metadata.
    """
    chunked_docs = []
    chunk_counter = 0

    for doc in documents:
        chunks = chunk_text(
            doc["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )
        for chunk in chunks:
            chunked_docs.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"]
            })

            chunk_counter += 1
    
    return chunked_docs

if __name__ == "__main__":
    docs = load_documents("data/raw_docs")
    chunks = chunk_documents(docs, chunk_size=100, overlap=20)

    print(f"Created {len(chunks)} chunks from {len(docs)} pages")

    for c in chunks[:2]:
        print(c)
        print("---")