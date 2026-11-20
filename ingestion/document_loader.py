from pathlib import Path
from pypdf import PdfReader
def load_documents(folder_path: str) -> list[dict]:
        """
    Loads all PDFs from folder_path and extracts text page-by-page.
    Returns a list of dicts: {"text": ..., "source": ..., "page": ...}
    """
        documents = []

        pdf_files = Path(folder_path).glob("*.pdf")
        for pdf_file in pdf_files:
          reader = PdfReader(str(pdf_file))
          for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            documents.append({
                "text": text,
                "source": pdf_file.name,
                "page": page_number
            })
        return documents

if __name__ == "__main__":
    docs = load_documents("data/raw_docs")
    print(f"Loaded {len(docs)} page-chunks")

    if docs:
        print("Sample entry:")
        print(docs[0])