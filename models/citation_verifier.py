import re
import ollama


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_verification_prompt(
    claim: str,
    source_text: str
) -> str:

    prompt = f"""
You are a strict fact-checker for a Retrieval-Augmented Generation system.

Your task is to determine whether the SOURCE text actually supports the CLAIM.

Use ONLY the SOURCE text.
Do NOT use outside knowledge.
Do NOT make assumptions.

SOURCE:
{source_text}

CLAIM:
{claim}

Rules:

- SUPPORTED = The source directly supports the claim.
- NOT_SUPPORTED = The source does not support the claim,
  contradicts it, or the claim contains information
  that cannot be found in the source.
- If even an important part of the claim is unsupported,
  choose NOT_SUPPORTED.
- Give a confidence score from 0 to 100.

Respond in EXACTLY this format:

VERDICT: SUPPORTED
CONFIDENCE: 95

OR

VERDICT: NOT_SUPPORTED
CONFIDENCE: 20
"""

    return prompt


# ============================================================
# SINGLE CLAIM VERIFICATION
# ============================================================

def verify_claim(
    claim: str,
    cited_chunks: list[dict],
    model: str = "llama3.2"
) -> dict:

    # No citations
    if not cited_chunks:
        return {
            "verdict": "NOT_SUPPORTED",
            "confidence": 0
        }

    # Combine cited source text
    source_text = "\n\n".join(
        c["text"]
        for c in cited_chunks
    )

    # Build prompt
    prompt = build_verification_prompt(
        claim,
        source_text
    )

    try:

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response_text = response["message"]["content"]

    except Exception as e:

        print(f"Verification error: {e}")

        return {
            "verdict": "NOT_SUPPORTED",
            "confidence": 0
        }

    # ========================================================
    # EXTRACT VERDICT
    # ========================================================

    verdict_match = re.search(
        r"VERDICT:\s*(SUPPORTED|NOT_SUPPORTED)",
        response_text,
        re.IGNORECASE
    )

    # ========================================================
    # EXTRACT CONFIDENCE
    # ========================================================

    confidence_match = re.search(
        r"CONFIDENCE:\s*(\d+)",
        response_text,
        re.IGNORECASE
    )

    # Defaults
    verdict = "NOT_SUPPORTED"
    confidence = 0

    if verdict_match:
        verdict = verdict_match.group(1).upper()

    if confidence_match:
        confidence = int(
            confidence_match.group(1)
        )

        confidence = max(
            0,
            min(100, confidence)
        )

    return {
        "verdict": verdict,
        "confidence": confidence
    }


# ============================================================
# VERIFY COMPLETE ANSWER
# ============================================================

def verify_answer(
    parsed_claims: list[dict],
    model: str = "llama3.2"
) -> list[dict]:

    verified_claims = []

    for claim_data in parsed_claims:

        result = verify_claim(
            claim=claim_data["claim"],
            cited_chunks=claim_data["cited_chunks"],
            model=model
        )

        verified_claim = {
            **claim_data,
            "verdict": result["verdict"],
            "confidence": result["confidence"]
        }

        verified_claims.append(
            verified_claim
        )

    return verified_claims


# ============================================================
# CLASS WRAPPER
# ============================================================
#
# Your Streamlit app imports:
#
# from models.citation_verifier import CitationVerifier
#
# Therefore we provide this class.
# ============================================================

class CitationVerifier:

    def __init__(self, model: str = "llama3.2"):

        self.model = model

    def verify_claim(
        self,
        claim: str,
        cited_chunks: list[dict]
    ) -> dict:

        return verify_claim(
            claim=claim,
            cited_chunks=cited_chunks,
            model=self.model
        )

    def verify_answer(
        self,
        parsed_claims: list[dict]
    ) -> list[dict]:

        return verify_answer(
            parsed_claims=parsed_claims,
            model=self.model
        )


# ============================================================
# TEST PIPELINE
# ============================================================

if __name__ == "__main__":

    from ingestion.document_loader import load_documents
    from ingestion.chunker import chunk_documents

    from retrieval.hybrid_search import HybridSearch
    from retrieval.reranker import Reranker

    from models.llm_client import generate_answer
    from models.citation_parser import parse_citations

    # --------------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------------

    print("Loading documents...")

    docs = load_documents(
        "data/raw_docs"
    )

    # --------------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------------

    chunks = chunk_documents(
        docs,
        chunk_size=100,
        overlap=20
    )

    print(f"Loaded {len(chunks)} chunks.")

    # --------------------------------------------------------
    # 3. Build hybrid search
    # --------------------------------------------------------

    hybrid = HybridSearch()

    hybrid.build_index(
        chunks
    )

    # --------------------------------------------------------
    # 4. Query
    # --------------------------------------------------------

    query = "What is the leave policy?"

    print(
        f"\nQuery: {query}"
    )

    # --------------------------------------------------------
    # 5. Retrieve candidates
    # --------------------------------------------------------

    candidates = hybrid.search(
        query,
        top_k=5,
        candidate_k=10
    )

    # --------------------------------------------------------
    # 6. Rerank
    # --------------------------------------------------------

    reranker = Reranker()

    top_chunks = reranker.rerank(
        query,
        candidates,
        top_k=3
    )

    # --------------------------------------------------------
    # 7. Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        query,
        top_chunks
    )

    print(
        "\n============================================================"
    )
    print("RAW ANSWER")
    print(
        "============================================================"
    )

    print(answer)

    # --------------------------------------------------------
    # 8. Parse citations
    # --------------------------------------------------------

    parsed = parse_citations(
        answer,
        top_chunks
    )

    # --------------------------------------------------------
    # 9. Verify citations
    # --------------------------------------------------------

    verifier = CitationVerifier()

    verified = verifier.verify_answer(
        parsed
    )

    # --------------------------------------------------------
    # 10. Display
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )
    print("VERIFIED CLAIMS")
    print(
        "============================================================"
    )

    for item in verified:

        print(
            f"\nClaim: {item['claim']}"
        )

        print(
            f"Verdict: {item['verdict']}"
        )

        print(
            f"Confidence: {item['confidence']}%"
        )

        for chunk in item["cited_chunks"]:

            print(
                f"  -> {chunk['source']} "
                f"(Page {chunk['page']})"
            )

        print("---")