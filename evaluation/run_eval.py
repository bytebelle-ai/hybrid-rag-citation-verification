from evaluation.eval_dataset import (
    EVAL_DATASET,
    VERIFICATION_TEST_CASES
)

from ingestion.document_loader import load_documents
from ingestion.chunker import chunk_documents

from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker

from models.citation_verifier import verify_claim


def evaluate_retrieval():
    """
    Evaluates whether the expected document and relevant
    keywords are retrieved.
    """

    docs = load_documents("data/raw_docs")

    chunks = chunk_documents(
        docs,
        chunk_size=100,
        overlap=20
    )

    hybrid = HybridSearch()
    hybrid.build_index(chunks)

    reranker = Reranker()

    correct_source = 0
    keyword_hits = 0

    total = len(EVAL_DATASET)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    for i, test_case in enumerate(EVAL_DATASET, start=1):

        query = test_case["query"]
        expected_source = test_case["expected_source"]
        keywords = test_case["relevant_keywords"]

        print(f"\nTest {i}: {query}")

        # Retrieve candidates
        candidates = hybrid.search(
            query,
            top_k=5,
            candidate_k=10
        )

        # Rerank
        results = reranker.rerank(
            query,
            candidates,
            top_k=5
        )

        # -----------------------------------------
        # Check expected source
        # -----------------------------------------

        retrieved_sources = [
            r["source"]
            for r in results
        ]

        source_found = expected_source in retrieved_sources

        if source_found:
            correct_source += 1

        # -----------------------------------------
        # Check relevant keywords
        # -----------------------------------------

        combined_text = " ".join(
            r["text"].lower()
            for r in results
        )

        matched_keywords = [
            keyword
            for keyword in keywords
            if keyword.lower() in combined_text
        ]

        if len(matched_keywords) == len(keywords):
            keyword_hits += 1

        print(
            f"Expected source: {expected_source}"
        )

        print(
            f"Retrieved sources: {retrieved_sources}"
        )

        print(
            f"Keywords found: "
            f"{matched_keywords}"
        )

        print(
            f"Source match: "
            f"{'YES' if source_found else 'NO'}"
        )

    # -----------------------------------------
    # Calculate metrics
    # -----------------------------------------

    retrieval_precision = (
        correct_source / total
        if total > 0
        else 0
    )

    retrieval_keyword_recall = (
        keyword_hits / total
        if total > 0
        else 0
    )

    print("\n" + "-" * 60)

    print(
        f"Source Retrieval Accuracy: "
        f"{retrieval_precision * 100:.2f}%"
    )

    print(
        f"Keyword Recall: "
        f"{retrieval_keyword_recall * 100:.2f}%"
    )

    return {
        "source_accuracy": retrieval_precision,
        "keyword_recall": retrieval_keyword_recall
    }


def evaluate_verification():
    """
    Evaluates whether the citation verifier correctly
    identifies supported and unsupported claims.
    """

    correct = 0
    total = len(VERIFICATION_TEST_CASES)

    print("\n" + "=" * 60)
    print("CITATION VERIFICATION EVALUATION")
    print("=" * 60)

    for i, test_case in enumerate(
        VERIFICATION_TEST_CASES,
        start=1
    ):

        claim = test_case["claim"]
        source_text = test_case["source_text"]
        expected = test_case["expected_verdict"]

        # Create a temporary chunk
        cited_chunks = [
            {
                "text": source_text,
                "source": "evaluation",
                "page": 1
            }
        ]

        result = verify_claim(
            claim,
            cited_chunks
        )

        predicted = result["verdict"]
        confidence = result["confidence"]

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        print(f"\nTest {i}")

        print(
            f"Claim: {claim}"
        )

        print(
            f"Expected: {expected}"
        )

        print(
            f"Predicted: {predicted}"
        )

        print(
            f"Confidence: {confidence}"
        )

        print(
            f"Result: "
            f"{'PASS' if is_correct else 'FAIL'}"
        )

    # -----------------------------------------
    # Verification accuracy
    # -----------------------------------------

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print("\n" + "-" * 60)

    print(
        f"Verification Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    return accuracy


if __name__ == "__main__":

    retrieval_metrics = evaluate_retrieval()

    verification_accuracy = evaluate_verification()

    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nSource Retrieval Accuracy: "
        f"{retrieval_metrics['source_accuracy'] * 100:.2f}%"
    )

    print(
        f"Keyword Recall: "
        f"{retrieval_metrics['keyword_recall'] * 100:.2f}%"
    )

    print(
        f"Citation Verification Accuracy: "
        f"{verification_accuracy * 100:.2f}%"
    )

    print("\n" + "=" * 60)