# evaluation/eval_dataset.py


# ============================================================
# RETRIEVAL EVALUATION DATASET
# ============================================================

EVAL_DATASET = [

    {
        "query": "What is the leave policy?",
        "expected_source": "HR_Policy_Sample.pdf",
        "relevant_keywords": ["leave", "annual"]
    },

    {
        "query": "How should employees request leave?",
        "expected_source": "HR_Policy_Sample.pdf",
        "relevant_keywords": ["leave", "employee portal"]
    },

    {
        "query": "What are the general workplace guidelines for employees?",
        "expected_source": "Employee_Handbook_Sample.pdf",
        "relevant_keywords": ["employee", "workplace"]
    },

    {
        "query": "What should employees do regarding information security?",
        "expected_source": "IT_Security_Guidelines.pdf",
        "relevant_keywords": ["security", "IT"]
    },

    {
        "query": "What should an employee do if they notice a security problem?",
        "expected_source": "IT_Security_Guidelines.pdf",
        "relevant_keywords": ["report", "security", "IT team"]
    }
]


# ============================================================
# CITATION VERIFICATION TEST CASES
# ============================================================

VERIFICATION_TEST_CASES = [

    # 1. Correct claim
    {
        "claim": "Employees are entitled to annual leave according to company policy.",
        "source_text": "Employees are entitled to annual leave according to company policy.",
        "expected_verdict": "SUPPORTED"
    },

    # 2. Clearly false claim
    {
        "claim": "Employees get unlimited leave with no approval needed.",
        "source_text": "Employees are entitled to annual leave according to company policy.",
        "expected_verdict": "NOT_SUPPORTED"
    },

    # 3. Correct claim about the employee portal
    {
        "claim": "Leave requests should normally be submitted through the employee portal before the planned absence.",
        "source_text": "Leave requests should normally be submitted through the employee portal before the planned absence.",
        "expected_verdict": "SUPPORTED"
    },

    # 4. Opposite of what the source says
    {
        "claim": "Employees can submit leave requests after their absence without using the portal.",
        "source_text": "Leave requests should normally be submitted through the employee portal before the planned absence.",
        "expected_verdict": "NOT_SUPPORTED"
    },

    # 5. Completely unrelated claim
    {
        "claim": "Employees receive unlimited free meals every day at the company cafeteria.",
        "source_text": "Employees are entitled to annual leave according to company policy.",
        "expected_verdict": "NOT_SUPPORTED"
    },

    # 6. Subtle wrong claim: annual vs sick leave
    {
        "claim": "Employees are entitled to sick leave according to company policy.",
        "source_text": "Employees are entitled to annual leave according to company policy.",
        "expected_verdict": "NOT_SUPPORTED"
    }
]


# ============================================================
# OPTIONAL: RUN THIS FILE DIRECTLY TO VIEW THE DATASET
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAG EVALUATION DATASET")
    print("=" * 60)

    print(f"\nRetrieval test cases: {len(EVAL_DATASET)}")

    for i, test in enumerate(EVAL_DATASET, start=1):
        print(f"\n{i}. {test['query']}")
        print(f"   Expected source: {test['expected_source']}")
        print(f"   Keywords: {test['relevant_keywords']}")

    print("\n" + "=" * 60)

    print(
        f"Verification test cases: "
        f"{len(VERIFICATION_TEST_CASES)}"
    )

    for i, test in enumerate(
        VERIFICATION_TEST_CASES,
        start=1
    ):
        print(f"\n{i}. {test['claim']}")
        print(f"   Expected: {test['expected_verdict']}")

    print("\n" + "=" * 60)