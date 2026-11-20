import time
import streamlit as st
import textwrap

from ingestion.document_loader import load_documents
from ingestion.chunker import chunk_documents
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker
from models.llm_client import generate_answer
from models.citation_parser import parse_citations
from models.citation_verifier import verify_answer


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG Knowledge Engine",
    page_icon="✦",
    layout="wide"
)




# =========================================================
# HTML HELPER
# =========================================================
# IMPORTANT: st.markdown(..., unsafe_allow_html=True) only renders raw HTML
# if the string has NO leading whitespace on each line. If a multi-line
# string is indented (as it naturally is when written inside indented
# Python code), Markdown treats 4+ leading spaces as a CODE BLOCK and
# prints the HTML as literal text instead of rendering it. That's exactly
# what was happening in the screenshots. textwrap.dedent() strips the
# common leading whitespace so Markdown parses it as real HTML.

def html(content: str):
    st.markdown(textwrap.dedent(content), unsafe_allow_html=True)


# =========================================================
# SIMPLE MODERN UI
# =========================================================

html("""
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background: #090d0c;
    color: #f4f7f5;
}

.block-container {
    max-width: 1100px;
    padding-top: 40px;
    padding-bottom: 50px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* =========================================================
   HEADER
   ========================================================= */

.header {
    margin-bottom: 85px;
}

.logo {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-icon {
    width: 38px;
    height: 38px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 10px;

    background: #48d6a0;
    color: #07110d;

    font-size: 19px;
    font-weight: 800;

    box-shadow: 0 8px 25px rgba(72, 214, 160, 0.12);
}

.logo-text {
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.2px;
}

.logo-subtitle {
    color: #70807a;
    font-size: 11px;
    margin-top: 3px;
}


/* =========================================================
   HERO
   ========================================================= */

.eyebrow {
    color: #48d6a0;

    font-size: 10px;
    font-weight: 700;

    letter-spacing: 3px;

    margin-bottom: 14px;
}

h1 {
    font-size: 52px !important;
    line-height: 1.05 !important;

    letter-spacing: -2.5px !important;

    margin: 0 0 20px 0 !important;
}

.gradient {
    color: #48d6a0;
}

.description {
    max-width: 720px;

    color: #899a94;

    font-size: 15px;
    line-height: 1.75;

    margin-bottom: 28px;
}


/* =========================================================
   STATUS
   ========================================================= */

.status {
    display: flex;
    align-items: center;
    gap: 9px;

    color: #84938e;

    font-family: monospace;
    font-size: 12px;

    margin-bottom: 42px;
}

.dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #48d6a0;

    box-shadow: 0 0 10px rgba(72, 214, 160, 0.45);
}


/* =========================================================
   QUESTION
   ========================================================= */

.question-label {
    font-size: 13px;
    font-weight: 600;

    color: #e9eeec;

    margin-bottom: 9px;
}

div[data-testid="stTextInput"] {
    margin-bottom: 8px;
}

div[data-testid="stTextInput"] input {

    background: #111715 !important;

    border: 1px solid #2a3431 !important;

    border-radius: 10px !important;

    color: #f5f7f6 !important;

    height: 52px !important;

    padding: 0 16px !important;

    font-size: 14px !important;

    transition: all 0.2s ease;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #65736e !important;
}

div[data-testid="stTextInput"] input:focus {

    border-color: #48d6a0 !important;

    box-shadow: 0 0 0 1px #48d6a0 !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

div.stButton > button {

    background: #48d6a0 !important;

    color: #07110d !important;

    border: none !important;

    border-radius: 9px !important;

    font-size: 13px !important;

    font-weight: 700 !important;

    padding: 10px 23px !important;

    min-height: 42px !important;

    margin-top: 5px;

    transition: all 0.2s ease;
}

div.stButton > button:hover {

    background: #5ee2ae !important;

    transform: translateY(-1px);

    box-shadow: 0 8px 22px rgba(72, 214, 160, 0.14);
}


/* =========================================================
   ANSWER
   ========================================================= */

.result {

    margin-top: 55px;

    padding: 25px 28px;

    border: 1px solid #202b27;

    border-radius: 14px;

    background: #0f1513;

    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
}

.result-title {

    color: #48d6a0;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 2.5px;

    margin-bottom: 14px;
}

.answer {

    color: #e9eeec;

    font-size: 15px;

    line-height: 1.75;

    max-width: 900px;
}


/* =========================================================
   SOURCES
   ========================================================= */

.sources-title {

    color: #71807a;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 2.5px;

    margin-top: 38px;

    margin-bottom: 11px;
}

.source {

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 14px 16px;

    background: #111715;

    border: 1px solid #202b27;

    border-radius: 9px;

    margin-bottom: 8px;

    font-size: 13px;

    transition: border-color 0.2s ease,
                background 0.2s ease;
}

.source:hover {

    border-color: #304039;

    background: #141b18;
}

.source-name {
    color: #dce5e1;
}

.source-score {
    color: #64746e;

    font-family: monospace;

    font-size: 11px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    margin-top: 75px;

    padding-top: 18px;

    border-top: 1px solid #1b2421;

    display: flex;

    justify-content: space-between;

    color: #52615c;

    font-family: monospace;

    font-size: 10px;

    letter-spacing: 0.5px;
}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 700px) {

    .block-container {
        padding-left: 24px;
        padding-right: 24px;
    }

    h1 {
        font-size: 40px !important;
    }

    .description {
        font-size: 14px;
    }

    .result {
        padding: 21px;
    }

    .footer {
        flex-direction: column;
        gap: 8px;
    }
}

</style>
""")


# =========================================================
# HYBRID RAG PIPELINE
# =========================================================

@st.cache_resource
def build_pipeline():
    documents = load_documents("data/raw_docs")

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        overlap=50
    )

    if not chunks:
        return documents, chunks, None, None

    hybrid = HybridSearch()
    hybrid.build_index(chunks)

    reranker = Reranker()

    return documents, chunks, hybrid, reranker



def run_rag_pipeline(question, hybrid, reranker):

    start = time.perf_counter()

    candidates = hybrid.search(
        question,
        top_k=10,
        candidate_k=10
    )

    print(f"Hybrid Search: {time.perf_counter() - start:.2f}s")

    if not candidates:
        return None, [], []

    start = time.perf_counter()

    top_chunks = reranker.rerank(
        question,
        candidates,
        top_k=3
    )

    print(f"Reranking: {time.perf_counter() - start:.2f}s")

    if not top_chunks:
        return None, [], []

    start = time.perf_counter()

    answer = generate_answer(question, top_chunks)

    print(f"Answer Generation: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()

    parsed = parse_citations(answer, top_chunks)

    verified = verify_answer(parsed)

    print(f"Citation Verification: {time.perf_counter() - start:.2f}s")

    return answer, top_chunks, verified

# =========================================================
# LOAD DATA
# =========================================================

documents, chunks, hybrid, reranker = build_pipeline()


# =========================================================
# HEADER
# =========================================================

html("""
<div class="header">
    <div class="logo">
        <div class="logo-icon">✦</div>
        <div>
            <div class="logo-text">RAG Knowledge Engine</div>
            <div class="logo-subtitle">Private document intelligence</div>
        </div>
    </div>
</div>
""")


# =========================================================
# HERO
# =========================================================

html("""
<div class="eyebrow">DOCUMENT INTELLIGENCE</div>
<h1>Ask your <span class="gradient">documents.</span></h1>
<div class="description">
    Search your documents, retrieve the most relevant information,
    and generate grounded answers using your local LLM.
</div>
""")


# =========================================================
# STATUS
# =========================================================

html(f"""
<div class="status">
    <span class="dot"></span>
    <span>RAG pipeline ready</span>
    <span>·</span>
    <span>{len(documents)} documents</span>
    <span>·</span>
    <span>{len(chunks)} chunks</span>
</div>
""")


# =========================================================
# QUESTION
# =========================================================

html('<div class="question-label">Ask a question</div>')

question = st.text_input(
    "",
    placeholder="e.g. What is the leave policy?",
    label_visibility="collapsed"
)

ask = st.button("Ask →")


# =========================================================
# ANSWER
# =========================================================

if ask:
    if not question.strip():
        st.warning("Please enter a question.")

    elif not chunks or hybrid is None:
        st.error(
            "No documents found. Add PDF files inside data/raw_docs."
        )

    else:
        try:
            start_time = time.perf_counter()

            with st.spinner("Searching and verifying your documents..."):
                answer, results, verified = run_rag_pipeline(
                question, hybrid, reranker
    )

            elapsed_time = time.perf_counter() - start_time
            st.info(f"Response generated in {elapsed_time:.2f} seconds")

            if answer is None:
                st.warning("No relevant document chunks were found.")

            else:
                st.markdown("### ANSWER")
                st.write(answer)

                st.markdown("### VERIFICATION")

                if not verified:
                    st.warning(
                        "No verifiable cited claims were found. "
                        "The answer has not been verified."
                    )

                for item in verified:
                    if not item["cited_chunks"]:
                        st.error("Unverified — No source citation")

                    elif item["verdict"] == "SUPPORTED":
                        st.success(
                        f"Supported ({item['confidence']}%)"
                        )

                    else:
                        st.warning(
                        f"Not supported ({item['confidence']}%)"
                        )

                    st.write(item["claim"])

                st.markdown("### SOURCES")

                for i, result in enumerate(results, start=1):
                    st.write(
                        f"[Source {i}] {result['source']} "
                        f"— Page {result['page']}"
                    )

        except Exception as e:
            st.error(f"Pipeline error: {e}")

# =========================================================
# FOOTER
# =========================================================

html("""
<div class="footer">
    <span>LOCAL · PRIVATE · GROUNDED</span>
    <span>LLAMA 3.2</span>
</div>
""")