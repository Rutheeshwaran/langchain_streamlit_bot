import streamlit as st
import os

from rag_core import (
    retrieve_chunks,
    generate_answer,
    ingest_pdf
)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student RAG Bot",
    page_icon="📘",
    layout="wide"
)

# ================= LIGHT UI + BLACK TEXT =================
st.markdown("""
<style>

/* Page background */
body {
    background-color: #f3f4f6;
}

/* Headings */
h1, h2, h3 {
    color: #000000 !important;
}

/* Normal text */
p, label, span, div {
    color: #000000 !important;
}

/* Caption */
.stCaption {
    color: #000000 !important;
}

/* Input box */
.stTextInput input,
.stTextInput input:focus,
.stTextInput input:active,
.stTextInput input:hover {
    background-color: #e5e7eb !important;
    color: #000000 !important;
    border-radius: 8px;
    border: 1px solid #9ca3af;
    box-shadow: none !important;
    outline: none !important;
}

/* Placeholder */
.stTextInput input::placeholder {
    color: #4b5563;
}

/* Button */
.stButton > button {
    background-color: #2563eb;
    color: #ffffff !important;
    border-radius: 8px;
    padding: 0.5em 1.2em;
    border: none;
}
.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Answer box */
.stAlert {
    background-color: #e5e7eb;
    color: #000000 !important;
    border-left: 5px solid #2563eb;
}

/* Chunk box */
.chunk {
    background-color: #d1d5db;
    color: #000000;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #9ca3af;
}

/* Expander title */
.st-expanderHeader {
    color: #000000 !important;
}

/* Animations */
.fade-in {
    animation: fadeIn 0.6s ease-in;
}
.slide-up {
    animation: slideUp 0.5s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from { transform: translateY(15px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1 class='fade-in'>📘 Student RAG Bot</h1>", unsafe_allow_html=True)
st.caption("Notebook-style PDF ingestion • Explainable RAG • Qdrant + LLM")
st.divider()

# ================= PDF UPLOAD =================
st.subheader("📄 Upload PDF (Save → Embed → Store)")

uploaded_pdf = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

if uploaded_pdf:
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_pdf.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    with st.spinner("📥 Saving, embedding, and ingesting PDF..."):
        chunks_added = ingest_pdf(pdf_path)

    st.success(f"✅ PDF stored and {chunks_added} chunks embedded")
    st.divider()

# ================= QUESTION INPUT =================
st.markdown("<h3 class='slide-up'>💬 Ask your question</h3>", unsafe_allow_html=True)

question = st.text_input(
    "",
    placeholder="Example: Create a quiz from the SQL project PDF",
    label_visibility="collapsed"
)

# ================= ASK BUTTON =================
if st.button("Ask") and question:

    with st.spinner("🔍 Retrieving relevant chunks..."):
        chunks = retrieve_chunks(question)

    with st.spinner("🧠 Generating answer..."):
        answer = generate_answer(question, chunks)

    # ================= ANSWER =================
    st.subheader("🤖 Answer")
    st.write(answer)

    # ================= EVIDENCE =================
    st.subheader("📌 Evidence (Chunks Used)")

    for i, c in enumerate(chunks, start=1):
        with st.expander(
            f"Chunk {i} | Source: {c['source']} | Score: {round(c['score'], 3)}"
        ):
            st.markdown(
                f"<div class='chunk'>{c['text']}</div>",
                unsafe_allow_html=True
            )

# ================= FOOTER =================
st.divider()
st.caption("✔ PDF saved locally | ✔ Embedded in Qdrant | 🎓 Student RAG Project")
