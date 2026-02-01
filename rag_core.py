import os
import uuid
from functools import lru_cache
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from qdrant_client.http.models import PointStruct

# ================= ENV =================
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
COLLECTION = "student_semantic_cloud_v2"
TOP_K = 3

if not QDRANT_URL or not QDRANT_API_KEY or not HF_TOKEN:
    raise ValueError("❌ Environment variables not loaded properly")

# ================= CACHED LOADERS =================
@lru_cache(maxsize=1)
def get_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_qdrant():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

@lru_cache(maxsize=1)
def get_llm():
    return InferenceClient(api_key=HF_TOKEN)

embed_model = get_embed_model()
qdrant = get_qdrant()
llm = get_llm()

# ================= RETRIEVAL =================
def retrieve_chunks(question):
    q_vec = embed_model.encode(question).tolist()

    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=TOP_K,
        with_payload=True
    ).points

    chunks = []
    for h in hits:
        chunks.append({
            "text": h.payload["text"],
            "source": h.payload.get("source", "unknown"),
            "score": round(h.score, 4)
        })

    return chunks

# ================= GENERATION =================
def generate_answer(question, chunks):
    context = "\n\n".join([c["text"] for c in chunks])

    prompt = f"""
You are a helpful academic assistant.

Using ONLY the context below, write a clear and well-structured paragraph
(3–5 sentences) that fully answers the question.

Do NOT use bullet points.
Explain like a project report.

If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    completion = llm.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.2
    )

    return completion.choices[0].message.content

# ================= INGEST PDF =================
def ingest_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + 200]))
        i += 150

    embeddings = embed_model.encode(chunks)

    points = []
    for chunk, emb in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "text": chunk,
                    "source": os.path.basename(file_path)
                }
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=points
    )

    return len(points)
# ================= RAG PIPELINE (FOR AGENT) =================
def rag_answer(question: str) -> str:
    chunks = retrieve_chunks(question)

    if not chunks:
        return "I don't know"

    answer = generate_answer(question, chunks)
    return answer
