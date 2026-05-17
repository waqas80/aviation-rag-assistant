import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
import os

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Aviation Manual Assistant",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Aviation Manual Assistant")
st.caption("Ask any question about your maintenance manual")

# ── Load models once (cached so they don't reload on every message)
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))
    return embedding_model, groq_client

@st.cache_resource
def load_knowledge_base():
    embedding_model, _ = load_models()

    # Aviation manual content
    manual_text = """
AIRCRAFT MAINTENANCE MANUAL - A380
Section 5.3 - Landing Gear Inspection

Crack Limit Policy:
Any crack found on main landing gear components must not exceed 0.5mm in depth.
Cracks exceeding this limit require immediate grounding of the aircraft and
replacement of the affected component before next flight.

Corrosion Standards:
Surface corrosion up to 0.2mm depth is acceptable after treatment.
Corrosion exceeding 0.3mm requires engineering review and possible replacement.
All corrosion must be treated within 72 hours of discovery.

Inspection Intervals:
Daily walkaround requires visual inspection of all gear doors.
A-Check at 500 hours requires detailed inspection of hinge brackets.
C-Check at 6000 hours requires full disassembly and NDT inspection.
D-Check at 25000 hours requires complete overhaul of all components.

Hydraulic Pressure:
Normal operating pressure is 5000 PSI.
Minimum acceptable pressure is 4800 PSI.
Any reading below 4800 PSI requires immediate maintenance action.

Oil System:
Normal oil consumption limit is 0.3 qt per hour.
Oil pressure normal range is 50 to 65 PSI during cruise.
Oil pressure below 40 PSI requires immediate engine shutdown.

Cabin Pressure:
Fault code PRESS-04 indicates intermittent pressure controller issue.
Backup system must be verified functional before next flight.
Three or more occurrences within one flight requires ground inspection.
"""

    # Split into paragraphs
    chunks = [c.strip() for c in manual_text.split("\n\n") if c.strip()]

    # Embed all chunks using numpy (no ChromaDB needed)
    embeddings = embedding_model.encode(chunks)

    return chunks, embeddings

# ── Simple vector search using numpy ─────────────────────────────
def search(query, chunks, embeddings, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])[0]

    # Cosine similarity = dot product of normalized vectors
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    similarities = np.dot(embeddings, query_embedding) / (norms * query_norm)

    # Get top_k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# ── Initialize ────────────────────────────────────────────────────
embedding_model, groq_client = load_models()
chunks, embeddings = load_knowledge_base()

# ── Chat history ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display existing messages ─────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Handle new input ──────────────────────────────────────────────
if question := st.chat_input("Ask about the maintenance manual..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):

            relevant_chunks = search(
                question, chunks, embeddings, embedding_model
            )
            context = "\n\n".join(relevant_chunks)

            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an aviation maintenance expert assistant.
Answer questions based ONLY on the provided document context.
If the answer is not in the context, say 'Not found in the manual'.
Always quote the specific value or section from the document."""
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ]
            )
            answer = response.choices[0].message.content
            st.write(answer)

            with st.expander("📄 View source sections from manual"):
                for i, chunk in enumerate(relevant_chunks):
                    st.caption(f"Section {i+1}")
                    st.text(chunk)

    st.session_state.messages.append({"role": "assistant", "content": answer})