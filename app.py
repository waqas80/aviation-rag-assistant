import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

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
def load_vector_db():
    embedding_model, _ = load_models()
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("aviation_docs")

    # Load and index the document
    with open("aviation_manual.txt", "r") as f:
        text = f.read()

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

# ── Initialize ────────────────────────────────────────────────────
embedding_model, groq_client = load_models()
collection = load_vector_db()

# ── Chat history stored in session ────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display existing chat messages ────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Handle new user input ─────────────────────────────────────────
if question := st.chat_input("Ask about the maintenance manual..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Search and answer
    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):

            # Find relevant chunks
            query_embedding = embedding_model.encode([question]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3
            )
            context = "\n\n".join(results["documents"][0])

            # Get answer from LLM
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an aviation maintenance expert.
Answer based ONLY on the document context provided.
If not found, say 'Not found in the manual'.
Be concise and always quote the specific value."""
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ]
            )
            answer = response.choices[0].message.content
            st.write(answer)

            # Show source sections in expander
            with st.expander("📄 View source sections from manual"):
                for i, chunk in enumerate(results["documents"][0]):
                    st.caption(f"Section {i+1}")
                    st.text(chunk)

    st.session_state.messages.append({"role": "assistant", "content": answer})