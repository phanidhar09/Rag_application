import streamlit as st
import asyncio

# Import your RAG functions
from ingestion import ingest, ask_question

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="RAG Chat App",
    layout="wide"
)

st.title("📚 RAG Chat Application")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Controls")

# Ingest button
if st.sidebar.button("📥 Ingest Documents"):
    with st.spinner("Ingesting documents..."):
        asyncio.run(ingest())
    st.sidebar.success("Ingestion completed!")

# Optional: Clear chat
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []

# ---------------- CHAT STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT INPUT ---------------- #
query = st.chat_input("Ask your question...")

if query:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Call RAG backend
    with st.spinner("Thinking..."):
        answer = asyncio.run(ask_question(query))

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# ---------------- DISPLAY CHAT ---------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])