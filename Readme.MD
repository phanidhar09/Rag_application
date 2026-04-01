🚀 RAG AI Agent using LangChain, Tavily, Chroma & OpenRouter

A Retrieval-Augmented Generation (RAG) system that crawls web content, stores it in a vector database, and answers user queries using an LLM with contextual grounding.

📌 Overview

This project builds an end-to-end RAG pipeline:

🌐 Crawls documentation using Tavily
📄 Splits and embeds content using HuggingFace
🧠 Stores embeddings in ChromaDB
🔍 Retrieves relevant context for queries
🤖 Generates answers using Qwen (OpenRouter)
🧱 Architecture


User Query
    ↓
Chroma Retriever
    ↓
Relevant Documents
    ↓
Context + Prompt
    ↓
LLM (Qwen via OpenRouter)
    ↓
Final Answer




⚙️ Setup
1. Clone Repo

git clone https://github.com/<your-username>/doc-RAG.git
cd doc-RAG

2. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate   

3. Install Dependencies

pip install langchain langchain-openai langchain-chroma langchain-huggingface \
langchain-tavily chromadb sentence-transformers python-dotenv certifi

4. Environment Variables

Create a .env file:

OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key


▶️ Run the Application
python ingestion.py
🔄 How It Works
1. Web Crawling
Uses Tavily to crawl https://docs.langchain.com/
Extracts structured content
2. Document Processing
Filters meaningful text
Splits into chunks using:
chunk_size = 800
chunk_overlap = 100
3. Embedding Generation
Model: sentence-transformers/all-mpnet-base-v2
Converts text into vector embeddings
4. Vector Storage
Stores embeddings in ChromaDB
Persists locally (chroma_db/)
5. Retrieval
Fetches top-k relevant documents (k=3)
Supports MMR for better diversity
6. LLM Response
Model: qwen/qwen-2.5-7b-instruct
Uses retrieved context to generate grounded answers
🧪 Example
query = "How to use LangChain for AI agents?"

Output:

FINAL ANSWER:
<Generated response based on retrieved documents>
⚡ Key Features
✅ End-to-end RAG pipeline
✅ Async ingestion pipeline
✅ Persistent vector database
✅ Context-aware Q&A
✅ Modular and extensible design
❗ Common Issues
1. No Results Returned
Check Tavily API key
Increase crawl depth
2. TypeError: string indices must be integers

Fix return format:

return {"answer": response.content}
3. Slow Performance
Reduce crawl depth/pages
Lower chunk size
🚀 Future Enhancements
📄 PDF / document upload support
💬 Streamlit UI (ChatGPT-style)
🔁 LangGraph agent workflows
🔍 Hybrid search (vector + keyword)
🤖 Multi-model support
🧠 Tech Stack
LangChain
Tavily
ChromaDB
HuggingFace Embeddings
OpenRouter (Qwen)
Python (AsyncIO)
👨‍💻 Author

Phanidhar Reddy

Full Stack Developer | GenAI Engineer
Experience in RAG, LangChain, Cloud & Microservices