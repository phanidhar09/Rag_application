🚀 RAG Chat Application (LangChain + Streamlit + Tavily + Chroma)

An end-to-end Retrieval-Augmented Generation (RAG) application with a chat interface, built using:

🧠 LangChain (RAG pipeline)
🌐 Tavily (web crawling)
📄 ChromaDB (vector store)
🤖 OpenRouter (Qwen LLM)
💬 Streamlit (UI)
📌 Features
💬 ChatGPT-style UI using Streamlit
🌐 Crawl & ingest web data dynamically
🧠 Context-aware responses (RAG)
⚡ Persistent vector database
🔄 Async ingestion pipeline
🧱 Architecture
User (Streamlit UI)
        ↓
   User Query
        ↓
   Chroma Retriever
        ↓
Relevant Documents
        ↓
Prompt + Context
        ↓
LLM (Qwen via OpenRouter)
        ↓
   Final Answer
📂 Project Structure
doc-RAG/
│
├── ingestion.py        # RAG pipeline (crawl + embed + retrieve)
├── app.py              # Streamlit chat UI
├── chroma_db/          # Vector DB (auto-created)
├── .env                # API keys
├── logger.py           # Logging utility
└── README.md
⚙️ Setup
1️⃣ Clone Repository
git clone https://github.com/<your-username>/doc-RAG.git
cd doc-RAG
2️⃣ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
3️⃣ Install Dependencies
pip install langchain langchain-openai langchain-chroma langchain-huggingface \
langchain-tavily chromadb sentence-transformers python-dotenv certifi streamlit
4️⃣ Environment Variables

Create .env file:

OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
▶️ Run the App
streamlit run app.py
💬 How to Use
Step 1: Ingest Data
Click 📥 Ingest Documents in sidebar
Crawls LangChain docs via Tavily
Stores embeddings in ChromaDB
Step 2: Ask Questions
Type your question in chat
App retrieves relevant chunks
LLM generates contextual answer
Step 3: Chat Experience
Maintains conversation history
Clear chat anytime using sidebar
🔄 Workflow Explained
1. Ingestion Pipeline (ingestion.py)
Crawl website using Tavily
Extract content
Split into chunks
Generate embeddings
Store in ChromaDB
2. Retrieval
retriever = chroma.as_retriever(search_kwargs={"k": 3})
3. RAG Prompting
Combine query + retrieved context
Send to LLM
4. Streamlit UI (app.py)
Handles chat interaction
Calls async backend functions
Displays messages
🧪 Example Query
How to use LangChain for AI agents?
⚡ Key Highlights
✅ Real-world GenAI project (RAG + UI)
✅ Uses modern stack (LangChain + LLM APIs)
✅ Production-style modular design
✅ Great for interviews & portfolio
❗ Common Issues
🔴 Async Errors in Streamlit

Use:

asyncio.run()
🔴 Empty Responses
Ensure ingestion completed
Check API keys
🔴 Slow Performance
Reduce Tavily crawl depth
Lower chunk size
🚀 Future Enhancements
📄 Upload PDFs / custom documents
🧠 Add memory (conversation context)
🔁 LangGraph agent workflows
🔍 Hybrid search (BM25 + vector)
🌍 Multi-source ingestion
🧠 Tech Stack
LangChain
Tavily
ChromaDB
HuggingFace Embeddings
OpenRouter (Qwen)
Streamlit
👨‍💻 Author

Phanidhar Reddy