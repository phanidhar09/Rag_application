import asyncio
import os
import ssl
import certifi

from typing import Any,Dict,List

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl,TavilyExtract,TavilyMap,TavilySearch
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI, ChatOpenAI

from logger import (Colors,log_error,log_header,log_info,log_success,log_warning)

#SSL
ssl_content = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

#MODEL 
#llm= ChatGroq(model="llama-3.3-70b-versatile")
llm_ollama= ChatOpenAI(model= "qwen/qwen-2.5-7b-instruct",
    base_url="https://openrouter.ai/api/v1",  
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.3,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",  # optional
        "X-Title": "RAG-App"
    }
    )

#Embeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

#VECTOR DB
PERSIST_DIR = "chroma_db" 
chroma = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

#TAVILY
tavily_extract = TavilyExtract()
tavily_map =TavilyMap(max_depth=5, max_breadth=20, max_pages=1000,limit=500)
tavily_crawl = TavilyCrawl()


url = "https://docs.langchain.com/"
site_map = tavily_map.invoke(url)

#----------------INGESTION--------------####
async def ingest():
    log_header("Ingestion Pipeline")
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        log_info("Using existing vector DB (skip ingestion)", Colors.YELLOW)
        return
    
    log_info("Tavily Crawl", Colors.PURPLE)

    res = tavily_crawl.invoke({ 
        "url":"https://docs.langchain.com/",
        "max_depth":3,
        "extract_depth": "advanced",
        "instructions": "content of using Python in AI agents"
    })

    all_docs = [
    Document(page_content=r["raw_content"], metadata={"source": r["url"]})
    for r in res["results"]
    if r.get("raw_content") and len(r["raw_content"]) > 200
    ]
    
    log_success(f"Tavily Crawl sucessfull with {len(all_docs)} ", Colors.GREEN)

   # log_info("Tavily map", Colors.PURPLE)

   # site_map= tavily_map.invoke("https://docs.langchain.com/")
   # log_success(f"mapped with TavilyMap {len(site_map['results'])} URLs from docs",Colors.GREEN)


    log_header("Document Chunking")
    log_info(f"Text splitter",Colors.YELLOW)

    text_splitter= RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text splitter")

    log_header("Storing in Chroma DB")
    chroma.add_documents(split_docs)
    chroma.persist()
    log_success(f"Stored {len(split_docs)} chunks in Chroma", Colors.GREEN)




#----------RAG-------------------#
async def ask_question(query: str):
    log_header("RAG Query")
    retriever = chroma.as_retriever(search_kwargs={"k": 3})   
    docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{query}
"""

    response = llm_ollama.invoke(prompt)
    sources = list(set([d.metadata["source"] for d in docs]))

    return response.content


##--- DOCUMNET SEARCH ----##########

#async def search_documents(query: str):
    log_header("Document Search")

    retriever = chroma.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )

    docs = retriever.invoke(query)

    if not docs:
        return {
            "answer": "❌ No relevant documents found.",
            "sources": [],
            "documents": []
        }

    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a document search assistant.

Your job:
1. Find relevant information from the documents
2. Summarize clearly
3. DO NOT hallucinate
4. If not found, say "Not found in documents"

Context:
{context}

Query:
{query}

Return:
- Short Answer
- Key Points (bullet list)
"""

    response = llm_ollama.invoke(prompt)

    # Extract sources
    sources = list(set([d.metadata.get("source", "unknown") for d in docs]))

    return {
        "answer": response.content,
        "sources": sources,
        "documents": docs
    }


async def main():
    await ingest()

    query = "How to use LangChain for AI agents?"

    answer = await ask_question(query)
    print("\n\nFINAL ANSWER:\n")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())