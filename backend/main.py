import os
import shutil
import time
import tempfile
import asyncio
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient # Added for manual client connection

# --- Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load Environment Variables
load_dotenv()

app = FastAPI(title="RAG Assessment API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Clients ---

def get_cohere_client():
    key = os.getenv("COHERE_API_KEY")
    if not key:
        raise ValueError("COHERE_API_KEY missing")
    return cohere.Client(key)

def get_embeddings():
    # USES GOOGLE API (Remote, Lightweight, 768 dimensions)
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY missing")
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError("QDRANT credentials are missing")
    return QdrantClient(url=url, api_key=api_key)

def get_vectorstore():
    # Returns the vectorstore object WITHOUT checking if collection exists yet
    # This prevents the "Collection not found" crash on startup
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name="rag_google_v4",
        embedding=get_embeddings()
    )

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0,
        max_retries=3,
    )

# --- Data Models ---

class QueryRequest(BaseModel):
    query: str

class Citation(BaseModel):
    text: str
    source: str
    page: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    processing_time: float

# --- Endpoints ---

@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(None), 
    text: str = Form(None)
):
    temp_path = None
    try:
        if not file and not text:
            raise HTTPException(status_code=400, detail="Provide file or text.")

        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, add_start_index=True
        )
        splits = []

        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            for split in splits:
                split.metadata["source"] = file.filename

        elif text:
            splits = text_splitter.create_documents(
                texts=[text], metadatas=[{"source": "User Input"}]
            )

        # Indexing with Smart Creation Logic
        if splits:
            batch_size = 20
            total_batches = len(splits) // batch_size + 1
            print(f"Ingesting {len(splits)} chunks in {total_batches} batches...")

            # We use the generic client to pass config
            client = get_qdrant_client()
            embeddings = get_embeddings()
            
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i+batch_size]
                if not batch: continue
                
                try:
                    # FIX: If it's the very first batch ever, we use from_documents to CREATE the collection
                    if i == 0:
                        QdrantVectorStore.from_documents(
                            batch,
                            embeddings,
                            url=os.getenv("QDRANT_URL"),
                            api_key=os.getenv("QDRANT_API_KEY"),
                            collection_name="rag_google_v4",
                            force_recreate=False # Don't delete if it already exists, just append
                        )
                    else:
                        # For subsequent batches, we just add to existing
                        vectorstore = get_vectorstore()
                        vectorstore.add_documents(batch)
                    
                    # Sleep to respect Google Rate Limits
                    await asyncio.sleep(1.0) 
                    
                except Exception as e:
                    print(f"Batch {i} failed: {e}")
                    # Allow one retry
                    await asyncio.sleep(5)
                    try:
                        vectorstore = get_vectorstore()
                        vectorstore.add_documents(batch)
                    except Exception as retry_e:
                        print(f"Retry failed: {retry_e}")

        return {"status": "success", "message": f"Indexed {len(splits)} chunks."}

    except Exception as e:
        print(f"Ingest Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    start_time = time.time()
    try:
        vectorstore = get_vectorstore()
        
        # 1. Retrieve
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            initial_docs = retriever.invoke(request.query)
        except Exception as e:
            # Handle case where collection doesn't exist yet (Chat before Ingest)
            print(f"Retrieval failed (likely empty DB): {e}")
            return {
                "answer": "Please upload a document first.", 
                "citations": [], 
                "processing_time": 0
            }
        
        if not initial_docs:
            return {
                "answer": "No information found.", 
                "citations": [], 
                "processing_time": 0
            }

        # 2. Rerank
        top_docs = []
        try:
            co = get_cohere_client()
            doc_texts = [d.page_content for d in initial_docs]
            
            rerank_results = co.rerank(
                model="rerank-english-v3.0",
                query=request.query,
                documents=doc_texts,
                top_n=3
            )
            
            for result in rerank_results.results:
                top_docs.append(initial_docs[result.index])
        except Exception as e:
            print(f"Reranking skipped ({e}), utilizing top retrieved docs.")
            top_docs = initial_docs[:3]

        # 3. Generate
        context_str = "\n\n".join([d.page_content for d in top_docs])
        
        template = """
        Answer based ONLY on the context below. If unknown, say "I cannot answer".
        
        Context:
        {context}
        
        Question: 
        {question}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | get_llm() | StrOutputParser()
        answer_text = chain.invoke({"context": context_str, "question": request.query})
        
        citations = [
            {"text": doc.page_content[:200] + "...", "source": doc.metadata.get("source", "Unknown")} 
            for doc in top_docs
        ]

        return {
            "answer": answer_text,
            "citations": citations,
            "processing_time": round(time.time() - start_time, 2)
        }

    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
