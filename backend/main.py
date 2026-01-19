import os
import shutil
import time
import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import cohere  # Direct Cohere Client

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# FIXED: New import path to remove the Deprecation Warning
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load Environment Variables
load_dotenv()

app = FastAPI(title="RAG Assessment API", version="1.0")

# CORS: Allow connection from your Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to your Vercel URL
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
    # Uses local CPU model (Free, No Rate Limits)
    # This removes the "429 Resource Exhausted" error from Google
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError("QDRANT credentials are missing")

    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        # Collection name matches the 384-dimension size of MiniLM
        collection_name="rag_assessment_v2", 
        url=url,
        api_key=api_key
    )

def get_llm():
    # Uses the stable Flash alias to avoid "404 Not Found" errors
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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, add_start_index=True
        )
        splits = []

        # Handle File Upload
        if file:
            # Create a temporary file safely (Works on Read-Only Cloud Filesystems)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            
            # Attach metadata
            for split in splits:
                split.metadata["source"] = file.filename

        # Handle Raw Text Paste
        elif text:
            splits = text_splitter.create_documents(
                texts=[text], metadatas=[{"source": "User Input"}]
            )

        # Indexing
        if splits:
            QdrantVectorStore.from_documents(
                splits,
                get_embeddings(),
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name="rag_assessment_v2", 
                force_recreate=False 
            )
        
        return {"status": "success", "message": f"Indexed {len(splits)} chunks."}

    except Exception as e:
        print(f"Ingest Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    start_time = time.time()
    try:
        vectorstore = get_vectorstore()
        
        # 1. Retrieve (Get Top 10)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        initial_docs = retriever.invoke(request.query)
        
        if not initial_docs:
            return {
                "answer": "No information found.", 
                "citations": [], 
                "processing_time": 0
            }

        # 2. Rerank (Cohere)
        try:
            co = get_cohere_client()
            doc_texts = [d.page_content for d in initial_docs]
            
            rerank_results = co.rerank(
                model="rerank-english-v3.0",
                query=request.query,
                documents=doc_texts,
                top_n=3
            )
            
            # Map back to original documents
            top_docs = []
            for result in rerank_results.results:
                top_docs.append(initial_docs[result.index])

        except Exception as e:
            print(f"Reranking failed ({str(e)}), falling back to standard retrieval.")
            top_docs = initial_docs[:3]

        # 3. Generate (Gemini)
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
        
        # 4. Format Citations
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
    # DYNAMIC PORT BINDING (Crucial for Render/Heroku)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)