import os
import shutil
import time
import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import cohere

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

REQUIRED_VARS = [
    "GOOGLE_API_KEY",
    "COHERE_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
]

for var in REQUIRED_VARS:
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI(title="RAG Assessment API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# HEALTH CHECK (IMPORTANT)
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "running"}

# --------------------------------------------------
# LAZY SINGLETONS (CRITICAL FOR RENDER)
# --------------------------------------------------
_embeddings = None
_llm = None
_cohere = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    return _embeddings

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0,
            max_retries=3,
        )
    return _llm

def get_cohere_client():
    global _cohere
    if _cohere is None:
        _cohere = cohere.Client(os.getenv("COHERE_API_KEY"))
    return _cohere

def get_vectorstore():
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        collection_name="rag_assessment_v2",
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

# --------------------------------------------------
# MODELS
# --------------------------------------------------
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

# --------------------------------------------------
# INGEST
# --------------------------------------------------
@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(None),
    text: str = Form(None),
):
    if not file and not text:
        raise HTTPException(status_code=400, detail="Provide file or text")

    temp_path = None

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            add_start_index=True,
        )

        docs = []

        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = file.filename

            splits = splitter.split_documents(docs)

        else:
            splits = splitter.create_documents(
                [text],
                metadatas=[{"source": "User Input"}],
            )

        if splits:
            QdrantVectorStore.from_documents(
                splits,
                embedding=get_embeddings(),
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name="rag_assessment_v2",
                force_recreate=False,
            )

        return {"status": "success", "chunks": len(splits)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --------------------------------------------------
# CHAT
# --------------------------------------------------
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    start = time.time()

    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(request.query)

        if not docs:
            return QueryResponse(
                answer="No information found",
                citations=[],
                processing_time=0,
            )

        # RERANK
        try:
            co = get_cohere_client()
            rerank = co.rerank(
                model="rerank-english-v3.0",
                query=request.query,
                documents=[d.page_content for d in docs],
                top_n=3,
            )
            top_docs = [docs[r.index] for r in rerank.results]
        except Exception:
            top_docs = docs[:3]

        context = "\n\n".join(d.page_content for d in top_docs)

        prompt = ChatPromptTemplate.from_template(
            """
            Answer ONLY from the context.
            If unknown, say "I cannot answer".

            Context:
            {context}

            Question:
            {question}
            """
        )

        chain = prompt | get_llm() | StrOutputParser()
        answer = chain.invoke(
            {"context": context, "question": request.query}
        )

        citations = [
            Citation(
                text=d.page_content[:200] + "...",
                source=d.metadata.get("source", "Unknown"),
            )
            for d in top_docs
        ]

        return QueryResponse(
            answer=answer,
            citations=citations,
            processing_time=round(time.time() - start, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
