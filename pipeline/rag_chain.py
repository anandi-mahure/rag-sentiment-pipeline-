"""
RAG Customer Intelligence Pipeline
Author: Anandi M | MSc Data Science, University of Bath
Description: Retrieval-Augmented Generation pipeline for customer review
             intelligence — semantic search + LLM-powered business Q&A
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── STEP 1: PREPROCESSING ─────────────────────────────────────
def load_and_preprocess(csv_path: str) -> list[dict]:
    """Load reviews and prepare documents for embedding."""
    df = pd.read_csv(csv_path)
    
    # Clean text
    df['review_text'] = df['review_text'].str.strip()
    df['review_text'] = df['review_text'].str.replace(r'\s+', ' ', regex=True)
    df = df[df['review_text'].str.len() > 20]  # remove very short reviews
    
    # Build documents with metadata
    documents = []
    for _, row in df.iterrows():
        doc = {
            'text': row['review_text'],
            'metadata': {
                'review_id':   row.get('review_id', ''),
                'rating':      row.get('rating', 0),
                'product':     row.get('product', ''),
                'date':        row.get('date', ''),
                'sentiment':   'positive' if row.get('rating', 3) >= 4
                               else 'negative' if row.get('rating', 3) <= 2
                               else 'neutral'
            }
        }
        documents.append(doc)
    
    print(f"Loaded {len(documents)} reviews for indexing")
    return documents


# ── STEP 2: EMBEDDING + FAISS INDEX ──────────────────────────
def build_vector_index(documents: list[dict], index_path: str = "faiss_index"):
    """
    Embed documents and build FAISS vector store.
    Uses sentence-transformers for local embedding — no API key needed.
    Swap to OpenAI/Bedrock embeddings for production.
    """
    try:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.docstore.document import Document
    except ImportError:
        print("Install: pip install langchain faiss-cpu sentence-transformers")
        return None

    # Convert to LangChain Document format
    lc_docs = [
        Document(page_content=d['text'], metadata=d['metadata'])
        for d in documents
    ]

    # Load embedding model (runs locally — no API cost)
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Build and save FAISS index
    print("Building vector index...")
    vectorstore = FAISS.from_documents(lc_docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"Index saved to {index_path}/ — {len(lc_docs)} documents indexed")
    return vectorstore


# ── STEP 3: RAG QUERY ENGINE ─────────────────────────────────
def build_rag_chain(index_path: str = "faiss_index"):
    """
    Build the full RAG chain:
    Query → FAISS retrieval → LLM answer generation → source docs
    """
    try:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
    except ImportError:
        print("Install: pip install langchain faiss-cpu sentence-transformers")
        return None

    # Load saved index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.load_local(index_path, embeddings,
                                   allow_dangerous_deserialization=True)

    # Retriever — fetch top 5 most relevant reviews
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Prompt template — grounds LLM in retrieved context
    prompt_template = """
You are a customer intelligence analyst. Answer the question below using ONLY
the customer reviews provided as context. Be specific, concise and business-focused.
If the context does not contain enough information, say so clearly.

Context (customer reviews):
{context}

Business Question: {question}

Answer (2-3 sentences, specific and actionable):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # NOTE: Replace this with your chosen LLM
    # Option A — HuggingFace local model (free, slower):
    #   from langchain.llms import HuggingFacePipeline
    #   llm = HuggingFacePipeline.from_model_id("google/flan-t5-base", ...)
    #
    # Option B — AWS Bedrock (production, enterprise):
    #   import boto3
    #   from langchain.llms import Bedrock
    #   bedrock = boto3.client("bedrock-runtime", region_name="eu-west-2")
    #   llm = Bedrock(client=bedrock, model_id="anthropic.claude-v2")
    #
    # Option C — OpenAI (fast, easy):
    #   from langchain.chat_models import ChatOpenAI
    #   llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    print("RAG chain ready. LLM placeholder — connect your chosen model.")
    print("See comments above for HuggingFace, AWS Bedrock, or OpenAI options.")
    return retriever  # return retriever for semantic search without LLM


# ── STEP 4: SEMANTIC SEARCH (no LLM required) ────────────────
def semantic_search(query: str, retriever, top_k: int = 5) -> list[dict]:
    """
    Pure semantic retrieval — no LLM needed.
    Returns the most relevant reviews for any business question.
    """
    docs = retriever.get_relevant_documents(query)
    results = []
    for i, doc in enumerate(docs[:top_k]):
        results.append({
            'rank':      i + 1,
            'review':    doc.page_content,
            'rating':    doc.metadata.get('rating', 'N/A'),
            'sentiment': doc.metadata.get('sentiment', 'N/A'),
            'product':   doc.metadata.get('product', 'N/A'),
            'date':      doc.metadata.get('date', 'N/A'),
        })
    return results


# ── DEMO ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load and index (run once)
    csv_path = "data/customer_reviews.csv"
    
    if not Path("faiss_index").exists():
        docs = load_and_preprocess(csv_path)
        vectorstore = build_vector_index(docs)
    
    retriever = build_rag_chain()
    
    if retriever:
        # Example business queries
        queries = [
            "What are the most common complaints about delivery?",
            "Which product features do customers praise most?",
            "What issues are causing 1-star reviews?",
            "Are there any safety concerns mentioned in reviews?",
        ]
        
        for query in queries:
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print(f"{'='*60}")
            results = semantic_search(query, retriever)
            for r in results[:3]:
                print(f"\nRank {r['rank']} | Rating: {r['rating']} | "
                      f"Sentiment: {r['sentiment']}")
                print(f"Review: {r['review'][:200]}...")
