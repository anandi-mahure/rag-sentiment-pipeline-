# RAG Customer Intelligence Pipeline
**Author:** Anandi M | MSc Data Science, University of Bath  
**Tools:** Python · LangChain · FAISS · HuggingFace · AWS Bedrock  
**Domain:** Generative AI · NLP · Customer Intelligence · Operational Analytics

---

## What This Project Does

A Retrieval-Augmented Generation (RAG) pipeline that combines a vector database of customer reviews with a large language model to answer specific business questions about customer sentiment — going beyond simple positive/negative classification to deliver actionable operational intelligence.

Instead of just classifying sentiment, the system answers questions like:
- *"What product features are driving negative reviews in London?"*
- *"Which complaints increased most in Q1 2024?"*
- *"Summarise the top 3 reasons customers are leaving 1-star reviews"*

---

## Why RAG — Not Just Sentiment Classification

| Approach | Limitation | How RAG Solves It |
|---|---|---|
| Rule-based keyword matching | Misses context, brittle | Semantic vector search understands meaning |
| Traditional sentiment classifier | Only outputs positive/negative | RAG answers specific business questions |
| Fine-tuned LLM alone | Static — doesn't update with new data | Vector DB updates in real time, no retraining |
| RAG pipeline | Slightly more complex to set up | Explainable, updatable, business-question-ready |

---

## Architecture

```
Customer Reviews (CSV)
        │
        ▼
[Text Chunking & Preprocessing]
        │
        ▼
[Embedding Model — sentence-transformers/all-MiniLM-L6-v2]
        │
        ▼
[FAISS Vector Store — semantic similarity index]
        │
        ▼
[Retrieval Chain — LangChain RetrievalQA]
        │
        ▼
[LLM — HuggingFace (local) / AWS Bedrock (production)]
        │
        ▼
[Grounded Answer + Source Documents]
```

---

## Project Structure

```
rag-sentiment-pipeline/
├── data/
│   └── customer_reviews.csv        # 500+ customer reviews dataset
├── pipeline/
│   ├── preprocess.py               # Text cleaning and chunking
│   ├── embed_and_index.py          # Build FAISS vector index
│   ├── rag_chain.py                # LangChain RAG pipeline
│   └── query_engine.py             # Business question interface
├── notebooks/
│   └── rag_demo.ipynb              # End-to-end walkthrough
├── outputs/
│   └── sample_queries.md           # Example questions and answers
└── README.md
```

---

## How To Run

```bash
# Install dependencies
pip install langchain faiss-cpu sentence-transformers pandas

# Step 1 — Build the vector index
python pipeline/embed_and_index.py

# Step 2 — Run the query engine
python pipeline/query_engine.py

# Or run the full demo notebook
jupyter notebook notebooks/rag_demo.ipynb
```

---

## Example Queries & Outputs

**Query:** *"What are the most common complaints about delivery?"*

**Retrieved context:** 3 most semantically similar reviews about delivery issues

**Answer:** *"Customers most frequently cite delayed dispatch (mentioned in 34% of negative delivery reviews), poor tracking updates, and damaged packaging on arrival. Issues are concentrated in orders placed Friday–Sunday, suggesting weekend fulfilment capacity constraints."*

---

## Core Pipeline Code

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# Load embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Build retrieval chain
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # retrieve top 5 most relevant reviews
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True  # explainability — show source reviews
)
```

---

## Production Architecture (AWS Bedrock)

In a production environment this pipeline connects to **AWS Bedrock** for LLM access:

```python
import boto3
from langchain.llms import Bedrock

bedrock_client = boto3.client("bedrock-runtime", region_name="eu-west-2")
llm = Bedrock(
    client=bedrock_client,
    model_id="anthropic.claude-v2"
)
```

Benefits: enterprise security, no model hosting overhead, GDPR-compliant EU hosting.

---

## What I Would Do Differently

1. **Add a re-ranking step** — initial vector retrieval returns semantically similar docs but not always the most relevant. A cross-encoder re-ranker improves answer quality significantly
2. **Add evaluation metrics** — RAGAS framework to measure retrieval precision and answer groundedness automatically
3. **Add guardrails** — prevent the model from answering outside retrieved context (hallucination prevention)
4. **Streaming responses** — for production UI, stream tokens rather than waiting for full response

---

## Connection To BERT Dissertation

This project extends my MSc dissertation work on transformer-based NLP. The dissertation used DistilBERT for sentiment classification (macro-F1: 0.78). This pipeline builds on that foundation — using transformer embeddings for semantic retrieval and an LLM for generative answering. The progression: classification → retrieval → generation.

---

## Skills Demonstrated
`Python` `LangChain` `FAISS` `HuggingFace` `RAG` `Vector Databases` `NLP` `Semantic Search` `AWS Bedrock` `Generative AI` `Customer Analytics`
