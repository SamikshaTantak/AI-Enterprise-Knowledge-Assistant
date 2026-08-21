# Enterprise Knowledge Assistant

An end-to-end **Enterprise Knowledge Assistant** built with **RAG, LangGraph, MCP, RAGAS, LangSmith, ChromaDB, Ollama, Groq, and Streamlit**.

The application allows users to ask questions about enterprise policies and documents in natural language. It retrieves relevant information, generates a context-aware answer, evaluates answer quality, and displays the result in a Streamlit UI.

---

## Features

- Retrieval-Augmented Generation (RAG)
- Semantic document search
- ChromaDB vector database
- Ollama embeddings
- LangGraph workflow orchestration
- Retriever, Response, and Evaluator agents
- MCP-based additional retrieval
- RAGAS evaluation
- LangSmith tracing and observability
- Groq LLM integration
- Streamlit UI
- Source document display
- Faithfulness and Answer Relevancy scores

---

# Architecture Overview

```text
                         ┌──────────────────────┐
                         │   Enterprise Docs    │
                         │    TXT / PDF Files   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Document Ingestion   │
                         │ Loading + Chunking   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Ollama Embeddings    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      ChromaDB        │
                         │    Vector Store      │
                         └──────────┬───────────┘
                                    │
                                    ▼
User Question ───► ┌───────────────────────────────┐
                   │       Streamlit UI            │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌───────────────────────────────┐
                   │       LangGraph Workflow      │
                   │                               │
                   │ START                         │
                   │   ↓                           │
                   │ Retriever Agent              │
                   │   ↓                           │
                   │ Response Agent               │
                   │   ↓                           │
                   │ Evaluator Agent              │
                   │   ↓                           │
                   │ END                           │
                   └──────────────┬────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                 ChromaDB        MCP        LangSmith
                 Retrieval    Retrieval    Observability
                    │
                    ▼
                 Groq LLM
                    │
                    ▼
             RAGAS Evaluation
                    │
                    ▼
              Final UI Response
```

---

# Project Structure

```text
enterprise-knowledge-assistant/
│
├── app/
│   ├── agents/
│   │   ├── retriever_agent.py
│   │   ├── response_agent.py
│   │   └── evaluator_agent.py
│   ├── config/
│   │   └── settings.py
│   ├── evaluation/
│   ├── mcp/
│   ├── observability/
│   │   └── tracing.py
│   ├── rag/
│   ├── ui/
│   │   └── streamlit_app.py
│   ├── vectorstore/
│   └── workflow/
│       ├── state.py
│       └── workflow.py
│
├── data/
├── scripts/
│   ├── ingest.py
│   ├── rag_test.py
│   └── workflow_test.py
│
├── .env
├── pyproject.toml
├── requirements.txt
├── main.py
└── README.md
```

---

# Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Workflow | LangGraph |
| RAG Framework | LangChain |
| LLM | Groq |
| Embeddings | Ollama |
| Vector Database | ChromaDB |
| Evaluation | RAGAS |
| Observability | LangSmith |
| Tool Integration | MCP |
| Configuration | `.env` |

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd enterprise-knowledge-assistant
```

## 2. Install uv

Verify that `uv` is installed:

```bash
uv --version
```

## 3. Install Dependencies

From the project root:

```bash
uv sync
```

Alternatively, if the project uses `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

Streamlit can be added with:

```bash
uv add streamlit
```

## 4. Install and Run Ollama

Ollama is used for embeddings. Make sure it is installed and available locally.

Start the service if required:

```bash
ollama serve
```

Pull the embedding model configured in the project:

```bash
ollama pull <embedding-model-name>
```

The exact model name should match the value configured in `app/config/settings.py`.

The application communicates with the local Ollama service, typically at:

```text
http://localhost:11434
```

---

# Configuration

Create a `.env` file in the project root and configure the required credentials.

Example:

```env
# Groq
GROQ_API_KEY=your_groq_api_key

# LangSmith / LangChain tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=enterprise-knowledge-assistant

# LangSmith equivalents used by the observability setup
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=enterprise-knowledge-assistant
```

The exact environment variables and additional configuration should match:

```text
app/config/settings.py
```

---

# Execution Steps

## Step 1: Add Enterprise Documents

Place enterprise documents in the project's data/document source directory.

Example documents used during testing:

```text
remote_work_policy.txt
expense_reimbursement_policy.txt
it_security_policy.pdf
```

These documents form the knowledge base used by the assistant.

## Step 2: Ingest Documents

Run the ingestion pipeline to:

1. Load documents.
2. Split them into chunks.
3. Generate embeddings.
4. Store embeddings and metadata in ChromaDB.

Example command:

```bash
uv run python -m scripts.ingest
```

If your ingestion module differs, use the corresponding command implemented in the project.

## Step 3: Test RAG Retrieval

Test the retrieval pipeline:

```bash
uv run python -m scripts.rag_test
```

This is useful for confirming that relevant chunks are being returned before testing the complete workflow.

## Step 4: Test the Complete Workflow

Run:

```bash
uv run python -m scripts.workflow_test
```

Sample input:

```text
How much can I expense for internet per month?
```

The workflow performs:

```text
Query
  ↓
ChromaDB Retrieval + MCP Retrieval
  ↓
Merge Relevant Context
  ↓
Generate Answer
  ↓
RAGAS Evaluation
  ↓
Final Answer + Sources + Scores
```

## Step 5: Run the Streamlit UI

From the project root:

```bash
uv run streamlit run app/ui/streamlit_app.py
```

The UI will normally be available at:

```text
http://localhost:8501
```

---

# Sample Input and Expected Output

## Sample Input

```text
What is the remote work policy?
```

Another example:

```text
How much can I expense for internet per month?
```

## Expected Output

The assistant retrieves relevant policy information and generates an answer grounded in the available enterprise documents.

For the internet expense example, the retrieved policy documents indicate that internet costs are not reimbursed when employees are responsible for maintaining their own internet connection and the expense reimbursement policy treats such costs as personal expenses.

Example output:

```text
Answer:
The company's policies do not provide reimbursement for internet service.

Sources:
- remote_work_policy.txt
- expense_reimbursement_policy.txt
- it_security_policy.pdf

Evaluation:
Faithfulness: 0.xx
Answer Relevancy: 0.xx
```

The exact generated answer and metric scores can vary between runs because generation and evaluation are LLM-based.

---

# RAG Design

## Document Source

The knowledge base contains enterprise documents such as:

- Remote work policies
- Expense reimbursement policies
- IT security policies

The demonstrated project uses both text documents and PDF content.

## Chunking Strategy

Documents are divided into smaller chunks before they are embedded and stored.

Chunking helps the system avoid sending entire documents to the LLM. Instead, the assistant retrieves only the most relevant pieces of information for a query.

The chunking pipeline enables:

- Efficient semantic search
- Better context selection
- Per-chunk embeddings
- Source-level metadata tracking

The exact chunk size and overlap should follow the ingestion configuration implemented in the project.

## Embedding Model

The project uses **Ollama** for embedding generation.

The flow is:

```text
Document Chunk → Embedding Vector
User Query     → Embedding Vector
```

Semantic similarity is then used to find relevant document chunks.

## Vector Database

The project uses **ChromaDB**.

ChromaDB stores:

- Document chunks
- Embeddings
- Metadata
- Source information

Retrieval flow:

```text
User Query
    ↓
Query Embedding
    ↓
ChromaDB Similarity Search
    ↓
Top Relevant Chunks
```

## Context Merging

The Retriever Agent combines results from:

- ChromaDB
- MCP retrieval

Duplicate sources are handled before the final context is passed to the Response Agent.

Example execution logs:

```text
[Retriever Agent] ChromaDB retrieved 4 chunk(s)
[Retriever Agent] MCP retrieved 2 result(s)
[Retriever Agent] Skipping duplicate MCP source: remote_work_policy.txt
[Retriever Agent] Skipping duplicate MCP source: expense_reimbursement_policy.txt
[Retriever Agent] Total merged chunks: 4
```

---

# LangGraph Design

The workflow is built using a LangGraph `StateGraph`.

The current graph contains three main nodes.

## 1. Retriever Agent

**Node:**

```text
retriever
```

**Responsibilities:**

- Receive the user's query.
- Search ChromaDB.
- Retrieve relevant document chunks.
- Retrieve additional information through MCP.
- Merge retrieval results.
- Remove duplicate sources.
- Store retrieved chunks and source information in the shared graph state.

Example:

```text
[Retriever Agent] Searching for: 'How much can I expense for internet per month?'
[Retriever Agent] ChromaDB retrieved 4 chunk(s)
[Retriever Agent] MCP retrieved 2 result(s)
[Retriever Agent] Total merged chunks: 4
```

## 2. Response Agent

**Node:**

```text
response
```

**Responsibilities:**

- Receive the query and retrieved context.
- Build the context for the LLM.
- Generate a natural-language answer.
- Store the answer in the graph state.

Example:

```text
[Response Agent] Generating answer using 4 chunk(s)
[Response Agent] Answer generated successfully
```

## 3. Evaluator Agent

**Node:**

```text
evaluator
```

**Responsibilities:**

- Receive the query, answer, and retrieved context.
- Run RAGAS evaluation.
- Collect metric scores.
- Generate an evaluation summary.
- Store the evaluation result in the graph state.

Example:

```text
[Evaluator Agent] Starting answer evaluation
[RAGAS] Evaluating metric: faithfulness
[RAGAS] faithfulness score: 0.500
[RAGAS] Evaluating metric: answer_relevancy
[RAGAS] answer_relevancy score: 0.601
[Evaluator Agent] Evaluation completed successfully
```

## Graph Flow

The current workflow is:

```text
START
  ↓
Retriever Agent
  ↓
Response Agent
  ↓
Evaluator Agent
  ↓
END
```

The core graph flow is equivalent to:

```python
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "response")
workflow.add_edge("response", "evaluator")
workflow.add_edge("evaluator", END)
```

This provides a clean baseline where every query goes through retrieval, answer generation, and evaluation.

---

# MCP Integration

The project includes an MCP-based retrieval layer as an additional source of enterprise knowledge.

The Retriever Agent uses MCP retrieval alongside ChromaDB rather than relying on vector search alone.

The implemented use case is:

```text
User Question
    ↓
Retriever Agent
    ├── ChromaDB Retrieval
    └── MCP Retrieval
             ↓
       Merge Results
             ↓
     Remove Duplicates
             ↓
     Relevant Context
             ↓
      Response Agent
```

A demonstrated execution showed:

```text
[Retriever Agent] ChromaDB retrieved 4 chunk(s)
[Retriever Agent] MCP retrieved 2 result(s)
[Retriever Agent] Skipping duplicate MCP source: remote_work_policy.txt
[Retriever Agent] Skipping duplicate MCP source: expense_reimbursement_policy.txt
[Retriever Agent] Total merged chunks: 4
```

This confirms that MCP retrieval is integrated into the end-to-end workflow and that duplicate source content is handled before generation.

> The exact MCP server and tool names should match the implementation inside `app/mcp/`. The behavior above is based on the current project's demonstrated execution logs.

---

# RAGAS Evaluation

The project uses **RAGAS** to evaluate answer quality.

## Metrics Collected

### 1. Faithfulness

Faithfulness checks whether the generated answer is supported by the retrieved context.

A higher score generally indicates that the answer is more grounded in the retrieved documents.

A demonstrated workflow run produced:

```text
Faithfulness: 0.50
```

Another execution shown in the UI produced:

```text
Faithfulness: 1.00
```

### 2. Answer Relevancy

Answer Relevancy measures how relevant the generated answer is to the original user question.

A demonstrated workflow run produced approximately:

```text
Answer Relevancy: 0.601
```

Another UI execution showed approximately:

```text
Answer Relevancy: 0.68
```

## Current Evaluation Structure

The workflow returns a structure similar to:

```text
{
    "faithfulness": ...,
    "answer_relevancy": ...,
    "context_precision": None,
    "context_recall": None
}
```

The currently demonstrated evaluation pipeline actively calculates:

- Faithfulness
- Answer Relevancy

`context_precision` and `context_recall` are present in the result structure but were not populated in the demonstrated run.

## Brief Analysis

One workflow test produced:

```text
Faithfulness: 0.50
Answer Relevancy: 0.601
```

The generated summary classified these as moderate:

```text
Faithfulness: 0.50 (moderate)
Answer Relevancy: 0.60 (moderate)
```

This means the answer was reasonably related to the question and partially supported by the retrieved context, but there is room for improvement.

Potential improvements include:

- Improving retrieval quality.
- Refining chunking strategy.
- Adjusting the number of retrieved chunks.
- Improving answer-generation prompts.
- Adding stronger grounding instructions.
- Adding Context Precision and Context Recall.
- Introducing reranking.
- Adding threshold-based re-retrieval.
- Adding answer refinement when evaluation scores are low.

---

# LangSmith Observability

LangSmith tracing is configured through:

```text
app/observability/tracing.py
```

When tracing is enabled and an API key is available, the project configures variables including:

```text
LANGCHAIN_TRACING_V2
LANGSMITH_TRACING
LANGCHAIN_API_KEY
LANGSMITH_API_KEY
LANGCHAIN_PROJECT
LANGSMITH_PROJECT
```

The project name used for tracing is:

```text
enterprise-knowledge-assistant
```

LangSmith can be used to inspect:

- Workflow runs
- Retriever execution
- Response generation
- LLM calls
- RAGAS evaluation operations
- Inputs and outputs
- Latency
- Token usage
- Errors

A conceptual trace looks like:

```text
Enterprise Knowledge Workflow
├── Retriever
├── Response
│   └── ChatGroq / LLM Call
└── Evaluator
    ├── RAGAS Faithfulness
    └── RAGAS Answer Relevancy
```

---

# Streamlit UI

The Streamlit interface is located at:

```text
app/ui/streamlit_app.py
```

The UI allows users to:

- Ask questions about enterprise knowledge.
- Use suggested example questions.
- View generated answers.
- View retrieved source documents.
- View Faithfulness scores.
- View Answer Relevancy scores.
- Check system status.

The complete UI flow is:

```text
Streamlit UI
    ↓
knowledge_assistant_graph.invoke(...)
    ↓
Retriever Agent
    ↓
Response Agent
    ↓
Evaluator Agent
    ↓
Answer + Sources + Scores
    ↓
Streamlit UI
```

---

# End-to-End Example

```text
USER QUESTION
"How much can I expense for internet per month?"
                    │
                    ▼
             STREAMLIT UI
                    │
                    ▼
           LANGGRAPH WORKFLOW
                    │
          ┌─────────▼─────────┐
          │  RETRIEVER AGENT  │
          └─────────┬─────────┘
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
   ChromaDB                   MCP Server
       │                         │
       └────────────┬────────────┘
                    ▼
            MERGED CONTEXT
                    │
                    ▼
          ┌─────────────────┐
          │ RESPONSE AGENT  │
          │    Groq LLM     │
          └────────┬────────┘
                   ▼
          GENERATED ANSWER
                   │
                   ▼
          ┌─────────────────┐
          │ EVALUATOR AGENT │
          │      RAGAS      │
          └────────┬────────┘
                   ▼
     FAITHFULNESS + RELEVANCY
                   │
                   ▼
          FINAL STREAMLIT UI
```

---

# Future Improvements

Potential next steps include:

- Conversation memory.
- Authentication and user roles.
- Support for additional document formats.
- Improved PDF extraction.
- Hybrid retrieval.
- Metadata filtering.
- Reranking.
- Better chunking strategies.
- Context Precision and Context Recall.
- Conditional LangGraph routing.
- Automatic answer refinement for low scores.
- User feedback collection.
- Persistent conversation history.
- Document upload through the UI.
- Analytics dashboards.
- Evaluation datasets and automated experiments.
- Docker containerization.
- Cloud deployment.

---

# Run Commands

Install dependencies:

```bash
uv sync
```

Ingest documents:

```bash
uv run python -m scripts.ingest
```

Test RAG:

```bash
uv run python -m scripts.rag_test
```

Test the complete workflow:

```bash
uv run python -m scripts.workflow_test
```

Run the Streamlit application:

```bash
uv run streamlit run app/ui/streamlit_app.py
```

---

# Conclusion

The **Enterprise Knowledge Assistant** combines modern AI engineering components into a single end-to-end application.

The system retrieves relevant enterprise knowledge using RAG and MCP, orchestrates the process using LangGraph, generates answers using an LLM, evaluates those answers with RAGAS, traces the complete workflow with LangSmith, and presents the results through Streamlit.

The current implementation provides a strong foundation for future improvements in retrieval quality, evaluation coverage, answer refinement, and production deployment.
