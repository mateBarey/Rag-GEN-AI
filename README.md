# Rag-GEN-AI

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Weaviate](https://img.shields.io/badge/Weaviate-1.15.2-00D1A0?style=for-the-badge&logo=weaviate&logoColor=white)
![DSPy](https://img.shields.io/badge/DSPy-Framework-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF_Parser-CC0000?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A generalized Retrieval-Augmented Generation (RAG) framework that integrates multiple PDF knowledge sources with local LLMs via Weaviate vector search and DSPy chain-of-thought reasoning.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Sources](#sources)
- [License](#license)

---

## Overview

**Rag-GEN-AI** is an abstract, modular class designed to integrate multiple information sources — primarily large PDF datasets — into the knowledge base of a local Large Language Model (LLM). It implements a full RAG pipeline leveraging:

- **Weaviate** for semantic vector search and document storage
- **DSPy** for structured chain-of-thought prompt engineering
- **Ollama** for running local LLM inference (e.g., `dolphin-llama3`)
- **PyMuPDF** for PDF text extraction
- **TextBlob** for response quality evaluation via sentiment analysis

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  GeneralizedRAG                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  WeaviateRM (Retriever)                           │  │
│  │  - Semantic near-text search                      │  │
│  │  - Top-K document chunk retrieval                 │  │
│  └───────────────────┬───────────────────────────────┘  │
│                      │                                  │
│                      ▼                                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  CustomRAG (DSPy Module)                          │  │
│  │  - ChainOfThought reasoning                       │  │
│  │  - Technical detail enrichment                    │  │
│  │  - Self-reflection & quality scoring              │  │
│  │  - Keyword matching + sentiment analysis          │  │
│  └───────────────────┬───────────────────────────────┘  │
│                      │                                  │
│                      ▼                                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Response (with relevance-adjusted output)        │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│               Infrastructure Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │  Weaviate   │  │ Contextionary│  │  Ollama (LLM) │  │
│  │  (Docker)   │  │  (Docker)   │  │  (Local)       │  │
│  └─────────────┘  └─────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Features

- **Multi-PDF Ingestion** — Process and index multiple PDF documents in parallel using `concurrent.futures`
- **Semantic Vector Search** — Leverage Weaviate's `text2vec-contextionary` for contextually relevant document retrieval
- **Chain-of-Thought Reasoning** — DSPy-powered structured prompting for detailed, explainable answers
- **Self-Reflective Quality Control** — Automatic response evaluation using keyword matching and sentiment analysis with configurable thresholds
- **Fully Local Pipeline** — Run everything on your own hardware with Ollama — no API keys or cloud dependencies
- **Modular Architecture** — Easily swap models, retrieval backends, or evaluation strategies
- **Docker-Composed Infrastructure** — One-command setup for Weaviate + Contextionary

---

## Prerequisites

| Tool       | Version    | Purpose                          |
|------------|------------|----------------------------------|
| Python     | >= 3.10    | Runtime                          |
| Docker     | >= 20.10   | Container runtime                |
| Docker Compose | >= 1.29 | Service orchestration           |
| Ollama     | latest     | Local LLM inference              |
| Git        | >= 2.0     | Version control                  |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Rag-GEN-AI.git
cd Rag-GEN-AI
```

### 2. Start Weaviate Services

```bash
docker-compose up -d
```

This launches:
- **Weaviate** on `http://localhost:8080` (REST) and `localhost:50051` (gRPC)
- **Contextionary** on `localhost:9999` for text vectorization

### 3. Install Python Dependencies

```bash
pip install weaviate-client pymupdf dspy-ai textblob ollama
```

Or with a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install weaviate-client pymupdf dspy-ai textblob ollama
```

### 4. Pull the LLM Model via Ollama

```bash
ollama pull dolphin-llama3
```

### 5. Verify Weaviate is Running

```bash
curl http://localhost:8080/v1/.well-known/ready
```

---

## Usage

### Single PDF Source

```python
from generalizedRagAgg import GeneralizedRAG

pdf_paths = ["path/to/tax_code.pdf"]

rag = GeneralizedRAG(
    model_name="TaxModel",
    model_input="dolphin-llama3",
    pdf_source_files=pdf_paths
)

answer = rag.ask_question("What are the tax implications of business expenses?")
print(answer)
```

### Multiple PDF Sources (Aggregated Knowledge Base)

```python
from generalizedRagAgg import GeneralizedRAG

pdf_paths = [
    "path/to/texas_business_law.pdf",
    "path/to/blacks_law_dictionary.pdf",
    "path/to/real_estate_law.pdf",
    "path/to/irs_code.pdf",
    "path/to/tax_liens_investing.pdf"
]

rag = GeneralizedRAG(
    model_name="AggregateModel",
    model_input="dolphin-llama3",
    pdf_source_files=pdf_paths
)

answer = rag.ask_question(
    "What is the tax strategy for a single-member LLC taxed as an S-Corp "
    "with gross income of $145k, expenses of $35k, and salary of $30k?"
)
print(answer)
```

### Interactive Notebook

For a step-by-step walkthrough, open the Jupyter notebook:

```bash
jupyter notebook RagModelsetupTax.ipynb
```

---

## Project Structure

```
Rag-GEN-AI/
├── generalizedRagAgg.py       # Core RAG pipeline (GeneralizedRAG + WeaviateRM)
├── generalizedRagtester.py    # Chunk presence verification tool
├── RagModelsetupTax.ipynb     # Interactive notebook walkthrough
├── docker-compose.yaml        # Weaviate + Contextionary services
├── usc26@118-64.pdf           # Sample dataset (U.S. Tax Code Title 26)
└── README.md                  # This file
```

---

## Configuration

### Weaviate Connection

```python
# Default: localhost
connection_params = {
    "url": "http://localhost:8080",
}
```

### Chunking Strategy

```python
# Adjust chunk size (default: 1000 characters)
chunks = self._chunk_text(text, chunk_size=1000)
```

### Retrieval Parameters

```python
# Adjust top-K results (default: 10 stored, 5 used per query)
retriever_model = WeaviateRM(model_name, weaviate_client, k=10)
context_results = self.retriever_model.retrieve(question, top_k=5)
```

### Quality Threshold

```python
# Adjust the relevance score threshold for self-reflection (default: 0.75)
if relevance_score < 0.75:
    answer = self.improve_response(answer, context, question)
```

### Evaluation Weights

```python
# Adjust keyword vs. sentiment scoring weights
overall_score = 0.6 * keyword_score + 0.4 * sentiment_score
```

---

## Testing

Use the included `RAGModelTester` to verify document chunks were properly indexed:

```python
from generalizedRagtester import RAGModelTester
import weaviate

client = weaviate.Client(url="http://localhost:8080", timeout_config=(30, 30))

tester = RAGModelTester(client, "AggregateModel")
results = tester.test_chunk_presence([
    "Sample text from document 1",
    "Sample text from document 2",
])

for chunk, present in results.items():
    print(f"Chunk: '{chunk}' — Indexed: {present}")
```

---

## API Reference

### `GeneralizedRAG`

| Method | Description |
|--------|-------------|
| `__init__(model_name, model_input, pdf_source_files)` | Initialize the RAG pipeline with Weaviate schema, PDF ingestion, and DSPy configuration |
| `ask_question(question: str)` | Query the RAG system and receive a chain-of-thought answer with self-reflection |

### `WeaviateRM`

| Method | Description |
|--------|-------------|
| `__init__(class_name, weaviate_client, k)` | Initialize the retrieval model targeting a Weaviate class |
| `retrieve(query, top_k)` | Perform semantic near-text search and return top-K document chunks |

### `RAGModelTester`

| Method | Description |
|--------|-------------|
| `__init__(weaviate_client, model_name)` | Initialize the tester for a specific Weaviate class |
| `test_chunk_presence(test_chunks)` | Verify whether given text chunks exist in the Weaviate index |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Sources

- [Weaviate DSPy + Llama3 Integration Recipe](https://github.com/weaviate/recipes/blob/main/integrations/llm-frameworks/dspy/llms/Llama3.ipynb)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Ollama Documentation](https://ollama.ai/)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
