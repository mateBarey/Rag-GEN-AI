# Rag-GEN-AI

An abstract class designed to integrate multiple information sources, enhancing the knowledge base of a Large Language Model (LLM).

Implemented a Retrieval-Augmented Generation (RAG) pipeline leveraging Weaviate for semantic search, enabling precise and contextually relevant information retrieval from large PDF datasets for enhanced model outputs.
Utilized Weaviate’s vectorized search capabilities to efficiently store, index, and retrieve document chunks based on contextual similarity, optimizing data access for the RAG system.
Integrated Dspy's OllamaLocal model within the RAG framework to generate accurate and context-driven answers, enhancing the AI’s ability to respond to complex queries with technically detailed explanations.
Engineered parallelized PDF processing pipelines with concurrent futures to load and chunk documents into Weaviate, significantly reducing data preprocessing time and improving overall system efficiency.



Please see the ipynb for the general workflow
sources:
https://github.com/weaviate/recipes/blob/main/integrations/llm-frameworks/dspy/llms/Llama3.ipynb
