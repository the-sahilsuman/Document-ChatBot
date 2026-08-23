# 📄 Document Chatbot — RAG Application

A document-based AI chatbot built using **Retrieval-Augmented Generation (RAG)**.

The application allows users to provide PDF documents and ask questions about their content. Instead of directly asking the LLM to answer the question from its own knowledge, the application first searches the documents for relevant information and then provides that information to the LLM as context.

This project is built to understand the complete RAG pipeline from:

**Document → Chunking → Embedding → Vector Database → Retrieval → Context → Prompt → LLM → Response**

---

# 🚀 Project Overview

📁 Project Structure

Chatbot-RAG-App/
│
├── chatbot.py
|
|── inputs/
│   └── *.pdf
│
├── models/
│   ├── chains.py
│   ├── chat_model.py
│   ├── memory.py
│   ├── parsing.py
│   └── prompt.py
│
├── rag/
│   │
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embedding_model.py
│   ├── huggingface_model.py
│   ├── embedding_vector.py
│   ├── vector_store.py
│   ├── retrieval.py
│   └── create_context.py
│
├── chroma_db/
│
├── .env
├── .gitignore
└── requirements.txt

This project implements a basic but complete **Retrieval-Augmented Generation system**.

The user provides PDF documents.

The system:

1. Loads the PDF documents.
2. Extracts the text.
3. Splits the text into smaller chunks.
4. Converts each chunk into an embedding vector.
5. Stores the vectors inside ChromaDB.
6. Converts the user's question into an embedding.
7. Performs semantic similarity search.
8. Retrieves the most relevant chunks.
9. Creates a context from those chunks.
10. Adds the context to the user's question.
11. Sends the augmented prompt to the LLM.
12. Parses the LLM response.
13. Returns the final answer to the user.

---

# 🎯 What Problem Does This Project Solve?

A normal LLM has a major limitation.

Suppose we have a private PDF:

```text
my_resume.pdf

## 🏗️ RAG Architecture

```text

                RAG
                 │
       ┌─────────┴─────────┐
       │                   │
   Indexing              Querying
       │                   │
       ▼                   ▼
    Load PDF           User Query
       │                   │
       ▼                   ▼
    Chunking           Embedding
       │                   │
       ▼                   ▼
    Embedding         Similarity Search
       │                   │
       ▼                   ▼
   ChromaDB            Top-K Chunks
                           │
                           ▼
                        Context
                           │
                           ▼
                         Prompt
                           │
                           ▼
                          LLM
                           │
                           ▼
                        Answer


## 👨‍💻 Author

### Sahil Suman
 
Cloud & DevOps | Generative AI | Backend Development

This project was designed and developed by **Sahil Suman** as a hands-on implementation of a production-oriented **Retrieval-Augmented Generation (RAG)** system using Python, LangChain, ChromaDB, Hugging Face, and LLM-based question answering.

---

⭐ **If you found this project useful, please consider giving it a star!**