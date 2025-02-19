### **Objective: RAG-Powered Code Assistant for Python Projects**

We aim to **enhance code generation, debugging, and refactoring** using a **Retrieval-Augmented Generation (RAG) system** tailored for Python projects. The system will **extract dependencies**, **compute embeddings**, and **enable efficient code retrieval** for AI-powered assistance.

---

## **🔹 Expected Results**
### **1️⃣ Extract Dependencies & Build a Call Graph**
- **Analyze Python source code** to extract:
  - **Functions**
  - **Classes**
  - **Imports**
  - **Function Calls** (which function calls which)
- **Generate a dependency graph** using `networkx`, showing:
  - Function-to-function relationships
  - Class dependencies
  - Module import connections

✅ **Deliverables:**
- A **graph structure** where nodes represent functions, classes, and imports.
- Edges represent **calls, inheritance, or module dependencies**.
- A **saved dependency graph** (`dependencies_graph.gpickle`) for later use.

---

### **2️⃣ Compute Code Embeddings (CodeBERT)**
- **Extract relevant code elements**:
  - Function definitions
  - Class definitions
  - Docstrings + comments (for additional context)
- **Generate embeddings using `CodeBERT`**, a model trained for source code understanding.
- **Store embeddings efficiently** in a vector database (FAISS).

✅ **Deliverables:**
- **Embeddings for each function/class** (numpy vectors).
- **Metadata for each code element**, including:
  - File name
  - Function/class name
  - Dependencies (linked to the call graph)
- A **searchable FAISS index** to retrieve similar code snippets.

---

### **3️⃣ Code Retrieval & AI-Assisted Development**
- **Search for similar functions/classes** based on:
  - A query (e.g., “How do I handle API requests?”)
  - Code structure similarities (via FAISS)
- **Use retrieval-augmented generation (RAG)** to improve:
  - **Code completion** (suggesting relevant functions)
  - **Debugging** (finding related issues)
  - **Refactoring** (optimizing function structures)

✅ **Deliverables:**
- **Search API to retrieve relevant code snippets** (from FAISS).
- **AI-powered responses grounded in the project’s real codebase**.
- **Refactoring suggestions** based on dependency analysis.

---

## **🔹 Technical Approach**
### **🔹 Step 1: Extract Dependencies**
- **Use `ast` to parse Python files** and extract:
  - Function/class definitions
  - Import statements
  - Function calls
- **Build a directed graph (`networkx`)** where:
  - Nodes = functions, classes, imports
  - Edges = function calls, class inheritance, module dependencies

### **🔹 Step 2: Compute Embeddings**
- **Extract meaningful code snippets** (functions, classes, modules).
- **Compute embeddings using `CodeBERT`** (`microsoft/codebert-base`).
- **Store embeddings in FAISS** along with metadata.

### **🔹 Step 3: Enable Code Retrieval**
- **Use FAISS to perform similarity searches**.
- **Retrieve relevant functions** based on:
  - Code structure
  - Semantic meaning (via embeddings)
  - Dependencies (from the graph)
- **Enhance AI responses with project-specific knowledge**.

---

## **🔹 Final Outcomes**
1. **A dependency graph (`networkx`)** representing function and module relationships.
2. **A FAISS vector database** storing CodeBERT embeddings for each function/class.
3. **A retrieval system** enabling **smart code search, debugging, and AI-powered refactoring**.

🚀 **This will allow AI to generate, debug, and refactor code while maintaining project integrity and consistency.**

Below is a sample README section that documents the RAG chatbot integration using our project’s dependency graph and CodeBERT embeddings. You can add this to your existing documentation:

---

# RAG Chatbot for Project Code Retrieval

## Overview

This feature extends our RAG-Powered Code Assistant by integrating a Retrieval-Augmented Generation (RAG) chatbot. It leverages the project's dependency graph and CodeBERT embeddings—previously extracted and stored in a FAISS vector store—to retrieve real code elements (functions, classes, etc.) and provide context-aware answers using ChatGPT via LangChain.

## Key Features

- **Real Project Context:**
  Uses a dependency graph to extract genuine code elements from your project. These elements are embedded using CodeBERT and stored in a FAISS index.

- **Pre-Generated FAISS Vector Store:**
  Instead of rebuilding the vector base each time, the chatbot loads a pre-generated FAISS index (and associated metadata) from a specified path.

- **RAG Integration with ChatGPT:**
  Combines the retrieved code context with a structured prompt to guide ChatGPT (via LangChain) in answering questions about your project's code.

- **Customizable Retrieval & Prompt:**
  Uses LangChain’s retriever configuration for enhanced search relevance and a structured prompt template that strictly bases answers on the retrieved code snippets.

## Architecture

- **Dependency Extraction & Graph Creation:**
  Uses AST parsing to analyze the project’s Python files, extract functions, classes, and their relationships, and build a dependency graph (see `directed_graph.py`).

- **Code Embedding & FAISS Storage:**
  Code elements are converted into embeddings using CodeBERT and stored alongside metadata in a FAISS vector store (see `code_embeddings.py`).

- **RAG Chatbot:**
  The chatbot (`rag_chatbot.py`) loads the FAISS index from disk, retrieves relevant code snippets based on user queries, formats a structured prompt, and uses ChatGPT (via LangChain) to generate context-aware answers.

## Setup and Configuration

1. **Prerequisites:**
   - Python 3.8+
   - Required packages: `transformers`, `torch`, `faiss-cpu`, `langchain`, `loguru`, etc. (see `requirements.txt`)
   - A valid OpenAI API key for both ChatGPT and embedding models.

2. **Configuration:**
   - Update the `config.py` file with your settings:
     ```python
     OPENAI_API_KEY = "your-openai-api-key"
     OPENAI_COMPLETIONS_MODEL = "gpt-3.5-turbo"  # or your preferred model
     OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"  # or another compatible model
     VECTOR_DB_PATH = "path/to/code_embeddings.index"
     CHATBOT_DATA = "data"  # Optional: any additional data paths
     ```
   - Ensure the FAISS index (e.g., `code_embeddings.index`) generated from your project is available at the specified location.

## Usage

To run the RAG Chatbot, execute the `rag_chatbot.py` script. You can optionally specify the FAISS vector store path via a command-line argument:

```bash
python rag_chatbot.py /path/to/code_embeddings.index
```

### Example Queries

The chatbot is designed to answer questions related to your project’s codebase. For example:
- "How is the dependency graph constructed in this project?"
- "What function is responsible for extracting code embeddings?"
- "How does the RAG system integrate with ChatGPT?"

## Troubleshooting

- **Vector Store Loading Errors:**
  Confirm that the FAISS index file exists at the specified path in `config.py` or via the command-line argument.

- **API Key Issues:**
  Verify your OpenAI API key and ensure that your account has the necessary permissions.

- **Low Retrieval Quality:**
  If the retrieved context seems insufficient, review your dependency extraction and code embedding processes for completeness.

## Summary

This RAG chatbot integration enables ChatGPT to answer queries using real, project-specific code context, significantly improving the relevance and accuracy of responses for tasks such as code generation, debugging, and refactoring.
