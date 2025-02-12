"""
Created by Analitika at 12/02/2025
contact@analitika.fr

This script implements a Retrieval-Augmented Generation (RAG) chatbot using a local Hugging Face model.
"""

# External imports
import os  # File handling
from langchain_community.document_loaders import TextLoader  # Loads text documents
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # Splits text into chunks
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)  # Converts text into vector embeddings
from langchain_community.vectorstores import (
    FAISS,
)  # Vector database for efficient search
from langchain.chains import RetrievalQA  # Builds a RAG pipeline
from langchain.callbacks import StdOutCallbackHandler  # Enables real-time logging

# Internal imports
from chatbot.llm_local import local_llm
from config.settings import CHATBOT_DATA  # Load chatbot data path

is_test = False
# -------------------------------
# 1. Load Text File
# -------------------------------

# Define the path to the text file (Alice in Wonderland)
text = os.path.join(CHATBOT_DATA, "Alice_in_Wonderland.txt")

# Load the text file
loader = TextLoader(file_path=text, encoding="utf-8")  # Ensure proper encoding

# -------------------------------
# 2. Split Text into Smaller Chunks
# -------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0  # Each chunk has 500 characters  # No overlap
)

# Apply text splitting to the loaded document
data = loader.load_and_split(text_splitter=text_splitter)

# -------------------------------
# 3. Convert Text Chunks into Embeddings
# -------------------------------

# Using a small, efficient Hugging Face embedding model (e5-small-v2)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# -------------------------  TEST EMBEDDINGS --------------------------------->
if is_test:
    print(f"Number of chunks processed: {len(data)}")
    print("Sample chunk:", data[0].page_content)  # Print the first chunk of text
    # Test embedding generation
    test_embedding = embeddings.embed_query("test query")
    print(
        f"Sample embedding vector (length {len(test_embedding)}): {test_embedding[:5]}"
    )  # Print first 5 values
# -------------------------  TEST EMBEDDINGS --------------------------------->


# -------------------------------
# 4. Store Embeddings in FAISS
# -------------------------------

# Create a FAISS index for similarity search
index = FAISS.from_documents(data, embeddings)

if is_test:
    print(f"FAISS index size: {index.index.ntotal}")  # Should not be 0

# -------------------------------
# 5. Configure the Retriever
# -------------------------------

retriever = index.as_retriever()
retriever.search_kwargs["fetch_k"] = 10  # Fetch 20 documents before re-ranking
retriever.search_kwargs["maximal_marginal_relevance"] = True  # Ensures diverse results
retriever.search_kwargs["k"] = 1  # Return top 10 most relevant documents

# ---------------------------------- TEST RETRIEVER ------------------------------------>
if is_test:
    retrieved_docs = retriever.invoke("Should people go back to yesterday?")
    print(f"Retrieved {len(retrieved_docs)} documents.")

    for i, doc in enumerate(retrieved_docs[:3]):  # Print first 3 retrieved docs
        print(f"Document {i+1}: {doc.page_content[:200]}")  # Print first 200 characters
# ---------------------------------- TEST RETRIEVER ------------------------------------>


# -------------------------------
# 6. Set Up the RAG Pipeline
# -------------------------------

# Build the RAG pipeline using the local LLM
chain = RetrievalQA.from_chain_type(
    llm=local_llm,  # Using the local model from llm_local.py
    retriever=retriever,
    verbose=True,  # Enables logging for debugging
)

# -------------------------------
# 9. Enable Real-Time Output Display
# -------------------------------

handler = StdOutCallbackHandler()

# -------------------------------
# 10. Query the Chatbot
# -------------------------------

# First query
query = "Should people go back to yesterday?"
response = chain.invoke({"query": query}, config={"callbacks": [handler]})
print(response["result"])  # Display the chatbot's response

# # Second query (another test question)
# query = "What path should you take if you don't know where you are going?"
# response = chain.invoke(
#     {"query": query},
#     config={"callbacks": [handler]}
# )
# print(response["result"])  # Display the chatbot's response

print("RAG chatbot with a local LLM is running successfully!")
