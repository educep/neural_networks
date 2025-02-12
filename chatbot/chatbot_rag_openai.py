"""
Created by Analitika on 12/02/2025
contact@analitika.fr

This script implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain.
It loads a text file, splits it into smaller chunks, converts it into vector embeddings,
stores it in a FAISS vector database, and then queries the database using an LLM.
"""

# External imports
import os  # Handles file paths and environment variables
import langchain  # Main framework for building LLM-based applications

# LangChain modules for document processing, embeddings, and vector storage
from langchain_community.document_loaders import TextLoader  # Loads text documents
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # Splits text into chunks
from langchain_openai import OpenAIEmbeddings  # Converts text into vector embeddings
from langchain_community.vectorstores import (
    FAISS,
)  # Vector database for efficient search

# LangChain modules for question-answering and LLM integration
from langchain.chains import RetrievalQA  # Constructs a RAG pipeline
from langchain_openai import ChatOpenAI  # OpenAI's LLM integration

# Callback handler to display intermediate results
from langchain.callbacks import StdOutCallbackHandler

# Internal imports
# Importing OpenAI API key and chatbot data path from project settings
from config.settings import OPENAI_API_KEY, CHATBOT_DATA

print(f"LangChain version: {langchain.__version__}")

# -------------------------------
# 1. Load text file
# -------------------------------

# Define the path to the text file (Alice in Wonderland)
text = os.path.join(CHATBOT_DATA, "Alice_in_Wonderland.txt")

# Load the text file using LangChain's TextLoader
loader = TextLoader(file_path=text, encoding="utf-8")

# -------------------------------
# 2. Split text into smaller chunks
# -------------------------------

# This helps improve retrieval efficiency and accuracy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Each chunk will have 500 characters
    chunk_overlap=0,  # No overlap between consecutive chunks
)

# Apply text splitting to the loaded document
data = loader.load_and_split(text_splitter=text_splitter)

# Display a small portion of the data (for debugging)
data[2:5]

# -------------------------------
# 3. Convert text chunks into embeddings
# -------------------------------

# OpenAI Embeddings model converts text into numerical vectors
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,  # API key for OpenAI access
    show_progress_bar=True,  # Displays progress while embedding
)

# -------------------------------
# 4. Store embeddings in FAISS vector database
# -------------------------------

# FAISS (Facebook AI Similarity Search) stores and retrieves similar embeddings efficiently
index = FAISS.from_documents(data, embeddings)

# -------------------------------
# 5. Perform a similarity search
# -------------------------------

# Define a query for the chatbot
query = "Should people go back to yesterday?"

# Perform a search in the FAISS index for similar text chunks
index.similarity_search_with_relevance_scores(query)

# -------------------------------
# 6. Configure the retriever (tuning search parameters)
# -------------------------------

retriever = index.as_retriever()

# Adjust search parameters for better retrieval:
retriever.search_kwargs["fetch_k"] = 20  # Fetch 20 documents before re-ranking
retriever.search_kwargs["maximal_marginal_relevance"] = True  # Ensures diverse results
retriever.search_kwargs["k"] = 10  # Return top 10 most relevant documents

# -------------------------------
# 7. Set up the LLM for answering questions
# -------------------------------

# Initialize the OpenAI language model (GPT-based)
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# 8. Build the RAG pipeline
# -------------------------------

# RetrievalQA combines the retriever (search) with the LLM (answer generation)
chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, verbose=True  # Enables logging for debugging
)

# -------------------------------
# 9. Enable real-time output display
# -------------------------------

# The callback handler prints intermediate results to the console
handler = StdOutCallbackHandler()

# -------------------------------
# 12. Query the chatbot
# -------------------------------

# First query
response = chain.invoke(
    {"query": query}, config={"callbacks": [handler]}  # Enables real-time output
)
print(response["result"])  # Display the chatbot's response

# Second query (another test question)
query = "What path should you take if you don't know where you are going?"
response = chain.invoke({"query": query}, config={"callbacks": [handler]})
print(response["result"])  # Display the chatbot's response
