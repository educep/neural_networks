"""
Created by Analitika on 18/02/2025
contact@analitika.fr

This script implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain.
It combines the best features from two implementations:
1. **Structured prompt handling** (from `coding_prompt_rag.py`)
2. **Advanced document processing** with file loading, chunking, and FAISS storage (from `chatbot_rag_openai.py`)
3. **Retriever tuning** for better search relevance
4. **Logging and debugging support**

It loads a text file, splits it into smaller chunks, converts it into vector embeddings,
stores it in a FAISS vector database, and queries the database using an LLM with a structured prompt.
"""


# External imports
import os
import sys
from loguru import logger
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StdOutCallbackHandler

# Internal imports
from config import (
    OPENAI_API_KEY,
    OPENAI_COMPLETIONS_MODEL,
    OPENAI_EMBEDDING_MODEL,
    CHATBOT_DATA,
)

# Set up data file path
DATA_FILE = os.path.join(CHATBOT_DATA, "Alice_in_Wonderland.txt")


class RAGChatbot:
    def __init__(self, data_file: str):
        """Initialize the RAG chatbot with OpenAI embeddings, FAISS retriever, and a structured prompt."""
        logger.info("Initializing RAG Chatbot")

        try:
            self.embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY
            )
            self.llm = ChatOpenAI(
                model=OPENAI_COMPLETIONS_MODEL, openai_api_key=OPENAI_API_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM or embeddings: {e}")
            sys.exit(1)

        self.vectorstore = self.create_vectorstore(data_file)
        self.retriever = self.configure_retriever()
        self.prompt_template = self.create_prompt_template()

    def create_vectorstore(self, data_file: str):
        """Loads text from a file, splits it into chunks, and stores it in a FAISS vector store."""
        logger.info(f"Loading text from {data_file}")

        try:
            loader = TextLoader(file_path=data_file, encoding="utf-8")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            documents = loader.load_and_split(text_splitter=text_splitter)
        except Exception as e:
            logger.error(f"Error loading or splitting documents: {e}")
            raise

        logger.info("Creating FAISS vector store")
        try:
            vectorstore = FAISS.from_documents(documents, embedding=self.embeddings)
        except Exception as e:
            logger.error(f"Error creating FAISS vector store: {e}")
            raise
        return vectorstore

    def configure_retriever(self):
        """Configures the retriever with optimized search parameters."""
        retriever = self.vectorstore.as_retriever()
        # Update search parameters: fetch more documents then narrow down to the top-k after re-ranking.
        retriever.search_kwargs.update(
            {
                "fetch_k": 20,  # Fetch 20 documents before re-ranking
                "maximal_marginal_relevance": True,  # Ensure diverse results
                "k": 10,  # Return top 10 most relevant documents
            }
        )
        logger.info(
            "Retriever configured with search parameters: %s", retriever.search_kwargs
        )
        return retriever

    def create_prompt_template(self) -> ChatPromptTemplate:
        """Creates a structured prompt template for the RAG chain."""
        logger.info("Creating structured prompt template")
        template = (
            "Answer the question based only on the following context:\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Answer in a comprehensive manner, providing relevant details from the context."
        )
        try:
            prompt = ChatPromptTemplate.from_template(template)
        except Exception as e:
            logger.error(f"Error creating prompt template: {e}")
            raise
        return prompt

    def generate_answer(self, question: str) -> str:
        """
        Retrieves relevant context using the retriever, constructs the prompt,
        and generates an answer using the LLM.
        """
        logger.info(f"Processing question: {question}")
        try:
            # Retrieve context using the retriever. Assume `invoke` returns text context.
            context = self.retriever.invoke(question)
            logger.debug("Retrieved context: {}", context)

            # Prepare the prompt with the retrieved context and question
            prompt_input = {"context": context, "question": question}
            formatted_prompt = self.prompt_template.format(**prompt_input)
            logger.debug("Formatted prompt: {}", formatted_prompt)

            # Call the language model with the formatted prompt.
            # Optionally, add callbacks or additional parameters if needed.
            raw_response = self.llm.invoke(formatted_prompt)
            logger.debug("LLM raw response: {}", raw_response)

            # Parse the raw response into a clean string answer
            answer = StrOutputParser().parse(raw_response)
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            answer = "Sorry, an error occurred while generating the answer."
        return answer


def main():
    logger.info("Starting RAG Chatbot application")
    chatbot = RAGChatbot(DATA_FILE)

    # Example questions to test the chatbot
    questions = [
        # "What is the story of Alice in Wonderland about?",
        "Who is the Mad Hatter?",
        "What advice does the Cheshire Cat give to Alice?",
        "What path should you take if you don't know where you are going?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        answer = chatbot.generate_answer(question)
        print(f"A{i}: {answer}")


if __name__ == "__main__":
    main()
