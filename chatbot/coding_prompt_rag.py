"""
Created by Analitika at 18/02/2025
contact@analitika.fr
"""
# External imports
import os
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger

# Internal imports
from config import OPENAI_API_KEY, OPENAI_COMPLETIONS_MODEL, OPENAI_EMBEDDING_MODEL


class RAGSystem:
    def __init__(self):
        logging.info("Initializing RAG System")
        self.embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        self.llm = ChatOpenAI(
            model=OPENAI_COMPLETIONS_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        self.vectorstore = self.create_vectorstore()
        self.retriever = self.vectorstore.as_retriever()
        self.rag_chain = self.create_rag_chain()

    def create_vectorstore(self):
        logging.info("Creating vector store")
        texts = [
            "ChatGPT 4o, released in 2024, is the latest version of OpenAI's language model.",
            "It features enhanced multilingual capabilities and improved context understanding.",
            "The model can process and generate text in over 100 languages with near-native fluency.",
            "ChatGPT 4o has shown remarkable improvements in logical reasoning and problem-solving tasks.",
            "Ethical considerations and bias mitigation were key focus areas in its development.",
        ]
        return FAISS.from_texts(texts, embedding=self.embeddings)

    def create_rag_chain(self):
        logging.info("Creating RAG chain")
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}

        Answer in a comprehensive manner, providing relevant details from the context.
        """
        prompt = ChatPromptTemplate.from_template(template)

        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def answer_question(self, question):
        logging.info(f"Answering question: {question}")
        return self.rag_chain.invoke(question)  # Ensure proper input formatting


def main():
    rag_system = RAGSystem()

    questions = [
        "What is the latest version of ChatGPT and when was it released?",
        "What are some key features of ChatGPT 4o?",
        "How does ChatGPT 4o handle different languages?",
        "What ethical aspects were considered in developing ChatGPT 4o?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        answer = rag_system.answer_question(question)
        print(f"A{i}: {answer}")


if __name__ == "__main__":
    main()
