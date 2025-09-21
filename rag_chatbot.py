# rag_chatbot.py

import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings # This is for local embeddings, so it stays
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ** THE FIX: Import the new Groq chat model **
from langchain_groq import ChatGroq

class RAGChatbot:
    """
    The RAG Chatbot engine, now powered by Groq and Llama 3.
    """
    def __init__(self, documents):
        load_dotenv()
        # ** THE FIX: Check for the new Groq API key **
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq API key not found. Please set it in a .env file.")
        
        self.retriever = self._create_retriever(documents)
        self.retrieval_chain = self._create_chain()

    def _create_retriever(self, documents):
        print("Creating embeddings and vector store...")
        # We still use this for creating the vectors locally. It's fast and free.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embeddings)
        print("Vector store created successfully.")
        return vector_store.as_retriever(search_kwargs={"k": 3})

    def _create_chain(self):
        print("Setting up the RAG chain with Groq...")
        
        # ** THE FIX: Instantiate the ChatGroq model **
        # Note: The model name is slightly different. Groq uses 'llama3-70b-8192'
        llm = ChatGroq(
            temperature=0.7, 
            model_name="llama-3.3-70b-versatile"
        )

        prompt_template = """
        Use only the following retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        
        print("RAG chain is ready.")
        return retrieval_chain

    def ask(self, question: str):
        if not self.retrieval_chain:
            return "The chatbot is not initialized. Please load a dataset first."
        
        response = self.retrieval_chain.invoke({'input': question})
        return response['answer']