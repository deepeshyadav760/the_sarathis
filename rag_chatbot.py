# rag_chatbot.py

import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

class RAGChatbot:
    """
    RAG Chatbot powered by Groq and Llama 3.
    """
    def __init__(self, documents):
        print("\n" + "="*70)
        print("INITIALIZING RAG CHATBOT")
        print("="*70)
        
        load_dotenv()
        
        # Check for Groq API key
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq API key not found. Set GROQ_API_KEY in .env file")
        
        print("\n[1/3] Creating vector store...")
        self.retriever = self._create_retriever(documents)
        
        print("\n[2/3] Initializing Groq LLM...")
        self.llm = self._create_llm()
        
        print("\n[3/3] Building RAG chain...")
        self.retrieval_chain = self._create_chain()
        
        print("\n" + "="*70)
        print("CHATBOT READY!")
        print("="*70 + "\n")

    def _create_retriever(self, documents):
        """Create FAISS vector store with embeddings."""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = FAISS.from_documents(documents, embeddings)
        print(f"   -> Vector store created with {len(documents)} chunks")
        
        # Retrieve more documents for better context
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
        )

    def _create_llm(self):
        """Initialize Groq LLM."""
        llm = ChatGroq(
            temperature=0.3,  # Lower temperature for more factual responses
            model_name="llama-3.3-70b-versatile",
            max_tokens=1024
        )
        print(f"   -> Using model: llama-3.3-70b-versatile")
        return llm

    def _create_chain(self):
        """Create the RAG retrieval chain."""
        
        # Improved prompt for better responses
        prompt_template = """You are a helpful AI assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {input}

Instructions:
- If the answer is in the context, provide a clear and detailed response
- If the information is not in the context, say "I don't have enough information in the provided documents to answer this question."
- Be specific and cite relevant details from the context
- Do not make up information

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        
        print(f"   -> RAG chain configured")
        return retrieval_chain

    def ask(self, question: str):
        """Ask a question and get an answer."""
        if not self.retrieval_chain:
            return "Chatbot not initialized properly."
        
        try:
            response = self.retrieval_chain.invoke({'input': question})
            return response['answer']
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def ask_with_sources(self, question: str):
        """Ask a question and get answer with source documents."""
        if not self.retrieval_chain:
            return "Chatbot not initialized properly.", []
        
        try:
            response = self.retrieval_chain.invoke({'input': question})
            return response['answer'], response.get('context', [])
        except Exception as e:
            return f"Error: {str(e)}", []
