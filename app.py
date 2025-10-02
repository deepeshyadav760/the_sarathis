# app.py

import os
import logging
from rag_chatbot import RAGChatbot
from data_loader import load_and_chunk_directory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

def main():
    """
    Main function to run the command-line chatbot interface.
    """
    print("\n" + "="*70)
    print("  NYD 2026 HACKATHON - KNOWLEDGE BASE CHATBOT")
    print("="*70)
    
    chatbot = None

    while True:
        if chatbot is None:
            # Get directory from user
            dir_path = input("\nEnter data directory path (e.g., ./data): ").strip()
            
            # Validate directory
            if not os.path.isdir(dir_path):
                print(f"\nERROR: Directory '{dir_path}' not found!")
                continue
            
            # Load and process documents
            documents = load_and_chunk_directory(dir_path)
            
            if not documents:
                print(f"\nERROR: No documents loaded from '{dir_path}'")
                print("Make sure the directory contains supported files (.pdf, .csv, .txt, .jsonl)")
                continue

            # Initialize chatbot
            try:
                chatbot = RAGChatbot(documents)
                print("\nReady! Ask questions or type 'exit' to quit, 'new' for new directory")
            except Exception as e:
                print(f"\nERROR: Failed to initialize chatbot: {e}")
                break

        else:
            # Get user question
            question = input("\n> ").strip()
            
            if not question:
                continue
            
            if question.lower() == 'exit':
                print("\nGoodbye!\n")
                break
            
            if question.lower() == 'new':
                print("\nResetting chatbot...\n")
                chatbot = None
                continue
            
            if question.lower() == 'debug':
                # Debug mode: show sources
                answer, sources = chatbot.ask_with_sources(question)
                print(f"\nAnswer: {answer}")
                print(f"\nSources used ({len(sources)} chunks):")
                for i, doc in enumerate(sources[:3], 1):
                    print(f"\n[{i}] {doc.page_content[:200]}...")
                continue

            # Get answer
            answer = chatbot.ask(question)
            print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
