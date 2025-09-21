# app.py

import os
import logging
from rag_chatbot import RAGChatbot
from data_loader import load_and_chunk_directory
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- SOLUTION FOR VERBOSE LOGS ---
# Get the logger used by the HTTP client and set its level to WARNING.
# This will hide the INFO level messages about requests being made.
logging.getLogger("httpx").setLevel(logging.WARNING)
# --- END SOLUTION ---


def main():
    """
    Main function to run the command-line chatbot interface.
    Now loads an entire directory to build a knowledge base.
    """
    print("="*50)
    print(" Welcome to the NYD 2026 Hackathon Knowledge Bot! ")
    print("="*50)
    
    chatbot = None

    while True:
        if chatbot is None:
            # Step 1: Get the directory from the user
            dir_path = input("\nEnter the path to your data directory (e.g., ./data): ")
            
            # Validate directory path
            if not os.path.isdir(dir_path):
                print("\n[ERROR] Directory not found. Please provide a valid path.")
                continue
            
            # Step 2: Load and process all files in the directory
            documents = load_and_chunk_directory(dir_path)
            
            if not documents:
                print(f"\n[WARNING] No processable documents found in '{dir_path}'.")
                continue

            # Step 3: Initialize the RAG chatbot with the unified documents
            try:
                chatbot = RAGChatbot(documents)
                print("\n✅ Knowledge Base is ready! Ask me anything about your documents.")
                print("   (Type 'exit' to quit or 'new' to load a new directory)")
            except Exception as e:
                print(f"\n[ERROR] Failed to initialize chatbot: {e}")
                print("Please check your API keys and dependencies.")
                break

        else:
            # Step 4: Start the conversation
            question = input("\n> ")
            
            if question.lower() == 'exit':
                print("\nGoodbye!")
                break
            
            if question.lower() == 'new':
                print("\nResetting to load a new knowledge base...")
                chatbot = None
                continue

            # Get the answer from the chatbot
            answer = chatbot.ask(question)
            print(f"\n🤖 Answer: {answer}")


if __name__ == "__main__":
    main()