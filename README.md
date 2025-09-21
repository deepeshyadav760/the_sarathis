# The Gita Chatbot

The core deliverable for this stage was to create a working chatbot that can load any new dataset without requiring code changes.

We have successfully built a powerful, multi-document **Knowledge Bot** that exceeds this requirement. Our solution can ingest an entire directory of mixed-format documents (CSVs, PDFs, TXTs) to create a unified, conversational knowledge base. It is built on a robust Retrieval-Augmented Generation (RAG) architecture, powered by the state-of-the-art Llama 3 model via the high-speed Groq API.

## ✅ Key Features

-   **Dynamic Knowledge Base Creation**: Instead of a single file, the bot accepts a directory path and automatically processes all documents within it.
-   **Multi-Format Document Support**: Seamlessly handles various file types, including `.csv`, `.pdf`, and `.txt`, using stable and reliable document loaders.
-   **Robust RAG Architecture**: Grounds responses in the provided documents, minimizing hallucinations and setting the stage for the Month 2 fact-checking engine.
-   **State-of-the-Art LLM**: Utilizes the powerful Llama 3 (70b) model for generation via the blazingly fast Groq API.
-   **Interactive Command-Line Interface**: A user-friendly CLI allows for easy interaction, loading new knowledge bases, and asking questions.
-   **Modular and Extensible Code**: The project is logically structured into separate modules for data loading, chatbot logic, and the main application, making it easy to maintain and build upon.

## 🛠️ Technology Stack

-   **Backend**: Python 3.9+
-   **Core Framework**: LangChain
-   **LLM Provider**: Groq API (Llama 3 70b)
-   **Embeddings**: `sentence-transformers` (running locally)
-   **Vector Store**: FAISS (in-memory)
-   **Document Loading**: `langchain-community`, `pypdf`

## 📂 Project Structure

The codebase is organized into logical modules for clarity and maintainability:

```/
|-- /data/                  # Folder to place your source documents
|   |-- cleaned_bhagavad_gita.csv
|   |-- ... any other documents (.pdf, .txt, .csv etc)
|
|-- app.py                  # Main entry point, runs the interactive CLI
|-- rag_chatbot.py          # Contains the core RAG logic and chatbot class
|-- data_loader.py          # Handles loading & chunking of all documents from a directory
|-- requirements.txt        # Lists all necessary Python libraries
|-- .env                    # For securely storing your API key -(store the groq API key)

for writing the groq api in the .env file
GROQ_API_KEY="write your api key"

```

### Steps to Create and Setup the groq API:

The bot requires a free Groq API key to function.

1. Create a free account at [Groq Console](https://console.groq.com/).
2. Navigate to the **API Keys** section and create a new key.
3. In the project's root directory, create a file named `.env`.
4. Add your API key to the `.env` file in the following format:

```env
GROQ_API_KEY="gsk_YourCopiedGroqApiKeyHere"

```
### How to run the bot:
```
1. pip install -r requirements.txt
2. python app.py
```
