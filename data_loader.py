# # data_loader.py

# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # --- STABILITY FIX: Import the more stable PyPDFLoader ---
# from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
# from langchain_unstructured import UnstructuredLoader

# # SET TO True TO SEE A PREVIEW OF LOADED CONTENT
# DEBUG_MODE = True

# # --- STABILITY FIX: Revert to PyPDFLoader for reliability ---
# LOADER_MAPPING = {
#     ".csv": (CSVLoader, {"encoding": "utf-8"}),
#     ".pdf": (PyPDFLoader, {}), # Reverted to the stable PyPDFLoader
#     ".txt": (TextLoader, {"encoding": "utf-8"}),
# }

# def load_single_document(file_path: str):
#     ext = "." + file_path.rsplit(".", 1)[-1].lower()
#     if ext in LOADER_MAPPING:
#         loader_class, loader_args = LOADER_MAPPING[ext]
#         print(f"   -> Using loader: {loader_class.__name__}")
#         loader = loader_class(file_path, **loader_args)
#     else:
#         print(f"   -> Using fallback loader: UnstructuredLoader")
#         loader = UnstructuredLoader(file_path)
    
#     return loader.load()

# def load_and_chunk_directory(directory_path: str):
#     all_docs = []
#     print(f"Loading documents from directory: {directory_path}")
    
#     for file_name in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, file_name)
        
#         if os.path.isfile(file_path):
#             try:
#                 print(f"\n-> Processing file: {file_name}")
#                 documents = load_single_document(file_path)
                
#                 if DEBUG_MODE and documents:
#                     print("="*20 + " [CONTENT PREVIEW] " + "="*20)
#                     print(documents[0].page_content[:500].strip())
#                     print("="*62)

#                 all_docs.extend(documents)
#             except Exception as e:
#                 print(f"   [!] Failed to load {file_name}: {e}")
#                 continue
    
#     if not all_docs:
#         return []

#     print(f"\nTotal documents loaded: {len(all_docs)}")
#     print("Now splitting documents into chunks...")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     split_chunks = text_splitter.split_documents(all_docs)
    
#     print(f"Total chunks created: {len(split_chunks)}")
#     return split_chunks




# data_loader.py

import os
import unicodedata
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- STABILITY FIX: Import the more stable PyPDFLoader ---
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import JSONLoader

# SET TO True TO SEE A PREVIEW OF LOADED CONTENT
DEBUG_MODE = True

# --- STABILITY FIX: Revert to PyPDFLoader for reliability ---
LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf-8"}),
    ".pdf": (PyPDFLoader, {}),  # Reverted to the stable PyPDFLoader
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".jsonl": (JSONLoader, {"jq_schema": '.text', "json_lines": True}),
}

def normalize_text(text: str) -> str:
    """
    Remove diacritics (accents like ā, ṇ, ṛ) and keep only plain English letters.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def clean_documents(documents):
    """
    Normalize all document page_content fields to plain English (remove diacritics).
    """
    for doc in documents:
        if hasattr(doc, "page_content"):
            doc.page_content = normalize_text(doc.page_content)
    return documents

def load_single_document(file_path: str):
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        print(f"   -> Using loader: {loader_class.__name__}")
        loader = loader_class(file_path, **loader_args)
    else:
        print(f"   -> Using fallback loader: UnstructuredLoader")
        loader = UnstructuredLoader(file_path)
    
    documents = loader.load()
    return clean_documents(documents)

def load_and_chunk_directory(directory_path: str):
    all_docs = []
    print(f"Loading documents from directory: {directory_path}")
    
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.isfile(file_path):
            try:
                print(f"\n-> Processing file: {file_name}")
                documents = load_single_document(file_path)
                
                if DEBUG_MODE and documents:
                    print("="*20 + " [CONTENT PREVIEW] " + "="*20)
                    print(documents[0].page_content[:500].strip())
                    print("="*62)

                all_docs.extend(documents)
            except Exception as e:
                print(f"   [!] Failed to load {file_name}: {e}")
                continue
    
    if not all_docs:
        return []

    print(f"\nTotal documents loaded: {len(all_docs)}")
    print("Now splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_chunks = text_splitter.split_documents(all_docs)
    
    print(f"Total chunks created: {len(split_chunks)}")
    return split_chunks
