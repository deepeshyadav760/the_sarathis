# data_loader.py

import os
import unicodedata
import hashlib
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document

# SET TO True TO SEE A PREVIEW OF LOADED CONTENT
DEBUG_MODE = True

LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf-8"}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".jsonl": (JSONLoader, {"jq_schema": '.text', "json_lines": True}),
}

def fix_spaced_text(text: str) -> str:
    """
    Fix text where every character has spaces: 'H e l l o' -> 'Hello'
    """
    # Check if text has the spaced pattern (more than 30% single chars with spaces)
    words = text.split()
    single_char_words = sum(1 for w in words if len(w) == 1)
    
    if len(words) > 10 and single_char_words / len(words) > 0.3:
        # Remove spaces between single characters
        fixed = re.sub(r'(?<=\b\w)\s+(?=\w\b)', '', text)
        return fixed
    
    return text

def normalize_text(text: str) -> str:
    """
    Clean and normalize text content.
    """
    # Fix spaced text issue first
    text = fix_spaced_text(text)
    
    # Remove diacritics
    text = ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
    
    # Clean up excessive whitespace but preserve paragraph breaks
    lines = text.split('\n')
    cleaned_lines = [' '.join(line.split()) for line in lines]
    text = '\n'.join(line for line in cleaned_lines if line.strip())
    
    return text

def get_content_hash(text: str) -> str:
    """Generate a hash for text content to detect duplicates."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def clean_documents(documents):
    """
    Normalize all document page_content fields and remove duplicates.
    """
    cleaned_docs = []
    seen_hashes = set()
    
    for doc in documents:
        if hasattr(doc, "page_content") and doc.page_content.strip():
            # Normalize the text
            doc.page_content = normalize_text(doc.page_content)
            
            # Check for duplicates and minimum length
            content_hash = get_content_hash(doc.page_content)
            if content_hash not in seen_hashes and len(doc.page_content) > 20:
                seen_hashes.add(content_hash)
                cleaned_docs.append(doc)
    
    return cleaned_docs

def load_pdf_with_pypdf(file_path: str):
    """
    Load PDF using PyPDF with better text extraction.
    """
    try:
        from pypdf import PdfReader
        
        print(f"   -> Using PyPDF for extraction...")
        reader = PdfReader(file_path)
        
        documents = []
        total_text = ""
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Fix the spaced text issue
                text = fix_spaced_text(text)
                total_text += text + "\n\n"
                
                # Create document for each page
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num + 1}
                )
                documents.append(doc)
        
        if len(total_text.strip()) > 50:
            print(f"   SUCCESS: Extracted {len(documents)} pages, {len(total_text)} chars")
            return clean_documents(documents)
        else:
            print(f"   FAILED: Only extracted {len(total_text)} chars")
            return []
            
    except Exception as e:
        print(f"   FAILED: PyPDF error: {str(e)[:100]}")
        return []

def load_pdf_with_pdfplumber(file_path: str):
    """
    Load PDF using pdfplumber (better for complex layouts).
    """
    try:
        import pdfplumber
        
        print(f"   -> Using pdfplumber for extraction...")
        documents = []
        total_text = ""
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    total_text += text + "\n\n"
                    
                    doc = Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_num + 1}
                    )
                    documents.append(doc)
        
        if len(total_text.strip()) > 50:
            print(f"   SUCCESS: Extracted {len(documents)} pages, {len(total_text)} chars")
            return clean_documents(documents)
        else:
            print(f"   FAILED: Only extracted {len(total_text)} chars")
            return []
            
    except ImportError:
        print(f"   SKIPPED: pdfplumber not installed (pip install pdfplumber)")
        return []
    except Exception as e:
        print(f"   FAILED: pdfplumber error: {str(e)[:100]}")
        return []

def load_pdf_with_fallback(file_path: str):
    """
    Try multiple methods to extract text from PDF.
    """
    
    # Method 1: pdfplumber (best for layout preservation)
    documents = load_pdf_with_pdfplumber(file_path)
    if documents:
        return documents
    
    # Method 2: pypdf (fast and reliable)
    documents = load_pdf_with_pypdf(file_path)
    if documents:
        return documents
    
    # Method 3: UnstructuredLoader
    try:
        print(f"   -> Using UnstructuredLoader...")
        loader = UnstructuredLoader(file_path)
        documents = loader.load()
        
        total_chars = sum(len(doc.page_content.strip()) for doc in documents)
        if total_chars > 50:
            print(f"   SUCCESS: Extracted {len(documents)} elements, {total_chars} chars")
            return clean_documents(documents)
    except Exception as e:
        print(f"   FAILED: UnstructuredLoader error: {str(e)[:100]}")
    
    # All methods failed
    print(f"\n   ERROR: All PDF extraction methods failed!")
    print("   Install better PDF libraries:")
    print("      pip install pdfplumber pypdf")
    
    return []

def load_single_document(file_path: str):
    """
    Load a single document with appropriate loader.
    """
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    
    # Special handling for PDFs
    if ext == ".pdf":
        return load_pdf_with_fallback(file_path)
    
    # Handle other file types
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        print(f"   -> Using loader: {loader_class.__name__}")
        loader = loader_class(file_path, **loader_args)
    else:
        print(f"   -> Using fallback loader: UnstructuredLoader")
        loader = UnstructuredLoader(file_path)
    
    try:
        documents = loader.load()
        return clean_documents(documents)
    except Exception as e:
        print(f"   ERROR: {str(e)[:100]}")
        return []

def remove_duplicate_chunks(chunks):
    """
    Remove duplicate chunks based on content hash.
    """
    unique_chunks = []
    seen_hashes = set()
    
    for chunk in chunks:
        content_hash = get_content_hash(chunk.page_content)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
    
    duplicates_removed = len(chunks) - len(unique_chunks)
    if duplicates_removed > 0:
        print(f"   -> Removed {duplicates_removed} duplicate chunks")
    
    return unique_chunks

def load_and_chunk_directory(directory_path: str):
    """
    Load all documents from a directory and split them into chunks.
    """
    all_docs = []
    print(f"\nLoading documents from directory: {directory_path}")
    print("="*70)
    
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    if not files:
        print(f"WARNING: No files found in {directory_path}")
        return []
    
    print(f"Found {len(files)} file(s) to process\n")
    
    for idx, file_name in enumerate(files, 1):
        file_path = os.path.join(directory_path, file_name)
        
        try:
            print(f"[{idx}/{len(files)}] Processing: {file_name}")
            documents = load_single_document(file_path)
            
            if documents:
                total_chars = sum(len(doc.page_content) for doc in documents)
                print(f"   SUCCESS: Loaded {len(documents)} document(s), {total_chars} chars")
                
                if DEBUG_MODE and documents:
                    print("\n" + "-"*70)
                    print("CONTENT PREVIEW:")
                    print("-"*70)
                    preview = documents[0].page_content[:400].strip()
                    print(preview)
                    if len(documents[0].page_content) > 400:
                        print(f"\n... (total: {len(documents[0].page_content)} chars)")
                    print("-"*70 + "\n")
                
                all_docs.extend(documents)
            else:
                print(f"   ERROR: No content extracted")
                
        except Exception as e:
            print(f"   ERROR: {str(e)[:100]}")
            continue
    
    if not all_docs:
        print("\nERROR: No documents were successfully loaded!")
        return []

    print("="*70)
    print(f"\nTotal documents loaded: {len(all_docs)}")
    print("Creating chunks optimized for Q&A...")

    # Better chunking strategy for Q&A
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better retrieval
        chunk_overlap=150,  # Moderate overlap for context
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    
    split_chunks = text_splitter.split_documents(all_docs)
    print(f"   -> Created {len(split_chunks)} chunks")
    
    # Remove duplicates
    unique_chunks = remove_duplicate_chunks(split_chunks)
    
    print(f"\nFINAL: {len(unique_chunks)} unique chunks ready")
    print("="*70 + "\n")
    
    return unique_chunks
