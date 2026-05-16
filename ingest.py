import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Path to the PDF document
PDF_PATH = "The Rust Programming Language.pdf"
CHROMA_PATH = "./chroma_db"

def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found. Please place it in the project root.")
        return

    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # Optimizing the text splitting strategy
    # The RecursiveCharacterTextSplitter defaults to breaking on paragraphs, then sentences.
    # By ensuring double newlines (\n\n) and single newlines (\n) take precedence, 
    # it respects natural code block boundaries and paragraphs better.
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} pages into {len(chunks)} chunks.")

    print("Initializing Ollama Embeddings (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("Storing embeddings in ChromaDB...")
    # Clear existing DB if you want a fresh start, or just add to it.
    if os.path.exists(CHROMA_PATH):
        print("Updating existing Chroma DB...")
    
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f"Completed! Vector database saved locally to {CHROMA_PATH}.")

if __name__ == "__main__":
    ingest_pdf()
