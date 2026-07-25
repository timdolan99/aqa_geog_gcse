import os
import tomllib
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. Load API Keys from .streamlit/secrets.toml ---
secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path):
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        api_key = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./syllabus"

def detect_subtopic(text: str) -> str:
    content = text.lower()
    
    # Component 01: Computer Systems
    if "1.1.1" in content or "registers" in content or "fetch-decode-execute" in content or "pipelining" in content:
        return "1.1.1 Structure & Function of the Processor"
    elif "1.1.2" in content or "cisc" in content or "risc" in content or "gpu" in content or "parallel" in content:
        return "1.1.2 Types of Processor"
    elif "1.1.3" in content or "magnetic" in content or "optical" in content or "flash" in content or "virtual storage" in content:
        return "1.1.3 Input, Output and Storage"
    elif "1.2.1" in content or "paging" in content or "segmentation" in content or "scheduling" in content or "interrupts" in content:
        return "1.2.1 Operating Systems & Systems Software"
    elif "1.2.2" in content or "compiler" in content or "interpreter" in content or "assembler" in content or "lexical" in content:
        return "1.2.2 Applications Generation & Translators"
    elif "1.2.3" in content or "waterfall" in content or "agile" in content or "extreme programming" in content or "spiral" in content:
        return "1.2.3 Software Development Lifecycles"
    elif "1.2.4" in content or "procedural" in content or "little man computer" in content or "lmc" in content or "addressing modes" in content:
        return "1.2.4 Types of Programming Language & Assembly"
    elif "1.2.5" in content or "object-oriented" in content or "encapsulation" in content or "inheritance" in content or "polymorphism" in content:
        return "1.2.5 Object-Oriented Programming"
    elif "1.3.1" in content or "compression" in content or "hashing" in content or "symmetric" in content or "asymmetric" in content:
        return "1.3.1 Compression, Encryption and Hashing"
    elif "1.3.2" in content or "relational" in content or "normalisation" in content or "3nf" in content or "acid" in content or "sql" in content:
        return "1.3.2 Databases & SQL"
    elif "1.3.3" in content or "tcp/ip" in content or "dns" in content or "packet switching" in content or "firewall" in content:
        return "1.3.3 Networks & Protocols"
    elif "1.3.4" in content or "html" in content or "css" in content or "javascript" in content or "pagerank" in content:
        return "1.3.4 Web Technologies"
    elif "1.4.1" in content or "two's complement" in content or "floating point" in content or "bitwise" in content or "unicode" in content:
        return "1.4.1 Data Types & Binary Representation"
    elif "1.4.2" in content or "linked-list" in content or "binary search tree" in content or "hash table" in content or "stack" in content:
        return "1.4.2 Data Structures"
    elif "1.4.3" in content or "karnaugh" in content or "de morgan" in content or "flip flop" in content or "truth table" in content:
        return "1.4.3 Boolean Algebra & Logic Gates"
    elif "1.5.1" in content or "data protection" in content or "computer misuse" in content or "copyright" in content or "ripa" in content:
        return "1.5.1 Computing Related Legislation"
    elif "1.5.2" in content or "moral" in content or "ethical" in content or "artificial intelligence" in content or "environmental" in content:
        return "1.5.2 Moral, Ethical, Social & Cultural Issues"
        
    # Component 02: Algorithms & Programming
    elif "2.1.1" in content or "thinking abstractly" in content or "thinking ahead" in content or "concurrently" in content:
        return "2.1 Elements of Computational Thinking"
    elif "2.2.1" in content or "recursion" in content or "scope" in content or "modularity" in content or "constructs" in content:
        return "2.2.1 Programming Techniques & Recursion"
    elif "2.2.2" in content or "backtracking" in content or "heuristics" in content or "data mining" in content or "pipelining" in content:
        return "2.2.2 Computational Methods"
    elif "2.3.1" in content and ("big o" in content or "complexity" in content):
        return "2.3.1 Algorithmic Complexity & Big O"
    elif "traversal" in content or "depth-first" in content or "breadth-first" in content:
        return "2.3.1 Data Structure Algorithms & Traversals"
    elif "dijkstra" in content or "a*" in content or "merge sort" in content or "quick sort" in content or "binary search" in content:
        return "2.3.1 Standard Searching & Sorting Algorithms"

    return "General OCR A-Level Specification Context"

def build_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory {DATA_PATH}. Place your syllabus PDF there.")
        return

    print("📄 Loading OCR A-Level Computer Science PDF syllabus files...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()

    if not raw_docs:
        print(f"No PDF documents found in {DATA_PATH}. Ingestion aborted.")
        return

    print(f"Loaded {len(raw_docs)} page(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    print("🏷️ Assigning subtopic metadata tags...")
    for chunk in chunks:
        chunk.metadata["sub_topic"] = detect_subtopic(chunk.page_content)

    print("🧠 Generating embeddings and indexing to ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ Ingestion complete! Persisted {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    build_vector_db()