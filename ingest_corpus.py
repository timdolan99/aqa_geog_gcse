import os, json, tomllib
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load API key from secrets.toml
secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path) and "GOOGLE_API_KEY" not in os.environ:
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        os.environ["GOOGLE_API_KEY"] = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY", "")

class CourseSpec(BaseModel):
    course_id: str = Field(description="e.g. aqa_gcse_geography")
    course_title: str = Field(description="e.g. AQA GCSE Geography")
    level: str = Field(description="GCSE or A-Level")
    target_turns: int = Field(description="5 for GCSE, 7 for A-Level")
    topics: Dict[str, List[str]]

def process_corpus():
    loader = DirectoryLoader("./syllabus", glob="*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()
    full_text = "\n\n".join([d.page_content for d in raw_docs])

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    spec = llm.with_structured_output(CourseSpec).invoke(
        f"Extract metadata and detailed subtopics for each main section. Set target_turns=5 if GCSE, 7 if A-Level:\n\n{full_text[:15000]}"
    )
    
    with open("course_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec.model_dump(), f, indent=2)
    print("✅ Created course_spec.json!")

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(raw_docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    process_corpus()
