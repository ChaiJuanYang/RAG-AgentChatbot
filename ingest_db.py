import os

from dotenv import load_dotenv
from supabase.client import Client, create_client
import google.generativeai as genai  
from pypdf import PdfReader
from tqdm import tqdm

load_dotenv() 

# env and clients setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
SUPABASE_CLIENT: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
# print("Supabase and OpenAI API keys loaded successfully.")

# embedding text
def embed_text(text:str): 
    response = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    embedding = response["embedding"]
    # print("Text embedded successfully." + str(embedding))
    return embedding

# pdf loader
def load_pdf(file_path: str) : 
    reader = PdfReader(file_path)
    pages = []  
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    print(f"Loaded {len(pages)} pages from PDF.")
    return "\n".join(pages)

# text chunking
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    print(f"Text chunked into {len(chunks)} chunks.")
    return chunks


def ingest_text(text: str, source: str):
    chunk = chunk_text(text)
    for i, chunk in enumerate(tqdm(chunk)):
        embedding = embed_text(chunk)
        data = {
            "content": chunk,
            "embedding": embedding,
            "metadata": {
                "source": source,
                "chunk_id": i
                }
        }
        SUPABASE_CLIENT.table("documents").insert(data).execute()

def ingest_pdf(file_path: str): 
    text = load_pdf(file_path)
    ingest_text(text, source=file_path)

if __name__ == "__main__":
    pdf_path = "documents/fx-cost-of-service-client-disclosure.pdf"  # Replace with PDF file path
    ingest_pdf(pdf_path)