import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))

res = genai.embed_content(
    model="models/embedding-001",
    content="Hello world"
)

print(len(res["embedding"]))
