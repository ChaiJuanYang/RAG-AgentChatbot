import os
from flask import Flask, request, jsonify, render_template
from supabase import Client, create_client
from dotenv import load_dotenv
from app_rag import answer_query
import google.generativeai as genai


load_dotenv()
app = Flask(__name__)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
genai.configure(api_key=os.getenv("GEMINI_KEY"))
model = genai.GenerativeModel("gemini-flash-latest")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message")

    if not query:
        return jsonify({"error": "Empty message"}), 400

    answer = answer_query(query, supabase, model)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)