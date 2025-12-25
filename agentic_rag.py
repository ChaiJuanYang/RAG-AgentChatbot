import os

from supabase import Client, create_client
import google.generativeai as genai
from dotenv import load_dotenv
import supabase

load_dotenv()

# env and clients setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
SUPABASE_CLIENT: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-flash-latest")


# Embeddings 
def embed_text(text: str):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    embedding = response["embedding"]
    return embedding

# Retrieve Context
def dense_retrieve(query: str, top_k: int = 5):
    query_embedding = embed_text(query)
    
    response = SUPABASE_CLIENT.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k # number of similar documents to retrieve
        }
    ).execute()
    
    results = response.data
    contexts = [item['content'] for item in results]
    return contexts

def sparse_retrieve(query: str, top_k: int = 5):    
    response = (SUPABASE_CLIENT.table('documents')
                .select('content,metadata')
                .text_search('content', query) # for direct text search
                .limit(top_k)
                .execute()) 
    
    results = response.data
    contexts = [item['content'] for item in results]
    return contexts

# Hybrid retrieval 
def hybrid_retrieve(query: str, top_k: int = 10):
    query_embedding = embed_text(query)
    response = SUPABASE_CLIENT.rpc(
        "hybrid_match_documents",
        {
            "query_embedding": query_embedding,
            "query_text": query,
            "match_count": top_k
        }
    ).execute()

    return response.data or []

# Reranking 
def rerank_prompt(query: str, passage: str):
    return f"""
You are a relevance judge for a retrieval system.
Read the query and passage carefully, Score relevance strictly.

10 = directly answers the query
7 - 9 = provides key supporting details
4 - 6 = related but not sufficient alone
0 - 3 = irrelevant
Query:
{query}

Passage:
{passage}

Score the relevance from 0 (irrelevant) to 10 (highly relevant).
Return ONLY the number.
"""

def rerank(query, documents,top_k):
    scored_docs = []
    for idx,doc in enumerate(documents): 
        metadata = doc.get("metadata", {})
        print(f"[Candidate {idx+1}]")
        print(f"Content Preview: {doc['content'][:120]}...")
        print(f"ChunkID: {metadata.get('chunk_id')}")
        print(f"Source: {metadata.get('source')}\n")
        prompt = rerank_prompt(query, doc['content'])
        response = model.generate_content(prompt, generation_config={"temperature": 0})

        try:
            score = float(response.text.strip())
        except:
            score = 0.0
        print(f"Relevance Score: {score}\n")
        scored_docs.append((score, doc))
    print("--- RERANKING RESULT ---")
    scored_docs.sort(key=lambda x: x[0], reverse=True) # highest relevance first
    for rank, (score, doc) in enumerate(scored_docs[:top_k], start=1):
        metadata = doc.get("metadata", {})
        print(f"Rank {rank} | Score {score}")
        print(f"Snippet: {doc['content'][:120]}...")
        print(f"ChunkID: {metadata.get('chunk_id')}")
        print(f"Source: {metadata.get('source')}\n")
    return [doc for _,doc in scored_docs[:top_k]]  # keep only top_k documentts

# Agentic Features
def plan_prompt(query: str):
    return f"""
You are an AI agent answering banking FX disclosure questions.

Decide the steps needed to answer the query.
Choose from:
- RETRIEVE (search documents)
- ANSWER (generate answer)
- REFUSE (out of scope)

Return steps as a JSON list.
Example:
["RETRIEVE", "ANSWER"]

Query:
{query}
"""

def plan(query, model): 
    response = model.generate_content(
        plan_prompt(query),
        generation_config={"temperature": 0}
    )   
    print("\n--- Planning Response ---")
    try : 
        steps = eval(response.text.strip())
        print(steps)
        return steps
    
    except :
        print("Error parsing planning response, defaulting to ['RETRIEVE', 'ANSWER']")
        return ["RETRIEVE", "ANSWER"]

def agentic_answer(query: str, model):
    steps = plan(query, model)
    memory = {
        "documents" : [],
        "answer" : None
    }

    for step in steps:  
        if step == "RETRIEVE":
            retrieved_docs = hybrid_retrieve(query, top_k=10)
            reranked_docs = rerank(query, retrieved_docs, top_k=3)
            memory["documents"] = reranked_docs

        elif step == "ANSWER":
            contexts = [doc["content"] for doc in memory["documents"]]
            prompt = build_prompt(query, contexts)
            response = model.generate_content(prompt)
            answer = response.text.strip()
            if grounded_check(answer, "\n".join(contexts), model):
                memory["answer"] = answer
            else:
                memory["answer"] = "I'm unable to answer confidently based on the provided disclosure."

        elif step == "REFUSE":
            memory["answer"] = "I'm sorry. This question is outside the scope of FX product disclosures."
    
    return memory["answer"]

def grounded_check(answer, contexts, model):
    prompt = f"""
        Is the following answer fully supported by the context?
        Answer YES or NO.

        Context:
        {contexts}

        Answer:
        {answer}
    """
    response = model.generate_content(prompt, generation_config={"temperature": 0})
    print("\n--- Grounded Check Response ---")
    print(response.text.strip())
    return "YES" in response.text.upper()


# Prompt Building 
def build_prompt(query: str, contexts: list):
    context_str = "\n\n".join(contexts)
    prompt = f"""Use the following pieces of retrieved context to answer the question. \
    Use three to five sentences maximum and keep the answer concise, while still giving depth.\
    If the context does not contain the answer, say 'I'm sorry I don't have an answer for this question, please try again.'.\n\n\
    
    Context:\n{context_str}\n\nQuestion: {query}\
    """

    return prompt

TOOLS = {
    "hybrid_retrieve": hybrid_retrieve,
    "rerank": rerank,
    "answer": build_prompt
}

# chat 
def chat_with_context(query: str):
    retrieved_docs = hybrid_retrieve(query, top_k=10)
    reranked_docs = rerank(query, retrieved_docs, top_k=3)
    
    contexts = [doc["content"] for doc in reranked_docs]
    prompt = build_prompt(query, contexts)

    response = model.generate_content(prompt)
    return response.text.strip()

if __name__ == "__main__":
    # for m in genai.list_models():
    #     print(m.name, m.supported_generation_methods)
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", agentic_answer(q, model))