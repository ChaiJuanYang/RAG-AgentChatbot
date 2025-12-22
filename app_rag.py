from supabase import Client 
import google.generativeai as genai


# Embeddings 
def embed_text(text: str):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    embedding = response["embedding"]
    return embedding

# Hybrid retrieval 
def hybrid_retrieve(query: str, supabase: Client, top_k: int = 5):
    query_embedding = embed_text(query)
    response = supabase.rpc(
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

Query:
{query}

Passage:
{passage}

Score the relevance from 0 (irrelevant) to 10 (highly relevant).
Return ONLY the number.
"""

def rerank(query, documents, model, top_k= 10):
    scored_docs = []
    for idx,doc in enumerate(documents): 
        # print(f"[Candidate {idx+1}]")
        # print(f"Content Preview: {doc['content'][:120]}...")
        prompt = rerank_prompt(query, doc['content'])
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        try:
            score = float(response.text.strip())
        except:
            score = 0.0
        # print(f"Relevance Score: {score}\n")
        scored_docs.append((score, doc))
    # print("--- RERANKING RESULT ---")
    scored_docs.sort(key=lambda x: x[0], reverse=True) # highest relevance first
    # for rank, (score, doc) in enumerate(scored_docs[:top_k], start=1):
    #     print(f"Rank {rank} | Score {score}")
    #     print(f"Snippet: {doc['content'][:120]}...\n")
    return [doc for _,doc in scored_docs[:top_k]]  # keep only top_k documentts

# Prompt Building 
def build_prompt(query: str, contexts: list):
    context_str = "\n\n".join(contexts)
    prompt = f"""Use the following pieces of retrieved context to answer the question. \
    Use three to five sentences maximum and keep the answer concise, while still giving depth.\
    If the context does not contain the answer, say 'I'm sorry I don't have an answer for this question, please try again.'.\n\n\
    
    Context:\n{context_str}\n\nQuestion: {query}\
    """
    return prompt

def answer_query(query, supabase, model):
    retrieved = hybrid_retrieve(query, supabase, top_k=10)
    reranked = rerank(query, retrieved, model)
    contexts = [doc['content'] for doc in reranked]
    prompt = build_prompt(query, contexts)
    response = model.generate_content(prompt)
    return response.text.strip()