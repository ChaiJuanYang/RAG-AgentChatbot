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
def hybrid_retrieve(query: str, supabase: Client, top_k: int = 10):
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

def rerank(query, documents, model, top_k):
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

#Agentic Features 
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

def agentic_answer(query: str, supabase, model):
    steps = plan(query, model)
    memory = {
        "documents" : [],
        "answer" : None
    }

    for step in steps:  
        if step == "RETRIEVE":
            retrieved_docs = hybrid_retrieve(query, supabase, top_k=10)
            reranked_docs = rerank(query, retrieved_docs, model, top_k=3)
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

# Ground Checking

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


# Old version of answer_query for compatibility, new version uses agentic_answer
def answer_query(query, supabase, model):
    retrieved = hybrid_retrieve(query, supabase, top_k=10)
    reranked = rerank(query, retrieved, model)
    contexts = [doc['content'] for doc in reranked]
    prompt = build_prompt(query, contexts)
    response = model.generate_content(prompt)
    return response.text.strip()