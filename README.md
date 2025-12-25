**Agentic FX Disclosure Chatbot (RAG)** 

Answers questions about Foreign Exchange (FX) product disclosures using official bank documents
🔍 **Retrieval**: Uses hybrid search
  🧠 Dense vector embeddings (semantic search)
  🔤 Sparse keyword search (full-text search)

🎯 **Reranking**: LLM-based reranker scores and selects the most relevant chunks before answering

🧭 **Agentic Flow**:
  1. Plans steps (Retrieve → Answer / Refuse)
  2.Maintains short-term memory
  3.Performs grounding checks to reduce hallucinations

🛡️ **Safety & Reliability**:
  - Refuses out-of-scope questions
  - Verifies answers are supported by retrieved context

📊 **Evaluation**:
  ✅ Recall@10: 0.91 (strong coverage of relevant disclosures)
  🎯 Precision@3: 0.80 (high relevance for generated answers)

🌐 **Interface**: 
  Flask-based web app with chat UI for interactive querying
