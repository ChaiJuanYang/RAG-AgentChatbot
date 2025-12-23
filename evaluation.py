import json
from agentic_rag import hybrid_retrieve, rerank


def normalize_retrieved_docs(docs):
    return {
        (doc["metadata"].get("source"), doc["metadata"].get("chunk_id"))
        for doc in docs
    }


def precision_recall_at_k(retrieved_docs, relevant_chunks, k):
    retrieved_at_k = retrieved_docs[:k]
    retrieved_set = normalize_retrieved_docs(retrieved_at_k)

    gt_set = {
        (item["source"], item["chunk_id"])
        for item in relevant_chunks
    }

    true_positives = retrieved_set & gt_set

    denom = min(k, len(retrieved_at_k))
    precision = len(true_positives) / denom if denom else 0
    recall = len(true_positives) / len(gt_set) if gt_set else 0

    return precision, recall


def evaluate_ground_truth(ground_truth_data, gen_k=3, retrieve_k=10):
    precision_at_3, recall_at_10 = [], []

    for item in ground_truth_data:
        query = item["query"]
        relevant_chunks = item["relevant_chunks"]

        if not relevant_chunks:
            print(f"Skipping (no ground truth): {query}")
            continue

        # Stage 1: retrieval
        retrieved_docs = hybrid_retrieve(query, top_k=retrieve_k)

        # Stage 2: reranking (keep enough to evaluate)
        reranked_docs = rerank(query, retrieved_docs, top_k=retrieve_k)

        # Evaluation
        _, r_10 = precision_recall_at_k(
            retrieved_docs, relevant_chunks, retrieve_k
        )
        # precision at 3 , are generated contexts relevant
        p_3, _ = precision_recall_at_k(
            reranked_docs, relevant_chunks, gen_k
        )

        print(f"\nQuery: {query}")
        print(f"Recall@{retrieve_k}:     {r_10:.2f}")
        print(f"Precision@{gen_k}:       {p_3:.2f}")

        recall_at_10.append(r_10)
        precision_at_3.append(p_3)

    avg_p3 = sum(precision_at_3) / len(precision_at_3)
    avg_r10 = sum(recall_at_10) / len(recall_at_10)
    print("\n=== OVERALL METRICS ===")
    print(f"Average Precision@{gen_k}: {avg_p3:.2f}")
    print(f"Average Recall@{retrieve_k}: {avg_r10:.2f}")
    return avg_p3, avg_r10


with open("ground_truth.json") as f:
    ground_truth_data = json.load(f)

evaluate_ground_truth(ground_truth_data, gen_k=3)
