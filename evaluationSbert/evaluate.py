import numpy as np
from tqdm.auto import tqdm

def encode_texts(model, texts, batch_size: int = 32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

def compute_cosine_similarity(query_emb, passage_embs):
    q = query_emb.squeeze()
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    p_norms = passage_embs / (np.linalg.norm(passage_embs, axis=1, keepdims=True) + 1e-8)
    
    scores = np.dot(p_norms, q_norm)
    
    scores = (scores + 1.0) / 2.0
    return scores

def rank_passages_for_query(model, query, passages, top_k: int):
    if not passages:
        return []
        
    query_emb = encode_texts(model, [query], batch_size=1)
    passage_embs = encode_texts(model, passages, batch_size=32)
    
    scores = compute_cosine_similarity(query_emb, passage_embs)
    
    indices = np.argsort(scores)[::-1]
    
    ranked = []
    for idx in indices[:top_k]:
        ranked.append((int(idx), float(scores[idx])))
        
    return ranked

def rank_all_queries(model, dev_data, top_k: int):
    rankings = {}
    for qid, item in tqdm(dev_data.items(), desc="Ranking Queries"):
        query = item["query"]
        candidates = item.get("candidates", [])
        
        passages = [c["passage"] for c in candidates]
        labels = [c["label"] for c in candidates]
        
        if not passages:
            rankings[qid] = []
            continue
            
        ranked_indices_scores = rank_passages_for_query(model, query, passages, top_k)
        
        q_ranking = []
        for idx, score in ranked_indices_scores:
            q_ranking.append((idx, score, labels[idx]))
            
        rankings[qid] = q_ranking
        
    return rankings

def recall_at_k(ranked_list, k: int, total_relevant: int = None):
    if total_relevant == 0 or total_relevant is None:
        return 0.0
    
    relevant_in_top_k = sum(1 for _, _, label in ranked_list[:k] if label == 1)
    return relevant_in_top_k / total_relevant

def mrr(ranked_list):
    for rank, (_, _, label) in enumerate(ranked_list, 1):
        if label == 1:
            return 1.0 / rank
    return 0.0

def accuracy_at_k(ranked_list, k: int):
    for _, _, label in ranked_list[:k]:
        if label == 1:
            return 1.0
    return 0.0

def evaluate_all(rankings, dev_data, k: int):

    total_queries = 0
    sum_recall = 0.0
    sum_mrr = 0.0
    sum_accuracy = 0.0
    
    for qid, ranked_list in rankings.items():
        candidates = dev_data[qid].get("candidates", [])
        total_relevant = sum(1 for c in candidates if c["label"] == 1)
        
        if total_relevant == 0:
            continue
            
        total_queries += 1
        sum_recall += recall_at_k(ranked_list, k, total_relevant)
        sum_mrr += mrr(ranked_list)
        sum_accuracy += accuracy_at_k(ranked_list, k)
        
    if total_queries == 0:
        return {f"Recall@{k}": 0.0, "MRR": 0.0, f"Accuracy@{k}": 0.0}
        
    return {
        f"Recall@{k}": sum_recall / total_queries,
        "MRR": sum_mrr / total_queries,
        f"Accuracy@{k}": sum_accuracy / total_queries
    }