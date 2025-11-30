from FlagEmbedding import FlagReranker

reranker_model = FlagReranker('BAAI/bge-reranker-v2-m3', devices='cuda:0', use_fp16=True
                                  )
def reranker(query, contexts, top_k):

    pairs =  [(query, context) for context in contexts]
    scores = reranker_model.compute_score(pairs)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_k_contexts = [contexts[i] for i in top_k_indices]
    return top_k_contexts

