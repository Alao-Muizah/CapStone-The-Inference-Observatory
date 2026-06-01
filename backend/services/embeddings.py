import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer


embedding_model = None

def load_embedding_model():
   
    global embedding_model

    if embedding_model is not None:
        return

    print("Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")


# ── Cosine Similarity ──────────────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    
    dot_product = np.dot(vec_a, vec_b)
    magnitude = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

    if magnitude == 0:
        return 0.0

    score = dot_product / magnitude

    return float(np.clip(score, 0.0, 1.0))


# ── Main Entry Point ───────────────────────────────────────────────────────

async def compute_similarity(texts: dict) -> list[dict]:
    
    load_embedding_model()

    valid_texts = {
        strategy: text
        for strategy, text in texts.items()
        if text and isinstance(text, str) and text.strip()
    }

    if len(valid_texts) < 2:
        
        return []

    strategies = list(valid_texts.keys())
    text_list = list(valid_texts.values())

    print(f"Embedding {len(text_list)} texts...")
    embeddings = embedding_model.encode(text_list, convert_to_numpy=True)
    print("Embeddings computed.")

    scores = []

    for i, j in combinations(range(len(strategies)), 2):
        score = cosine_similarity(embeddings[i], embeddings[j])

        scores.append({
            "strategy_a": strategies[i],
            "strategy_b": strategies[j],
            "score": round(score, 4),
        })

    scores.sort(key=lambda x: x["score"], reverse=True)

    return scores