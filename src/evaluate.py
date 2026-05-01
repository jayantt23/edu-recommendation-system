"""
Evaluation: Precision@K, Recall@K and NDCG@K using synthetic ground truth.
Ground truth = schools the synthetic user 'interacted with' (preference-aligned).
Geo penalty is disabled (gamma=0) during evaluation so distance doesn't bias
the measurement of topic + CF accuracy.

Also runs an ABLATION STUDY comparing:
  - Content-Only  (lambda_u forced to 0 → pure topic + metric)
  - CF-Only       (lambda_u forced to 1 → pure collaborative filtering)
  - Hybrid        (lambda_u adaptive, as per paper)
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling.recommender import RecommenderEngine

N_TOPICS   = 5
DATA_PATH  = "data/processed/final_hybrid_dataset.pkl"
USERS_PATH = "data/processed/synthetic_users.pkl"
INTER_PATH = "data/processed/synthetic_interactions.pkl"


# ── Metric helpers ───────────────────────────────────────────────────────────

def dcg_at_k(relevances, k):
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    return (relevances / np.log2(np.arange(2, len(relevances) + 2))).sum()


def ndcg_at_k(recommended_ids, relevant_ids, k):
    rel   = [1 if s in relevant_ids else 0 for s in recommended_ids[:k]]
    ideal = sorted(rel, reverse=True)
    dcg   = dcg_at_k(rel, k)
    idcg  = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended_ids, relevant_ids, k):
    return sum(1 for s in recommended_ids[:k] if s in relevant_ids) / k


def recall_at_k(recommended_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    return sum(1 for s in recommended_ids[:k] if s in relevant_ids) / len(relevant_ids)


# ── Core evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(recommender, df, users_df, interactions, k, n_test, seed):
    """Returns (mean_precision, mean_recall, mean_ndcg)."""
    np.random.seed(seed)
    test_users = users_df.sample(n=min(n_test, len(users_df)), random_state=seed)

    precisions, recalls, ndcgs = [], [], []

    for _, user in test_users.iterrows():
        uid         = user['user_id']
        wu          = user['wv']
        theta_q     = wu
        user_loc    = (34.05, -118.24)
        relevant_ids = interactions.get(uid, set())
        if not relevant_ids:
            continue

        results = recommender.get_recommendations(
            user_query_theta    = theta_q,
            user_loc            = user_loc,
            candidate_df        = df,
            wu                  = wu,
            historical_users_df = users_df[users_df['user_id'] != uid],
            interactions        = interactions,
        )

        recs = results['ncessch'].tolist()
        precisions.append(precision_at_k(recs, relevant_ids, k))
        recalls.append(recall_at_k(recs, relevant_ids, k))
        ndcgs.append(ndcg_at_k(recs, relevant_ids, k))

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


# ── Main ─────────────────────────────────────────────────────────────────────

def evaluate(k=5, n_test_users=50, seed=42):
    # ── Load data ──────────────────────────────────────────────────────────
    for path in [DATA_PATH, USERS_PATH, INTER_PATH]:
        if not os.path.exists(path):
            print(f"Missing: {path}\nRun run_full_pipeline.py first.")
            return

    df       = pd.read_pickle(DATA_PATH)
    users_df = pd.read_pickle(USERS_PATH)
    with open(INTER_PATH, "rb") as f:
        interactions = pickle.load(f)

    print(f"Evaluating K={k}  |  users={n_test_users}  |  schools={len(df)}")
    print(f"Ground truth: preference-aligned interactions  |  Geo penalty: DISABLED\n")

    # ── Ablation study ─────────────────────────────────────────────────────
    configs = {
        "Content-Only (λ_u=0)":    RecommenderEngine(alpha=0.5, beta=0.3, gamma=0.0, k_target=5, lambda_max=0.0),
        "CF-Only (λ_u=1)":         RecommenderEngine(alpha=0.5, beta=0.3, gamma=0.0, k_target=5, lambda_max=1.0),
        "Hybrid (λ_u adaptive)":   RecommenderEngine(alpha=0.5, beta=0.3, gamma=0.0, k_target=5, lambda_max=0.8),
    }

    header = f"{'Model':<28} | {'P@'+str(k):>6} | {'R@'+str(k):>6} | {'NDCG@'+str(k):>7}"
    print(header)
    print("-" * len(header))

    for name, rec in configs.items():
        p, r, n = run_evaluation(rec, df, users_df, interactions, k, n_test_users, seed)
        print(f"{name:<28} | {p:>6.4f} | {r:>6.4f} | {n:>7.4f}")

    print("-" * len(header))
    print("\nAblation complete.")


if __name__ == "__main__":
    evaluate(k=5)
