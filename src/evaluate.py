"""
Evaluation: Precision@K, Recall@K and NDCG@K using synthetic ground truth.
Includes extended baselines (ALS Matrix Factorization and Switching Hybrid) 
to validate the continuous mathematical superiority of the Adaptive Confidence Factor.
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
import implicit
import scipy.sparse as sparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling.recommender import RecommenderEngine

N_TOPICS   = 5
DATA_PATH  = "data/processed/final_hybrid_dataset.pkl"
USERS_PATH = "data/processed/synthetic_users.pkl"
INTER_PATH = "data/processed/synthetic_interactions.pkl"

# ── ALS Baseline Wrapper ─────────────────────────────────────────────────────
class ALSRecommenderWrapper:
    """Wraps Implicit ALS to match the RecommenderEngine interface."""
    def __init__(self):
        self.model = None
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.is_fit = False # NEW: Track if model actually trained

    def get_recommendations(self, user_query_theta, user_loc, candidate_df, wu, historical_users_df, interactions):
        if self.model is None:
            self.model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20, random_state=42)
            self.item_mapping = {sid: i for i, sid in enumerate(candidate_df['ncessch'])}
            self.reverse_item_mapping = {i: sid for sid, i in self.item_mapping.items()}

            hist_uids = historical_users_df['user_id'].tolist()
            user_mapping = {uid: i for i, uid in enumerate(hist_uids)}

            rows, cols, data = [], [], []
            for uid in hist_uids:
                sids = interactions.get(uid, set())
                u_idx = user_mapping.get(uid)
                if u_idx is None: continue
                for sid in sids:
                    if sid in self.item_mapping:
                        rows.append(self.item_mapping[sid])
                        cols.append(u_idx)
                        data.append(1.0)

            # Prevent crash if historical data is empty (Cold Start)
            if not data:
                self.is_fit = False
            else:
                item_user_csr = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.item_mapping), len(user_mapping)))
                self.model.fit(item_user_csr, show_progress=False)
                self.is_fit = True

        # If it never trained on data, safely return baseline 0 for ALL users
        if not self.is_fit:
            return candidate_df.assign(final_score=0.0)

        user_items = sparse.csr_matrix((1, len(self.item_mapping)))
        ids, scores = self.model.recommend(0, user_items, N=len(candidate_df), recalculate_user=True)

        res = [{'ncessch': self.reverse_item_mapping[i], 'final_score': float(s)} for i, s in zip(ids, scores)]
        return pd.DataFrame(res).merge(candidate_df[['ncessch']], on='ncessch')

# ── Metric helpers ───────────────────────────────────────────────────────────
def dcg_at_k(relevances, k):
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0: return 0.0
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
    if not relevant_ids: return 0.0
    return sum(1 for s in recommended_ids[:k] if s in relevant_ids) / len(relevant_ids)

# ── Core evaluation loop ─────────────────────────────────────────────────────
def run_evaluation(recommender, df, test_users_df, historical_users_df, interactions, k):
    precisions, recalls, ndcgs = [], [], []

    for _, user in test_users_df.iterrows():
        uid         = user['user_id']
        wu          = user['wv']
        relevant_ids = interactions.get(uid, set())
        if not relevant_ids: continue
            
        hist_df = historical_users_df[historical_users_df['user_id'] != uid] if not historical_users_df.empty else historical_users_df

        results = recommender.get_recommendations(
            user_query_theta    = wu,
            user_loc            = (34.05, -118.24),
            candidate_df        = df,
            wu                  = wu,
            historical_users_df = hist_df,
            interactions        = interactions,
        )

        recs = results['ncessch'].tolist()
        precisions.append(precision_at_k(recs, relevant_ids, k))
        recalls.append(recall_at_k(recs, relevant_ids, k))
        ndcgs.append(ndcg_at_k(recs, relevant_ids, k))

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

# ── Main ─────────────────────────────────────────────────────────────────────
def evaluate(k=5, n_test_users=50, seed=42):
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

    # ── Extended Ablation Study ──────────────────────────────────────────────
    configs = {
        "ALS Matrix Factorization": RecommenderEngine(), # Overridden below
        "Content-Only (λ_u=0)":     RecommenderEngine(alpha=0.8, beta=0.6, gamma=0.0, k_target=25, lambda_max=0.0),
        "CF-Only (λ_u=1)":          RecommenderEngine(alpha=0.4, beta=0.6, gamma=0.0, k_target=25, lambda_max=1.0),
        "Switching Hybrid (Bool)":  RecommenderEngine(alpha=0.4, beta=0.6, gamma=0.0, k_target=25, lambda_max=1.0, hard_switch=True),
        "Adaptive Hybrid (Ours)":   RecommenderEngine(alpha=0.4, beta=0.6, gamma=0.0, k_target=25, lambda_max=1.0),
    }
    
    # Inject ALS properly
    configs["ALS Matrix Factorization"] = ALSRecommenderWrapper()

    print("\n=== 1. EXTENDED ABLATION STUDY (Full User History) ===")
    header = f"{'Model':<28} | {'P@'+str(k):>6} | {'R@'+str(k):>6} | {'NDCG@'+str(k):>7}"
    print(header)
    print("-" * len(header))

    test_users_df = users_df.sample(n=min(n_test_users, len(users_df)), random_state=seed)

    for name, rec in configs.items():
        p, r, n = run_evaluation(rec, df, test_users_df, users_df, interactions, k)
        print(f"{name:<28} | {p:>6.4f} | {r:>6.4f} | {n:>7.4f}")

    print("-" * len(header))

    # ── Cold Start Simulation ──────────────────────────────────────────────
    print("\n=== 2. COLD-START SIMULATION (NDCG@5 System Evolution) ===")
    print("Observing NDCG@5 as the platform grows from 0 users to 500 users.")
    
    history_sizes = [0, 10, 50, 499]
    header_cold = f"{'Model':<28} | {'0 Users':>8} | {'10 Users':>8} | {'50 Users':>8} | {'499 Users':>9}"
    print(header_cold)
    print("-" * len(header_cold))

    # We will test the three main competitors across time
    cold_models = [
        ("ALS Matrix Factorization", None), 
        ("Content-Only (λ_u=0)",    RecommenderEngine(alpha=0.8, beta=0.6, gamma=0.0, k_target=25, lambda_max=0.0)),
        ("CF-Only (λ_u=1)",         RecommenderEngine(alpha=0.4, beta=0.6, gamma=0.0, k_target=25, lambda_max=1.0)),
        ("Switching Hybrid (Bool)", RecommenderEngine(alpha=0.4, beta=0.6, gamma=0.0, k_target=25, lambda_max=1.0, hard_switch=True)),
        ("Adaptive Hybrid (Ours)",  RecommenderEngine(alpha=0.4, beta=0.6, gamma=0.0, k_target=25, lambda_max=1.0))
    ]

    for name, base_rec in cold_models:
        row_str = f"{name:<28} |"
        for size in history_sizes:
            limited_users_df = users_df.head(size) if size > 0 else pd.DataFrame(columns=users_df.columns)
            
            # We must instantiate a fresh ALS wrapper each loop so it doesn't cache the old data
            rec = ALSRecommenderWrapper() if name == "ALS Matrix Factorization" else base_rec
            
            # We only care about NDCG for this matrix
            _, _, n = run_evaluation(rec, df, test_users_df, limited_users_df, interactions, k)
            row_str += f" {n:>8.4f} |"
        print(row_str)

    print("-" * len(header_cold))
    print("\nEvaluation complete.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") # Suppress OpenBLAS thread warning
    evaluate(k=5)