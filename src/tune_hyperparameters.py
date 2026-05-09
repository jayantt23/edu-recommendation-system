"""
tune_hyperparameters.py
=======================
Grid Search script to find the optimal alpha, beta, and k_target values.
Runs two phases:
  1. Content-Only Tuning (finds the best balance of Text vs Quality).
  2. Adaptive Hybrid Tuning (finds the best CF transition curve).
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modeling.recommender import RecommenderEngine

DATA_PATH  = "data/processed/final_hybrid_dataset.pkl"
USERS_PATH = "data/processed/synthetic_users.pkl"
INTER_PATH = "data/processed/synthetic_interactions.pkl"

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

def tune():
    print("Loading data for Hyperparameter Tuning...")
    df       = pd.read_pickle(DATA_PATH)
    users_df = pd.read_pickle(USERS_PATH)
    with open(INTER_PATH, "rb") as f:
        interactions = pickle.load(f)

    # Use a smaller sample for faster tuning
    test_users = users_df.sample(n=30, random_state=42)
    
    # Expanded Grid for maximum optimization
    alphas = [0.2, 0.4, 0.6, 0.8]       # Topic Match weight
    betas  = [0.2, 0.4, 0.6, 0.8]       # School Quality weight
    k_targets = [10, 25, 30, 50]        # Neighbors needed for full CF trust
    
    # =======================================================
    # PHASE 1: CONTENT-ONLY TUNING
    # =======================================================
    print(f"\n--- PHASE 1: Tuning Content-Only ({len(alphas)*len(betas)} combinations) ---")
    content_results = []
    
    for a in alphas:
        for b in betas:
            # lambda_max=0.0 strictly forces Content-Only. k_target is ignored.
            rec = RecommenderEngine(alpha=a, beta=b, gamma=0.0, k_target=5, lambda_max=0.0)
            ndcgs = []
            
            for _, user in test_users.iterrows():
                uid = user['user_id']
                wu  = user['wv']
                rel_ids = interactions.get(uid, set())
                if not rel_ids: continue
                
                # historical_users_df not needed for pure content, but passed to satisfy API
                hist_df = users_df[users_df['user_id'] != uid]
                res = rec.get_recommendations(
                    user_query_theta=wu, user_loc=(34.0, -118.0), candidate_df=df,
                    wu=wu, historical_users_df=hist_df, interactions=interactions
                )
                
                recs = res['ncessch'].tolist()
                ndcgs.append(ndcg_at_k(recs, rel_ids, 5))
            
            content_results.append((np.mean(ndcgs), a, b))

    content_results.sort(reverse=True, key=lambda x: x[0])
    
    print("\n=== TOP 3: CONTENT-ONLY ===")
    print(f"{'Rank':<5} | {'NDCG@5':>7} | {'Alpha (Topic)':>13} | {'Beta (Quality)':>14}")
    print("-" * 50)
    for i, (ndcg, a, b) in enumerate(content_results[:3]):
        print(f"{i+1:<5} | {ndcg:>7.4f} | {a:>13.1f} | {b:>14.1f}")


    # =======================================================
    # PHASE 2: ADAPTIVE HYBRID TUNING
    # =======================================================
    print(f"\n--- PHASE 2: Tuning Adaptive Hybrid ({len(alphas)*len(betas)*len(k_targets)} combinations) ---")
    hybrid_results = []
    
    for a in alphas:
        for b in betas:
            for k_tgt in k_targets:
                # lambda_max=1.0 allows full dynamic transition
                rec = RecommenderEngine(alpha=a, beta=b, gamma=0.0, k_target=k_tgt, lambda_max=1.0)
                ndcgs = []
                
                for _, user in test_users.iterrows():
                    uid = user['user_id']
                    wu  = user['wv']
                    rel_ids = interactions.get(uid, set())
                    if not rel_ids: continue
                    
                    hist_df = users_df[users_df['user_id'] != uid]
                    res = rec.get_recommendations(
                        user_query_theta=wu, user_loc=(34.0, -118.0), candidate_df=df,
                        wu=wu, historical_users_df=hist_df, interactions=interactions
                    )
                    
                    recs = res['ncessch'].tolist()
                    ndcgs.append(ndcg_at_k(recs, rel_ids, 5))
                
                hybrid_results.append((np.mean(ndcgs), a, b, k_tgt))

    hybrid_results.sort(reverse=True, key=lambda x: x[0])
    
    print("\n=== TOP 3: ADAPTIVE HYBRID ===")
    print(f"{'Rank':<5} | {'NDCG@5':>7} | {'Alpha (Topic)':>13} | {'Beta (Quality)':>14} | {'K_Target':>9}")
    print("-" * 65)
    for i, (ndcg, a, b, k) in enumerate(hybrid_results[:3]):
        print(f"{i+1:<5} | {ndcg:>7.4f} | {a:>13.1f} | {b:>14.1f} | {k:>9}")

if __name__ == "__main__":
    tune()