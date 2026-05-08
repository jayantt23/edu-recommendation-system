"""
evaluate_diversity.py
=====================
Calculates Beyond-Accuracy metrics for the recommendation engine:
1. Catalog Coverage: What percentage of the total school database is actually recommended?
2. Intra-List Diversity (ILD): How diverse are the topics of the 5 schools recommended to a single user?
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modeling.recommender import RecommenderEngine

DATA_PATH  = "data/processed/final_hybrid_dataset.pkl"
USERS_PATH = "data/processed/synthetic_users.pkl"
INTER_PATH = "data/processed/synthetic_interactions.pkl"

def calculate_ild(recommended_thetas):
    """
    Intra-List Diversity (ILD) measures the average distance between all pairs
    of recommended items. Higher means the list is more diverse.
    """
    if len(recommended_thetas) <= 1:
        return 0.0
    
    distances = []
    for i in range(len(recommended_thetas)):
        for j in range(i + 1, len(recommended_thetas)):
            # Euclidean distance between topic vectors
            dist = np.linalg.norm(recommended_thetas[i] - recommended_thetas[j])
            distances.append(dist)
    return np.mean(distances)

def run_diversity_evaluation():
    print("Loading data for Diversity Evaluation...")
    df       = pd.read_pickle(DATA_PATH)
    users_df = pd.read_pickle(USERS_PATH)
    with open(INTER_PATH, "rb") as f:
        interactions = pickle.load(f)

    # Use 50 users to match the ablation study
    test_users = users_df.head(50)
    
    total_schools = len(df)
    
    configs = {
        "Content-Only": RecommenderEngine(alpha=0.8, beta=0.2, gamma=0.0, k_target=30, lambda_max=0.0),
        "Hybrid":       RecommenderEngine(alpha=0.8, beta=0.2, gamma=0.0, k_target=30, lambda_max=0.8),
    }
    
    print("\n=== BEYOND-ACCURACY METRICS ===")
    print(f"Total Candidate Schools: {total_schools}")
    print(f"{'Model':<15} | {'Catalog Coverage (%)':>20} | {'Intra-List Diversity':>20}")
    print("-" * 62)
    
    for name, rec in configs.items():
        all_recommended_ids = set()
        ild_scores = []
        
        for _, user in test_users.iterrows():
            uid = user['user_id']
            wu  = user['wv']
            
            hist_df = users_df[users_df['user_id'] != uid]
            res = rec.get_recommendations(
                user_query_theta=wu, user_loc=(34.0, -118.0), candidate_df=df,
                wu=wu, historical_users_df=hist_df, interactions=interactions
            )
            
            # Get top 5 recommendations and retrieve theta_s from the main df
            top_5 = res.head(5)
            top_5 = top_5.merge(df[['ncessch', 'theta_s']], on='ncessch', how='left')
            
            # 1. Update Coverage
            all_recommended_ids.update(top_5['ncessch'].tolist())
            
            # 2. Calculate ILD for this user
            thetas = np.vstack(top_5['theta_s'].values)
            ild_scores.append(calculate_ild(thetas))
            
        coverage_pct = (len(all_recommended_ids) / total_schools) * 100
        mean_ild = np.mean(ild_scores)
        
        print(f"{name:<15} | {coverage_pct:>19.2f}% | {mean_ild:>20.4f}")

if __name__ == "__main__":
    run_diversity_evaluation()
