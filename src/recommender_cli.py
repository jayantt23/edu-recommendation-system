import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp_pipeline.lda_trainer import LDATrainer
from src.modeling.recommender import RecommenderEngine

N_TOPICS  = 5
DATA_PATH = "data/processed/final_hybrid_dataset.pkl"
MODEL_DIR = "models"
USERS_PATH       = "data/processed/synthetic_users.pkl"
INTERACT_PATH    = "data/processed/synthetic_interactions.pkl"


def get_recommendations(query, lat, lon, top_k=5):
    # ── 1. Load hybrid dataset ──────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print("Run src/main_pipeline.py first!"); return

    df = pd.read_pickle(DATA_PATH)

    # ── 2. Get query topic distribution θ_q ─────────────────────────────────
    trainer  = LDATrainer(n_topics=N_TOPICS)
    has_model = os.path.exists(os.path.join(MODEL_DIR, "lda_model.pkl"))

    if has_model:
        trainer.load_model(MODEL_DIR)
        theta_q = trainer.transform(query)[0]
        print(f"θ_q from LDA: {np.round(theta_q, 3)}")
    else:
        theta_q = np.full(N_TOPICS, 1.0/N_TOPICS)
        print("No LDA model → using uniform θ_q")

    # ── 3. Load synthetic users (for CF) ────────────────────────────────────
    historical_users_df = None
    interactions        = None
    wu                  = None  # current user preference vector

    if os.path.exists(USERS_PATH) and os.path.exists(INTERACT_PATH):
        historical_users_df = pd.read_pickle(USERS_PATH)
        with open(INTERACT_PATH, "rb") as f:
            interactions = pickle.load(f)
        # Treat θ_q as the current user's preference vector
        wu = theta_q
        print(f"CF enabled: {len(historical_users_df)} historical users loaded.")
    else:
        print("No synthetic users found. Running content-only mode. (Run generate_synthetic_users.py)")

    # ── 4. Run recommender ──────────────────────────────────────────────────
    recommender = RecommenderEngine(alpha=0.5, beta=0.3, gamma=0.2, k_target=5, lambda_max=0.8)
    results = recommender.get_recommendations(
        user_query_theta    = theta_q,
        user_loc            = (lat, lon),
        candidate_df        = df,
        wu                  = wu,
        historical_users_df = historical_users_df,
        interactions        = interactions,
    )

    # ── 5. Display ──────────────────────────────────────────────────────────
    print(f"\nTop {top_k} Recommendations for: '{query}'")
    print(f"Location: ({lat}, {lon})")
    print("-" * 90)
    print(f"{'School Name':<35} | {'Final':>7} | {'Content':>7} | {'CF':>6} | {'λ_u':>5} | {'Dist km':>7}")
    print("-" * 90)

    for _, row in results.head(top_k).iterrows():
        print(
            f"{str(row.get('school_name','Unknown'))[:35]:<35} | "
            f"{row['final_score']:>7.4f} | "
            f"{row['content_utility']:>7.4f} | "
            f"{row['cf_score']:>6.4f} | "
            f"{row['lambda_u']:>5.3f} | "
            f"{row['distance']:>7.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="School Recommender")
    parser.add_argument("--query", type=str, required=True, help="e.g. 'STEM robotics'")
    parser.add_argument("--lat",   type=float, default=34.05)
    parser.add_argument("--lon",   type=float, default=-118.24)
    parser.add_argument("--top_k", type=int,   default=5)
    args = parser.parse_args()

    get_recommendations(args.query, args.lat, args.lon, args.top_k)
