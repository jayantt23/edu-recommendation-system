import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp_pipeline.lda_trainer import LDATrainer
from src.modeling.recommender import RecommenderEngine

N_TOPICS      = 5
DATA_PATH     = "data/processed/final_hybrid_dataset.pkl"
MODEL_DIR     = "models"
USERS_PATH    = "data/processed/synthetic_users.pkl"
INTERACT_PATH = "data/processed/synthetic_interactions.pkl"

# ── Recommender hyperparameters ───────────────────────────────────────────────
# gamma=0   : geo penalty removed — distance shown for info only, not penalised
# k_target=30: lambda_u = 0.8 × (n_neighbors/30) → ramps up slowly as CF data grows
#              New/cold-start users with wu=None → lambda_u=0 → pure content
ALPHA      = 0.8   # topic similarity weight
BETA       = 0.2   # school quality (score_met) weight
GAMMA      = 0.0   # geo penalty DISABLED
K_TARGET   = 30    # neighbours needed for full CF confidence (dynamic lambda_u)
LAMBDA_MAX = 0.8   # max CF weight (only hit when ≥30 similar neighbours found)


def get_recommendations(query, lat, lon, top_k=5, new_user=False, radius_km=50.0):
    # ── 1. Load hybrid dataset ────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print("Run src/main_pipeline.py first!")
        return

    df = pd.read_pickle(DATA_PATH)

    # ── 2. Get query topic distribution θ_q ──────────────────────────────────
    trainer   = LDATrainer(n_topics=N_TOPICS)
    has_model = os.path.exists(os.path.join(MODEL_DIR, "lda_model.pkl"))

    if has_model:
        trainer.load_model(MODEL_DIR)
        theta_q = trainer.transform(query)[0]
        
        best_topic_idx = np.argmax(theta_q)
        feature_names = trainer.vectorizer.get_feature_names_out()
        top_words = [feature_names[i] for i in trainer.lda.components_[best_topic_idx].argsort()[:-6:-1]]
        
        print(f"θ_q (LDA vector) →  {np.round(theta_q, 2)}")
        print(f"LDA Matched Context → {', '.join(top_words)}")
    else:
        theta_q = np.full(N_TOPICS, 1.0 / N_TOPICS)
        print("No LDA model → using uniform θ_q")

    # ── 3. CF setup — cold start aware ────────────────────────────────────────
    historical_users_df = None
    interactions        = None
    wu                  = None   # None → lambda_u = 0 (cold start)

    if new_user:
        print("Mode: COLD START — new user, no history. Using pure content-based scoring.")
    elif os.path.exists(USERS_PATH) and os.path.exists(INTERACT_PATH):
        historical_users_df = pd.read_pickle(USERS_PATH)
        with open(INTERACT_PATH, "rb") as f:
            interactions = pickle.load(f)
        wu = theta_q   # use query vector as user preference proxy
        print(f"Mode: HYBRID — {len(historical_users_df)} historical users loaded.")
        print(f"       k_target={K_TARGET}, λ_max={LAMBDA_MAX} → λ_u ramps 0→{LAMBDA_MAX} as neighbours grow")
    else:
        print("No synthetic users found → content-only mode.")
        print("Run: python src/data_collection/generate_synthetic_users.py")

    # ── 3.5 Apply Hard Radius Filter ──────────────────────────────────────────
    # Since gamma=0 (no geo penalty), we must filter out schools that are too far.
    recommender = RecommenderEngine(
        alpha      = ALPHA,
        beta       = BETA,
        gamma      = GAMMA,       # geo penalty off
        k_target   = K_TARGET,    # dynamic lambda_u
        lambda_max = LAMBDA_MAX,
    )
    
    print(f"Filtering schools within {radius_km} km of ({lat}, {lon})...")
    df['dist_km'] = df.apply(lambda r: recommender.haversine(lat, lon, r['latitude'], r['longitude']), axis=1)
    df = df[df['dist_km'] <= radius_km].copy()
    
    if len(df) == 0:
        print(f"No schools found within {radius_km} km!")
        return

    # ── 4. Run recommender ────────────────────────────────────────────────────
    results = recommender.get_recommendations(
        user_query_theta    = theta_q,
        user_loc            = (lat, lon),
        candidate_df        = df,
        wu                  = wu,
        historical_users_df = historical_users_df,
        interactions        = interactions,
    )

    # ── 5. Display ────────────────────────────────────────────────────────────
    mode_label = "Cold Start (Content Only)" if new_user else "Hybrid (Content + CF)"
    print(f"\nTop {top_k} Recommendations — '{query}'")
    print(f"Location: ({lat}, {lon})   |   Mode: {mode_label}")
    print(f"Weights: α(topic)={ALPHA}  β(quality)={BETA}  γ(geo)={GAMMA}")
    print("-" * 100)
    print(f"{'School Name':<38} | {'Final':>7} | {'Content':>7} | {'CF':>6} | {'λ_u':>5} | {'Dist km':>8}")
    print("-" * 100)

    for _, row in results.head(top_k).iterrows():
        lam = row["lambda_u"]
        # Flag cold start clearly
        lam_str = f"{lam:.3f}" if lam > 0 else " 0.000"
        print(
            f"{str(row.get('school_name', 'Unknown'))[:38]:<38} | "
            f"{row['final_score']:>7.4f} | "
            f"{row['content_utility']:>7.4f} | "
            f"{row['cf_score']:>6.4f} | "
            f"{lam_str:>5} | "
            f"{row['distance']:>8.1f}"
        )

    # λ_u summary
    avg_lam = results.head(top_k)["lambda_u"].mean()
    if avg_lam == 0:
        print("\n  λ_u = 0.000 → Cold start: 100% content-based, 0% CF")
    elif avg_lam < 0.3:
        print(f"\n  λ_u = {avg_lam:.3f} → Few neighbours found: content-dominant")
    elif avg_lam < 0.6:
        print(f"\n  λ_u = {avg_lam:.3f} → Moderate neighbours: balanced hybrid")
    else:
        print(f"\n  λ_u = {avg_lam:.3f} → Many neighbours: CF-dominant")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="School Recommender CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Returning user (hybrid CF + content):
  python src/recommender_cli.py --query "STEM robotics coding" --lat 34.05 --lon -118.24

  # New user / cold start (pure content, no CF):
  python src/recommender_cli.py --query "arts music theater" --lat 40.71 --lon -74.00 --new_user

  # Show top 10:
  python src/recommender_cli.py --query "sports athletics" --lat 41.87 --lon -87.62 --top_k 10
        """
    )
    parser.add_argument("--query",    type=str,   required=True, help="What you're looking for")
    parser.add_argument("--lat",      type=float, default=34.05, help="Your latitude")
    parser.add_argument("--lon",      type=float, default=-118.24, help="Your longitude")
    parser.add_argument("--top_k",   type=int,   default=5,     help="Number of results")
    parser.add_argument("--radius",  type=float, default=50.0,  help="Max distance in km")
    parser.add_argument("--new_user", action="store_true",       help="Cold start: skip CF, pure content")
    args = parser.parse_args()

    get_recommendations(args.query, args.lat, args.lon, args.top_k, args.new_user, args.radius)
