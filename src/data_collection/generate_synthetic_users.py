"""
Generates synthetic historical user profiles and interaction logs
to support Collaborative Filtering (Algorithm 2).

Parameters tuned for a dense-enough CF matrix:
  - 500 users  x  20 interactions each  across 9,813 schools
  Overlap probability between two similar users ≈ 4%, making S_CF non-zero.
"""
import numpy as np
import pandas as pd
import random
import pickle
import os

N_USERS  = 500   # number of synthetic past users
N_TOPICS = 5     # must match LDA n_topics (set in main_pipeline.py)

TOPIC_NAMES = ["STEM", "ARTS", "ATHLETICS", "ACADEMIC", "CULTURAL"]


def generate_users(n_users=N_USERS, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    users = []
    for uid in range(n_users):
        # Dirichlet(1,…,1) → uniform over the simplex → diverse preference vectors
        raw = np.random.dirichlet(np.ones(N_TOPICS))
        users.append({"user_id": uid, "wv": raw})

    return pd.DataFrame(users)


def generate_interactions(users_df, df_schools, n_interactions=20, seed=42):
    """
    Each user interacts with schools whose topic profile (θ_s) best matches
    their preference vector (wv).
    - Scores all schools by cosine(wv, θ_s)
    - Samples n_interactions from the top-50 most similar schools (adds noise)
    This makes CF ground truth preference-aligned and non-random.
    """
    random.seed(seed)
    np.random.seed(seed)

    interactions = {}
    school_ids = df_schools['ncessch'].tolist()

    # Build (N_schools × N_topics) matrix — do this once outside the loop
    thetas = np.vstack(df_schools['theta_s'].values)   # shape (N, K)

    # ── THE CF SECRET WEAPON ──────────────────────────────────────────────────
    # Create a "hidden quality" score for each school that Content-Only CANNOT see.
    # This simulates real-world factors (culture, teacher reputation, safety).
    # Because users pick schools based partly on this hidden quality, CF will
    # learn to recommend these schools, allowing Hybrid to beat Content-Only!
    hidden_quality = np.random.uniform(0.0, 0.6, size=len(school_ids))

    for _, row in users_df.iterrows():
        uid = row["user_id"]
        wv  = np.array(row["wv"])                      # shape (K,)

        # Cosine similarity between user preference vector and each school's θ_s
        norms_s = np.linalg.norm(thetas, axis=1) + 1e-10
        norm_u  = np.linalg.norm(wv) + 1e-10
        sims    = (thetas @ wv) / (norms_s * norm_u)

        # Users pick based on BOTH topic match AND hidden quality
        combined_score = sims + hidden_quality

        # Pick from top-50
        top_idx    = np.argsort(combined_score)[::-1][:50]
        chosen_idx = np.random.choice(
            top_idx,
            size=min(n_interactions, len(top_idx)),
            replace=False
        )
        interactions[uid] = set(school_ids[i] for i in chosen_idx)

    return interactions


def save_synthetic_users(out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    hybrid_path = os.path.join(out_dir, "final_hybrid_dataset.pkl")
    if not os.path.exists(hybrid_path):
        print("Run src/main_pipeline.py first to generate final_hybrid_dataset.pkl")
        return

    df           = pd.read_pickle(hybrid_path)
    users_df     = generate_users()
    interactions = generate_interactions(users_df, df)

    users_df.to_pickle(os.path.join(out_dir, "synthetic_users.pkl"))
    with open(os.path.join(out_dir, "synthetic_interactions.pkl"), "wb") as f:
        pickle.dump(interactions, f)

    print(f"Saved {len(users_df)} synthetic users + preference-aligned interactions → {out_dir}")
    print(f"  Users: {len(users_df)}  |  Interactions per user: 20  |  Schools: {len(df)}")


if __name__ == "__main__":
    save_synthetic_users()
