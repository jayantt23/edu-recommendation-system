import os
import sys
import pandas as pd
import numpy as np
import argparse

# Add project root to sys.path to resolve 'src' imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp_pipeline.lda_trainer import LDATrainer
from src.modeling.recommender import RecommenderEngine

def get_recommendations(query, lat, lon):
    # 1. Load data and models
    DATA_PATH = "data/processed/final_hybrid_dataset.pkl"
    MODEL_DIR = "models"
    
    if not os.path.exists(DATA_PATH):
        print("Run src/main_pipeline.py first!")
        return
        
    df = pd.read_pickle(DATA_PATH)
    
    trainer = LDATrainer()
    has_model = os.path.exists(os.path.join(MODEL_DIR, "lda_model.pkl"))
    
    if has_model:
        trainer.load_model(MODEL_DIR)
        print(f"Processing query using LDA model: '{query}'")
        theta_q = trainer.transform(query)[0]
    else:
        print("No LDA model found (no brochure data). Using uniform distribution for query.")
        n_topics = len(df.iloc[0]['theta_s']) if len(df) > 0 else 5
        theta_q = np.full(n_topics, 1.0/n_topics)
    
    recommender = RecommenderEngine()
    
    # 2. Get recommendations
    user_loc = (lat, lon)
    results = recommender.get_recommendations(theta_q, user_loc, df)
    
    # 3. Display results
    print(f"\nTop 5 Recommendations for '{query}':")
    print("-" * 80)
    print(f"{'School Name':<30} | {'Final':<8} | {'Topic Sim':<10} | {'Admin':<8} | {'Dist (km)':<8}")
    print("-" * 80)
    
    for _, row in results.head(5).iterrows():
        # Re-calculating components for display (using the engine's internal weights)
        # alpha=0.5, beta=0.3, gamma=0.2 (defaults)
        sim_jsd = recommender.get_jsd_similarity(theta_q, row.get('theta_s', [0.2]*5))
        admin_score = row.get('norm_enrollment', 0.5)
        dist = row['distance']
        
        print(f"{row['school_name'][:30]:<30} | {row['final_score']:<8.4f} | {sim_jsd:<10.4f} | {admin_score:<8.4f} | {dist:<8.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Your search query (e.g., 'STEM and athletics')")
    parser.add_argument("--lat", type=float, default=34.5, help="Your latitude")
    parser.add_argument("--lon", type=float, default=-118.2, help="Your longitude")
    
    args = parser.parse_args()
    get_recommendations(args.query, args.lat, args.lon)
