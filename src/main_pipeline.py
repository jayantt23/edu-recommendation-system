import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# Add project root to sys.path to resolve 'src' imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp_pipeline.lda_trainer import LDATrainer
from src.modeling.recommender import RecommenderEngine

def run_pipeline():
    # 1. Load data
    INPUT_PATH = "data/agent_input.csv"
    BROCHURE_DIR = "data/raw/brochures"
    OUT_DIR = "data/processed"
    MODEL_DIR = "models"
    
    if not os.path.exists(INPUT_PATH):
        print("Error: Input CSV not found!")
        return
        
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Collect brochures that exist
    documents = []
    school_ids = []
    
    if os.path.exists(BROCHURE_DIR):
        print("Collecting available brochures...")
        for idx, row in df.iterrows():
            sid = str(row['ncessch']).zfill(12)
            path = os.path.join(BROCHURE_DIR, f"{sid}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
                    school_ids.append(sid)
    
    # 3. Handle LDA Training
    n_topics = 5 
    trainer = LDATrainer(n_topics=n_topics)
    
    if documents:
        print(f"Found {len(documents)} brochures. Training LDA...")
        trainer.train(documents)
        trainer.save_model(MODEL_DIR)
        theta_s = trainer.transform(documents)
        theta_dict = {sid: dist for sid, dist in zip(school_ids, theta_s)}
    else:
        print("No brochures found. Proceeding with uniform topic distributions (Content-Only fallback).")
        theta_dict = {}
        # Save a dummy vectorizer/model if they don't exist to prevent CLI errors
        if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    # 4. Merge results
    # For schools without brochures, we assign a uniform distribution
    uniform_dist = np.full(n_topics, 1.0/n_topics)
    
    def get_theta(ncessch):
        sid = str(ncessch).zfill(12)
        return theta_dict.get(sid, uniform_dist)

    print("Finalizing hybrid dataset...")
    df['theta_s'] = df['ncessch'].apply(get_theta)
    
    # 5. Save final dataset
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    df.to_pickle(os.path.join(OUT_DIR, "final_hybrid_dataset.pkl"))
    print(f"Pipeline complete! Saved {len(df)} records.")

if __name__ == "__main__":
    run_pipeline()
