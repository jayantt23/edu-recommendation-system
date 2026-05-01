import os
import sys
import pandas as pd
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp_pipeline.lda_trainer import LDATrainer
from src.modeling.recommender import RecommenderEngine
from src.data_collection.generate_synthetic_brochures import generate_brochure_from_metrics

N_TOPICS = 5
DATA_PATH = "data/agent_input.csv"
OUT_DIR   = "data/processed"
MODEL_DIR = "models"


def run_pipeline():
    # ── 1. Load structured school data ──────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run ccd_processor → crdc_processor → prepare_agent_data first.")
        return

    df = pd.read_csv(DATA_PATH, dtype={'ncessch': str})
    df['ncessch'] = df['ncessch'].str.zfill(12)
    print(f"Loaded {len(df)} schools from {DATA_PATH}")

    # ── 2. Generate synthetic brochures for every school ────────────────────
    print("Generating synthetic brochures from school metrics...")

    # Fill missing metric columns with neutral defaults
    for col in ['norm_enrollment', 'norm_total_athletes', 'norm_student_teacher_ratio']:
        if col not in df.columns:
            df[col] = 0.5

    documents  = []
    school_ids = []

    for _, row in df.iterrows():
        brochure = generate_brochure_from_metrics(
            norm_enrollment = float(row.get('norm_enrollment', 0.5)),
            norm_athletes   = float(row.get('norm_total_athletes', 0.0)),
            norm_str        = float(row.get('norm_student_teacher_ratio', 0.5)),
        )
        documents.append(brochure)
        school_ids.append(row['ncessch'])

    print(f"Generated {len(documents)} synthetic brochures.")

    # ── 3. Train LDA ─────────────────────────────────────────────────────────
    trainer = LDATrainer(n_topics=N_TOPICS)
    trainer.train(documents)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    trainer.save_model(MODEL_DIR)

    # ── 4. Infer θ_s for every school ───────────────────────────────────────
    print("Inferring topic distributions θ_s for all schools...")
    theta_matrix = trainer.transform(documents)   # shape (N, N_TOPICS)
    theta_dict   = {sid: theta_matrix[i] for i, sid in enumerate(school_ids)}

    df['theta_s'] = df['ncessch'].apply(lambda sid: theta_dict.get(sid, np.full(N_TOPICS, 1.0/N_TOPICS)))

    # ── 5. Compute composite Score_met ───────────────────────────────────────
    # Score_met = weighted combo of normalized M_s features (Eq 2 in paper)
    w_enroll  = 0.4
    w_athlete = 0.3
    w_str     = 0.3  # lower student-teacher ratio is better → use (1 - norm_str)

    df['score_met'] = (
        w_enroll  * df['norm_enrollment'] +
        w_athlete * df['norm_total_athletes'] +
        w_str     * (1.0 - df['norm_student_teacher_ratio'])
    )

    # ── 6. Save hybrid dataset ───────────────────────────────────────────────
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    out_path = os.path.join(OUT_DIR, "final_hybrid_dataset.pkl")
    df.to_pickle(out_path)
    print(f"Pipeline complete! Saved {len(df)} schools → {out_path}")


if __name__ == "__main__":
    run_pipeline()
