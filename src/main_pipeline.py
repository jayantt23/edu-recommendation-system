import os
import sys
import pandas as pd
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp_pipeline.lda_trainer import LDATrainer
from src.modeling.recommender import RecommenderEngine
from src.data_collection.generate_synthetic_brochures import generate_brochure_from_metrics

N_TOPICS  = 5
DATA_PATH = "data/agent_input.csv"
OUT_DIR   = "data/processed"
MODEL_DIR = "models"


def run_pipeline():
    # ── 1. Load structured school data ──────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print("Run: python src/data_collection/prepare_agent_data.py")
        print("Then: python src/data_collection/enrich_synthetic_fields.py")
        return

    df = pd.read_csv(DATA_PATH, dtype={"ncessch": str})
    df["ncessch"] = df["ncessch"].str.zfill(12)
    print(f"Loaded {len(df)} schools from {DATA_PATH}")

    # Check if enriched fields exist
    has_enriched = all(c in df.columns for c in
                       ["stem_score", "norm_arts", "norm_ap_count", "norm_world_lang", "is_magnet"])
    if has_enriched:
        print("✅ Enriched fields detected (stem_score, norm_arts, norm_ap_count, norm_world_lang, is_magnet)")
    else:
        print("⚠️  Enriched fields NOT found — using fallback proxies.")
        print("   Run: python src/data_collection/enrich_synthetic_fields.py")

    # Fill any missing metric columns with neutral defaults
    defaults = {
        "norm_enrollment":              0.5,
        "norm_total_athletes":          0.0,
        "norm_student_teacher_ratio":   0.5,
        "stem_score":                   0.3,
        "norm_arts":                    0.3,
        "norm_ap_count":                0.2,
        "norm_world_lang":              0.2,
        "is_magnet":                    0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    # ── 2. Generate synthetic brochures per school ───────────────────────────
    print("Generating synthetic brochures from school metrics...")

    documents  = []
    school_ids = []

    for _, row in df.iterrows():
        brochure = generate_brochure_from_metrics(
            school_name      = str(row.get("school_name", "")),
            norm_enrollment  = float(row.get("norm_enrollment", 0.5)),
            norm_athletes    = float(row.get("norm_total_athletes", 0.0)),
            norm_str         = float(row.get("norm_student_teacher_ratio", 0.5)),
            stem_score       = float(row.get("stem_score", 0.3))    if has_enriched else None,
            norm_arts        = float(row.get("norm_arts", 0.3))     if has_enriched else None,
            norm_ap_count    = float(row.get("norm_ap_count", 0.2)) if has_enriched else None,
            norm_world_lang  = float(row.get("norm_world_lang", 0.2)) if has_enriched else None,
            is_magnet        = int(row.get("is_magnet", 0))         if has_enriched else 0,
        )
        documents.append(brochure)
        school_ids.append(row["ncessch"])

    print(f"Generated {len(documents)} synthetic brochures.")

    # ── 3. Train LDA ─────────────────────────────────────────────────────────
    trainer = LDATrainer(n_topics=N_TOPICS)
    trainer.train(documents)
    os.makedirs(MODEL_DIR, exist_ok=True)
    trainer.save_model(MODEL_DIR)

    # ── 4. Infer θ_s for every school ────────────────────────────────────────
    print("Inferring topic distributions θ_s for all schools...")
    theta_matrix = trainer.transform(documents)    # shape (N, N_TOPICS)
    theta_dict   = {sid: theta_matrix[i] for i, sid in enumerate(school_ids)}
    df["theta_s"] = df["ncessch"].apply(
        lambda sid: theta_dict.get(sid, np.full(N_TOPICS, 1.0 / N_TOPICS))
    )

    # ── 5. Compute composite score_met ───────────────────────────────────────
    # Weighted combination of all available quality signals
    if has_enriched:
        df["score_met"] = (
            0.25 * df["norm_ap_count"]                      +   # academic depth
            0.20 * df["norm_total_athletes"]                +   # sports richness
            0.15 * df["norm_arts"]                          +   # arts richness
            0.15 * df["stem_score"]                         +   # STEM offerings
            0.15 * (1.0 - df["norm_student_teacher_ratio"]) +   # class size quality
            0.10 * df["is_magnet"].astype(float)                # specialised school bonus
        )
        print("score_met computed using: AP count, Athletes, Arts, STEM, Class size, Magnet")
    else:
        # Fallback: old formula
        df["score_met"] = (
            0.4 * df["norm_enrollment"] +
            0.3 * df["norm_total_athletes"] +
            0.3 * (1.0 - df["norm_student_teacher_ratio"])
        )
        print("score_met computed using fallback formula (enrollment + athletes + ratio)")

    # ── 6. Save hybrid dataset ────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "final_hybrid_dataset.pkl")
    df.to_pickle(out_path)
    print(f"Pipeline complete! Saved {len(df)} schools → {out_path}")
    print(f"  score_met range: [{df['score_met'].min():.3f}, {df['score_met'].max():.3f}]")


if __name__ == "__main__":
    run_pipeline()
