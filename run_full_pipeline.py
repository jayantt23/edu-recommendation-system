"""
run_full_pipeline.py
====================
Master script that runs the ENTIRE system end-to-end:
  1. main_pipeline.py      → build hybrid dataset (θ_s + score_met)
  2. generate_synthetic_users.py → build CF interaction matrix
  3. evaluate.py           → Precision@K, Recall@K, NDCG@K

Run from project root:
    python run_full_pipeline.py
"""
import subprocess
import sys
import time

STEPS = [
    ("Building hybrid dataset (LDA + θ_s + score_met)",
     [sys.executable, "src/main_pipeline.py"]),
    ("Generating synthetic users + preference-aligned interactions",
     [sys.executable, "src/data_collection/generate_synthetic_users.py"]),
    ("Running evaluation (Precision@5, Recall@5, NDCG@5)",
     [sys.executable, "src/evaluate.py"]),
]

def run_step(label, cmd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed (exit code {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    print(f"  Done in {elapsed:.1f}s")

if __name__ == "__main__":
    print("\nEdu-Recommendation System — Full Pipeline")
    print("=========================================")
    for label, cmd in STEPS:
        run_step(label, cmd)
    print("\n✅ Full pipeline complete.")
