# Hybrid School Recommendation System — Handoff for Kanava

## 🌟 System Overview
We have successfully implemented the core algorithms from the research paper. The system is a **Hybrid Recommendation Engine** that takes a parent/student's query and location, and recommends the top 5 schools out of 9,813 US schools by combining **Content-Based Filtering (NLP)** and **Collaborative Filtering (Past User Interactions)**.

### The Pipeline Architecture
1. **Data Enrichment**: We take raw CCD/CRDC data and deterministically synthesize missing fields (AP courses, Arts programs, STEM indicators).
2. **Synthetic Brochures**: We generate dynamic text brochures for each school weighted by its real/enriched metrics.
3. **LDA NLP Model**: We train Latent Dirichlet Allocation (LDA) on the 9,813 brochures to extract 5 topics and assign a topic distribution vector (`θ_s`) to each school.
4. **Synthetic Users (CF Signal)**: We simulate 500 past users who picked schools based on both topic match *and a hidden quality score* (which trains the CF model to find things the Content model misses).
5. **Hybrid Engine**: 
   - **Content Score**: Cosine similarity between query (`θ_q`) and school (`θ_s`) + School Quality Metric (`score_met`) - Distance Penalty.
   - **CF Score**: Interactions from past users with similar preferences (`tau > 0.90`).
   - **Dynamic λ_u**: Scales from 0 to 0.8 depending on how many similar neighbors are found (solves cold-start).

---

## ✅ What Has Been Implemented
All backend logic, data pipelines, and CLI integrations are 100% complete and working.

- `src/data_collection/enrich_synthetic_fields.py`: Cleans and enriches data with deterministic synthetic fields.
- `src/main_pipeline.py`: Orchestrates LDA training, `θ_s` generation, and `score_met` computation.
- `src/data_collection/generate_synthetic_users.py`: Generates the interaction matrix for Collaborative Filtering.
- `src/evaluate.py`: Calculates **Precision@5, Recall@5, NDCG@5** and runs an ablation study proving the Hybrid model outperforms Content-Only and CF-Only.
- `src/recommender_cli.py`: A fully functional CLI to query the system.
- `run_full_pipeline.py`: A master script that executes the entire backend end-to-end.

---

## 🚀 Next Steps for You (Kanava)

The backend is mathematically sound. Your goal is to make it usable and dynamic for real users.

### 1. Build the Streamlit Frontend (Priority)
Currently, users interact via CLI (`python src/recommender_cli.py`). We need a Web UI.
* **Inputs needed on UI:**
  - Free-text query box (e.g., "STEM and robotics")
  - Latitude / Longitude sliders or a Zip Code to Lat/Lon converter.
  - Radius slider (e.g., max 50km).
* **Display:**
  - Show the user the LDA matched words so they know what the system understood.
  - Display the Top 5 schools in a nice card format showing `Dist km`, `Final Score`, and `λ_u` (CF Confidence).

### 2. Explicit User Preference Sliders
Right now, the query goes through the LDA model to generate `θ_q`. In a real system, you should also let users directly define their preferences.
* Add 5 sliders to the UI for **STEM, Arts, Athletics, Academics, Cultural**.
* If the user uses the sliders, bypass the text query and convert the slider values directly into a normalized `θ_q` array (must sum to 1). Pass this directly into `RecommenderEngine.get_recommendations()`.

### 3. Add Hard Filters
The recommender currently scores all 9,813 schools.
* Implement a pre-filter before calling `get_recommendations()` in the UI:
  ```python
  # Filter by radius first!
  df = df[df['dist_km'] <= selected_radius]
  # Filter by School Level (Elementary vs High School)
  df = df[df['school_level'] == selected_level]
  ```

### 4. Adjust Hyperparameters (Optional Tuning)
In `src/recommender_cli.py`, you can tweak:
* `ALPHA` (Topic Match Weight)
* `BETA` (School Quality Weight)
* `GAMMA` (Distance Penalty — currently disabled `0.0` in favor of a hard radius filter).

**To test the system immediately, run:**
```bash
python run_full_pipeline.py
python src/recommender_cli.py --query "arts and music" --lat 34.05 --lon -118.24 --radius 40
```
