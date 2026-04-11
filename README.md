# Adaptive Hybrid Recommendation System for Educational Institution Selection

This project implements an **Adaptive Hybrid Recommendation System** designed to assist families in selecting the right K-12 educational institutions. It addresses the data sparsity and cold-start problems inherent in the education domain by fusing structured administrative datasets with unstructured textual content (school brochures and prospectuses).

## 🚀 Project Overview

The system transitions dynamically from a **zero-shot content-based utility model** to a **preference-driven collaborative filtering engine** using an adaptive confidence factor.

### Key Innovations:
- **Multi-Modal Feature Representation:** Fuses administrative metrics ($M_s$) with latent topic distributions ($\theta_s$) generated via Latent Dirichlet Allocation (LDA).
- **Explainable AI (XAI):** Provides transparent "taste profiles" for schools, allowing users to understand the reasoning behind recommendations (e.g., focus on STEM, Athletics, or Performing Arts).
- **Cold-Start Mitigation:** Employs zero-shot preference elicitation to build initial user profiles, bypassing the need for dense historical interaction matrices.

## 💻 Technical Implementation: Codebase & Pipeline

The system is built as a modular Python pipeline that transforms raw administrative data and school brochures into a searchable, hybrid recommendation engine.

### 1. NLP & Text Preprocessing (`src/nlp_pipeline/`)
To process unstructured school brochures, we implemented a custom NLP stack:
- **`preprocessor.py`:** Uses **NLTK** to clean raw text. It performs lowercase conversion, removes non-alphabetic characters, and applies **Lemmatization** and stop-word removal (including domain-specific terms like "school" or "education").
- **`lda_trainer.py`:** Wraps **Scikit-Learn's Latent Dirichlet Allocation**. It vectorizes the cleaned text using a `CountVectorizer` and trains an LDA model to discover latent themes (e.g., STEM, Arts, Athletics). It saves the trained models as `.pkl` files for inference.

### 2. Recommendation Logic (`src/modeling/`)
The core engine is encapsulated in the `RecommenderEngine` class within **`recommender.py`**:
- **Content Utility:** Calculates similarity between a user's query and a school's "Taste Profile" using **Jensen-Shannon Divergence**.
- **Spatial Awareness:** Uses the **Haversine formula** to calculate the physical distance (in km) between the user and the school, applying a distance-based penalty to the final score.
- **Hybrid Scoring:** Combines the content score with a **Collaborative Filtering (CF)** component. It uses an **Adaptive Confidence Factor** ($\lambda_u$) to decide how much to trust CF data versus the Content model.

### 3. The Unified Pipeline (`src/main_pipeline.py`)
This is the "brain" of the system that ties everything together:
- **Data Loading:** Reads the merged school directory from `data/agent_input.csv`.
- **Model Training:** Automatically detects available brochures in `data/raw/brochures/`. If brochures exist, it trains the LDA model; if not, it applies a **Uniform Distribution Fallback** so the system still works using structured data.
- **Dataset Generation:** Merges the LDA "Taste Profiles" with administrative metrics to create a final, high-performance pickled dataset (`final_hybrid_dataset.pkl`).

### 4. Search & Inference (`src/recommender_cli.py`)
A user-facing CLI tool that performs the following steps:
- **Path Resolution:** Includes internal `sys.path` handling to ensure the `src` module is discoverable from any directory.
- **Query Processing:** Transforms a natural language query (e.g., "arts and theater") into a topic distribution.
- **Top-K Ranking:** Scores all schools in the dataset and returns the Top 5 matches based on the hybrid model.

### 5. Development Utilities
- **`generate_synthetic_brochures.py`:** A utility to generate dummy brochure data based on templates (STEM, Arts, Athletics). This allows you to test the full NLP pipeline immediately without waiting for the Gemini scraper to finish.

## ✅ Implementation Status
...
...

## 🛠️ How to Run

### 1. Prerequisites
Ensure you are using the virtual environment:
```bash
# Activate virtual environment
recsys\Scripts\activate
```

### 2. Initialize the System
If you don't have brochure data yet, you can still run the pipeline. It will use structured metrics as a primary signal:
```bash
python src/main_pipeline.py
```
*Note: If you want to test the NLP components specifically, run `python src/data_collection/generate_synthetic_brochures.py` first.*

### 3. Get Recommendations
Run the CLI from the project root. The scripts now include internal path handling to resolve import errors:
```bash
python src/recommender_cli.py --query "STEM and robotics" --lat 34.49 --lon -118.21
```

## 📂 Project Structure
- `src/nlp_pipeline/`: Text processing and LDA modeling.
- `src/modeling/`: Hybrid recommendation engine logic.
- `src/data_collection/`: API clients, processors, and brochure management.
- `data/processed/`: Final hybrid feature vectors.
