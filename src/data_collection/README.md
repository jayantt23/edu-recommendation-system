# School Recommendation System - Data Collection Agent

This folder contains the AI-powered researcher designed to collect unstructured school data (mission statements, curriculum, culture) to feed into the recommendation system's LDA Topic Model.

## 🚀 Step-by-Step Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Data (Run ONCE)
This script merges location data into a single input file for the agent:
```bash
python src/data_collection/prepare_agent_data.py
```
This creates `data/agent_input.csv`.

### 3. API Keys
Create a `.env` file in the root directory with:
```env
GEMINI_API_KEY=your_key
SERPER_API_KEY=your_key
```

### 4. Run the Agent (3 Teammates)
Each teammate runs their assigned "slice" of the 9,000 schools using the same input file:

- **Teammate 1:** `python src/data_collection/gemini_brochure_agent.py --part 1`
- **Teammate 2:** `python src/data_collection/gemini_brochure_agent.py --part 2`
- **Teammate 3:** `python src/data_collection/gemini_brochure_agent.py --part 3`

The script automatically calculates which 3,000 schools you should process.

## 📁 Output
Results are saved as `.txt` files in `data/raw/brochures/`.
- **Format:** Prose text block (~300 words).
- **Goal:** Optimized for Latent Dirichlet Allocation (LDA) to extract cultural topics ($\theta_s$).
