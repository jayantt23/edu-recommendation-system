# 🚀 Gemini Brochure Ingestion Setup Guide

This guide covers how to run the `gemini_brochure_agent.py` script to fetch and summarize real school data using Google's Gemini API and Serper.dev.

## 1. Prerequisites & API Keys

You will need two API keys. Create a `.env` file in the root directory if it doesn't exist.

1.  **Google Gemini API Key:**
    *   Go to [Google AI Studio](https://aistudio.google.com/).
    *   Create a new API key.
    *   Add to `.env`: `GEMINI_API_KEY=your_gemini_key_here`

2.  **Serper.dev API Key (for Google Search):**
    *   Go to [Serper.dev](https://serper.dev/).
    *   Sign up (usually includes free credits).
    *   Add to `.env`: `SERPER_API_KEY=your_serper_key_here`

## 2. Environment Setup

Ensure you have the required Python packages installed:

```bash
pip install -r requirements.txt
# Specifically: google-genai, requests, pandas, tqdm, python-dotenv
```

## 3. Data Preparation

Before running the agent, ensure the input data is ready:

```bash
# This script merges CCD and CRDC data into data/agent_input.csv
python src/data_collection/prepare_agent_data.py
```

## 4. Running the Agent

The agent is designed to be run in 3 parallel parts to handle large datasets and rate limits efficiently. You can run one part at a time or multiple parts in different terminal windows.

```bash
# To run the first 1/3 of the dataset
python src/data_collection/gemini_brochure_agent.py --part 1

# To run the second 1/3
python src/data_collection/gemini_brochure_agent.py --part 2

# To run the final 1/3
python src/data_collection/gemini_brochure_agent.py --part 3
```

### How it Works:
- **Discovery:** The script automatically finds which Gemini models (Flash/Pro) are available for your API key.
- **Search:** It uses Serper to find mission statements and program info for each school.
- **Generation:** Gemini synthesizes the search results into a 300-word profile.
- **Storage:** Results are saved as `.txt` files in `data/raw/brochures/` using the school's NCES ID as the filename (e.g., `060000103278.txt`).
- **Resumption:** If interrupted, the script will skip any files already successfully generated.

## 5. Troubleshooting

- **429 Rate Limit:** The script has built-in retry logic. If you hit limits often, try running only one part at a time.
- **Quota Exhausted:** If your free tier runs out, the script will attempt to switch to another available model (e.g., from 1.5-flash to 2.0-flash).
- **Empty Files:** If a file is smaller than 500 bytes, it's considered a failure and the script will attempt to re-generate it in the next run.
