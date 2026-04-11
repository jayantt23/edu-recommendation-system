import os
import time
import pandas as pd
import requests
import argparse
import numpy as np
from google import genai
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GEMINI_API_KEY or not SERPER_API_KEY:
    print("Error: GEMINI_API_KEY or SERPER_API_KEY missing in .env")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

def get_available_flash_models():
    """Fetches models actually available to this API key and returns Flash versions."""
    try:
        models = []
        print("🔍 Discovering available models...")
        all_models = list(client.models.list())
        
        for m in all_models:
            m_name = m.name.lower()
            # Look for any version of flash or pro that supports content generation
            methods = getattr(m, 'supported_generation_methods', [])
            if not methods: # Fallback for some SDK versions
                 methods = getattr(m, 'supported_methods', [])
            
            if 'generateContent' in str(methods) or 'generate_content' in str(methods):
                if 'flash' in m_name or 'pro' in m_name:
                    models.append(m.name)
        
        if not models:
            print("⚠️ No flash models found in discovery. Using hardcoded defaults.")
            return ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest"]
            
        models.sort(reverse=True)
        return models
    except Exception as e:
        print(f"⚠️ Failed to list models: {e}")
        return ["gemini-1.5-flash", "gemini-2.0-flash"]

# Discover models at startup
AVAILABLE_MODELS = get_available_flash_models()
print(f"✅ Confirmed access to: {AVAILABLE_MODELS}")

def get_deep_info(school_name, city, district):
    search_url = "https://google.serper.dev/search"
    query = f'"{school_name}" {city} {district} mission statement curriculum athletics programs'
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    
    try:
        search_res = requests.post(search_url, headers=headers, json={"q": query}, timeout=15).json()
        snippets = "\n".join([f"- {r.get('snippet', '')}" for r in search_res.get('organic', [])[:6]])
        
        prompt = f"""
        School: {school_name} ({city})
        Search Context: {snippets}
        Task: Write a 300-word 'Educational Taste Profile' focusing on STEM, Arts, Athletics, and Mission.
        Format: Descriptive prose.
        """

        global AVAILABLE_MODELS
        for model_name in list(AVAILABLE_MODELS):
            try:
                # Use retry logic for 429s
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(model=model_name, contents=prompt)
                        if response.text:
                            return response.text
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str:
                            wait_time = 60 if "limit: 0" in err_str else 30
                            print(f"\n⚠️ Rate limit hit for {model_name}. Waiting {wait_time}s...")
                            time.sleep(wait_time)
                            if attempt == max_retries - 1:
                                raise e
                        else:
                            raise e
            except Exception as e:
                err_str = str(e).lower()
                if "404" in err_str:
                    print(f"\n❌ Model {model_name} not found. Removing from session.")
                    AVAILABLE_MODELS.remove(model_name)
                elif "limit: 0" in err_str or "quota" in err_str:
                    print(f"\n🚫 Quota exhausted for {model_name}. Trying next model...")
                    AVAILABLE_MODELS.remove(model_name)
                else:
                    print(f"\n⚠️ Error with {model_name}: {e}")
                continue
        return None
    except Exception as e:
        print(f"General Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    INPUT_PATH = "data/agent_input.csv"
    if not os.path.exists(INPUT_PATH):
        print("Run prepare_agent_data.py first!")
        exit(1)

    df = pd.read_csv(INPUT_PATH)
    chunk_size = int(np.ceil(len(df) / 3))
    start_idx = (args.part - 1) * chunk_size
    end_idx = start_idx + chunk_size
    df_part = df.iloc[start_idx:end_idx]

    OUT_DIR = "data/raw/brochures"
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

    print(f"🚀 Starting Part {args.part} using dynamic discovery.")

    for idx, row in tqdm(df_part.iterrows(), total=len(df_part)):
        school_id = str(row['ncessch']).zfill(12)
        target = os.path.join(OUT_DIR, f"{school_id}.txt")
        
        if os.path.exists(target) and os.path.getsize(target) > 500:
            continue

        info = get_deep_info(row['school_name'], str(row['city_location']), str(row['lea_name']))
        
        if info:
            with open(target, "w", encoding="utf-8") as f:
                f.write(info)
            time.sleep(5) 
        else:
            if not AVAILABLE_MODELS:
                print("🛑 FATAL: No working models found for your API key/region.")
                exit(1)
            time.sleep(2)
