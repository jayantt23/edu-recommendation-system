import os
import pandas as pd
import requests
import pdfplumber
import time
from tqdm import tqdm
from duckduckgo_search import DDGS
from fuzzywuzzy import fuzz

# Configuration
INPUT_DATA = "data/processed/final_structured_ms.csv"
RAW_CCD_DATA = "data/raw/ccd_directory_2021.csv"
OUTPUT_DIR = "data/raw/brochures"
MAX_RESULTS_PER_SCHOOL = 5
MIN_RELEVANCE_SCORE = 45 # Lowered slightly to be more inclusive initially

# Keywords that indicate a "brochure" or "prospectus"
RELEVANCE_KEYWORDS = ["mission", "vision", "curriculum", "handbook", "prospectus", "brochure", "enrollment", "about us", "learning", "student"]

def validate_pdf(file_path, school_name):
    """
    Validates if the PDF is actually a school brochure for the given school.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for i in range(min(3, len(pdf.pages))): # Check first 3 pages
                page_text = pdf.pages[i].extract_text()
                if page_text:
                    text += page_text.lower()
            
            if not text:
                return False, 0

            # 1. Check for School Name (Fuzzy matching)
            name_score = fuzz.partial_ratio(school_name.lower(), text)
            
            # 2. Check for keywords
            keyword_count = sum(1 for word in RELEVANCE_KEYWORDS if word in text)
            keyword_score = min(40, keyword_count * 8)
            
            total_score = (name_score * 0.6) + keyword_score
            return total_score >= MIN_RELEVANCE_SCORE, total_score
    except Exception:
        return False, 0

def scrape_brochures(limit=10):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df_main = pd.read_csv(INPUT_DATA, dtype={'ncessch': str})
    df_raw = pd.read_csv(RAW_CCD_DATA, dtype={'ncessch': str})
    
    df_main['ncessch'] = df_main['ncessch'].str.zfill(12)
    df_raw['ncessch'] = df_raw['ncessch'].str.zfill(12)
    
    df = df_main.merge(df_raw[['ncessch', 'city_location', 'lea_name']], on='ncessch', how='left')
    
    print(f"Starting scraping for {limit} schools...")

    with DDGS() as ddgs:
        for idx, row in tqdm(df.head(limit).iterrows(), total=limit):
            school_name = row['school_name']
            city = str(row.get('city_location', '')).replace('nan', '')
            district = str(row.get('lea_name', '')).replace('nan', '')
            school_id = row['ncessch']
            
            target_path = os.path.join(OUTPUT_DIR, f"{school_id}_brochure.pdf")
            if os.path.exists(target_path): continue

            # Robust query variations
            queries = [
                f"{school_name} {city} {district} school brochure prospectus filetype:pdf",
                f"{school_name} {city} student handbook pdf"
            ]
            
            print(f"\n🔍 Searching for: {school_name} ({city})")
            
            found_valid = False
            for query in queries:
                if found_valid: break
                
                try:
                    results = list(ddgs.text(query, max_results=MAX_RESULTS_PER_SCHOOL))
                    if not results:
                        print(f"  ⚠️ No results for query: {query}")
                        continue

                    for r in results:
                        url = r.get('href', '')
                        if not url.lower().endswith(".pdf"): continue
                        
                        print(f"  --> Checking PDF: {url}")
                        
                        try:
                            headers = {'User-Agent': 'Mozilla/5.0'}
                            resp = requests.get(url, timeout=15, headers=headers)
                            if resp.status_code == 200:
                                with open(target_path, 'wb') as f:
                                    f.write(resp.content)
                                
                                is_valid, score = validate_pdf(target_path, school_name)
                                if is_valid:
                                    print(f"  ✅ SAVED (Score: {score:.2f})")
                                    found_valid = True
                                    break
                                else:
                                    if os.path.exists(target_path): os.remove(target_path)
                        except Exception:
                            continue
                except Exception as e:
                    print(f"  ❌ Search error: {e}")
                    time.sleep(5)
            
            time.sleep(3) # Respectful delay

if __name__ == "__main__":
    scrape_brochures(limit=10)
