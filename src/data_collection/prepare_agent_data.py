import pandas as pd
import os

def prepare():
    print("🚀 Preparing one unified dataset for all teammates...")
    df_main = pd.read_csv("data/processed/final_structured_ms.csv", dtype={'ncessch': str})
    df_raw = pd.read_csv("data/raw/ccd_directory_2021.csv", dtype={'ncessch': str})

    # Ensure IDs are 12 digits
    df_main['ncessch'] = df_main['ncessch'].str.zfill(12)
    df_raw['ncessch'] = df_raw['ncessch'].str.zfill(12)

    # Merge location data (needed for search queries)
    df = df_main.merge(df_raw[['ncessch', 'city_location', 'lea_name']], on='ncessch', how='left')
    
    # Save ONE unified file
    if not os.path.exists("data"): os.makedirs("data")
    df.to_csv("data/agent_input.csv", index=False)
    print(f"✅ Created data/agent_input.csv with {len(df)} schools. No more redundant files!")

if __name__ == "__main__":
    prepare()
