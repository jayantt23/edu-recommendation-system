import pandas as pd
from urban_api_client import fetch_urban_data

def get_crdc_offerings(year, fips_code):
    # CRDC offerings endpoint contains athletic and extracurricular participation variables
    base_url = f"https://educationdata.urban.org/api/v1/schools/crdc/offerings/{year}/?fips={fips_code}"
    
    raw_crdc_data = fetch_urban_data(base_url)
    
    df = pd.DataFrame(raw_crdc_data)
    output_path = f"../../data/raw/crdc_offerings_{year}.csv"
    df.to_csv(output_path, index=False)
    print(f"CRDC data saved to {output_path}")

if __name__ == "__main__":
    get_crdc_offerings(2017, "06") # fips=06 is California