import requests
import pandas as pd

def fetch_urban_data(endpoint_url):
    """
    Fetches data from the Urban Institute API, handling pagination automatically.
    """
    all_results = []
    current_url = endpoint_url
    
    print(f"Starting data ingestion from: {endpoint_url}")
    
    while current_url:
        response = requests.get(current_url)
        response.raise_for_status()
        data = response.json()
        
        # Append the current page's results
        results = data.get('results', [])
        all_results.extend(results)
        
        # The API provides a 'next' key with the URL for the next page of results
        current_url = data.get('next')
        if current_url:
            print(f"Fetching next page... (Current records: {len(all_results)})")
            
    print(f"Ingestion complete. Total records fetched: {len(all_results)}")
    return all_results

if __name__ == "__main__":
    # Example: Fetching Common Core of Data (CCD) school directory for a specific year and state (fips=06 is California)
    base_url = "https://educationdata.urban.org/api/v1/schools/ccd/directory/2021/?fips=06"
    
    raw_school_data = fetch_urban_data(base_url)
    
    # Convert to a DataFrame and save to the raw data folder
    df = pd.DataFrame(raw_school_data)
    df.to_csv("../../data/raw/ccd_directory_2021.csv", index=False)