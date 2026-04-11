import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def integrate_crdc_metrics(ccd_clean_path, crdc_raw_path, output_path):
    # Load previously cleaned CCD data and raw CRDC data
    ccd_df = pd.read_csv(ccd_clean_path)
    crdc_df = pd.read_csv(crdc_raw_path)
    
    # 1. Feature Engineering: School Culture Proxies (Athletics)
    # The CRDC API uses these fields for sports participation
    sports_cols = ['participants_single_sex_sports_m', 'participants_single_sex_sports_f']
    
    # Fill missing values with 0 (assuming no participation reported means 0)
    for col in sports_cols:
        if col in crdc_df.columns:
            crdc_df[col] = pd.to_numeric(crdc_df[col], errors='coerce').fillna(0)
        else:
            crdc_df[col] = 0
            
    # The proxy metric: Total athletic participation
    crdc_df['total_athletes'] = crdc_df['participants_single_sex_sports_m'] + crdc_df['participants_single_sex_sports_f']
    
    # Keep only the joining key and our new culture proxy metric
    crdc_subset = crdc_df[['ncessch', 'total_athletes']]
    
    # 2. Merging the Datasets
    # Join the CRDC data onto the primary CCD metrics vector using the NCES school ID
    final_ms_df = pd.merge(ccd_df, crdc_subset, on='ncessch', how='left')
    
    
    
    # 3. Normalizing the New Metric
    # Schools without CRDC data get a 0 for athletic participation before normalization
    final_ms_df['total_athletes'] = final_ms_df['total_athletes'].fillna(0)
    
    scaler = MinMaxScaler()
    final_ms_df[['norm_total_athletes']] = scaler.fit_transform(final_ms_df[['total_athletes']])
    
    # Save the finalized structured M_s vector
    final_ms_df.to_csv(output_path, index=False)
    print(f"Final structured dataset saved to {output_path}. Total schools ready: {len(final_ms_df)}")

if __name__ == "__main__":
    integrate_crdc_metrics(
        "../../data/processed/ms_metrics_clean.csv", 
        "../../data/raw/crdc_offerings_2017.csv", 
        "../../data/processed/final_structured_ms.csv"
    )