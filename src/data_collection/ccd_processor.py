import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_ccd_data(raw_csv_path, output_csv_path):
    # 1. Load the raw API data
    df = pd.read_csv(raw_csv_path)
    
    # 2. Select only the features needed for Ms and Pengeo
    columns_to_keep = [
        'ncessch', 'school_name', 'latitude', 'longitude', 
        'enrollment', 'teachers_fte', 'school_level', 'school_type'
    ]
    df = df[columns_to_keep].copy()
    
    # 3. Handle missing data (Drop rows where critical metrics are missing)
    df.dropna(subset=['enrollment', 'teachers_fte', 'latitude', 'longitude'], inplace=True)
    
    # 4. Feature Engineering: Calculate Student-Teacher Ratio
    df['student_teacher_ratio'] = df['enrollment'] / df['teachers_fte']
    
    # Handle any infinite values caused by 0 teachers
    df = df.replace([float('inf'), -float('inf')], pd.NA).dropna(subset=['student_teacher_ratio'])
    
    # 5. Normalization (Algorithm 1, Step 7 requirement)
    # Using MinMaxScaler to bound values between 0 and 1 for Score_met
    scaler = MinMaxScaler()
    df[['norm_enrollment', 'norm_student_teacher_ratio']] = scaler.fit_transform(
        df[['enrollment', 'student_teacher_ratio']]
    )
    
    # Save the processed administrative metrics
    df.to_csv(output_csv_path, index=False)
    print(f"Processed data saved to {output_csv_path}. Total valid schools: {len(df)}")

if __name__ == "__main__":
    import os, sys
    # Works when run from the project root: python src/data_collection/ccd_processor.py
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    process_ccd_data(
        os.path.join(root, "data", "raw", "ccd_directory_2021.csv"),
        os.path.join(root, "data", "processed", "ms_metrics_clean.csv")
    )