import os
import pandas as pd
from tqdm import tqdm

def generate_informed_brochure(school_name, row):
    parts = [f"Welcome to {school_name}, where our mission is to provide a holistic and challenging educational environment."]
    
    # Calculate STEM Intensity
    stem_cols = [
        'num_classes_algebra1', 'num_classes_algebra2', 'num_classes_advanced_math', 
        'num_classes_calculus', 'num_classes_biology', 'num_classes_chemistry', 
        'num_classes_geometry', 'num_classes_physics'
    ]
    # Replace negative values with 0 (CRDC uses negatives for missing/suppressed)
    stem_sum = sum(max(0, row.get(c, 0)) for c in stem_cols)
    ap_stem = (row.get('ap_courses_math_indicator') == 'Yes' or row.get('ap_courses_science_indicator') == 'Yes')

    if stem_sum > 20 or ap_stem:
        parts.append(f"We are proud to offer a robust STEM curriculum, featuring {int(stem_sum)} specialized sections in mathematics and science, including advanced labs and AP-level coursework.")
    elif stem_sum > 5:
        parts.append("Our curriculum includes strong foundational courses in mathematics and science, preparing students for various technical career paths.")
    else:
        parts.append("We focus on core literacy and numeracy, ensuring all students have a solid academic foundation.")

    # Arts/Humanities (Proxy)
    ap_other = row.get('ap_courses_other_indicator') == 'Yes'
    if ap_other or any(kw in str(school_name).lower() for kw in ['art', 'music', 'theater', 'design', 'perform']):
        parts.append("Beyond the core subjects, our school is a hub for creativity, offering rich programs in the arts and humanities that encourage self-expression and critical thinking.")
    
    # Athletics
    athletes = row.get('norm_total_athletes', 0)
    if athletes > 0.2:
        parts.append(f"Athletics play a central role in our school culture, with high student participation in a wide range of competitive sports teams.")
    elif athletes > 0:
        parts.append("We offer various athletic opportunities and physical education programs to promote a healthy and active lifestyle.")

    # Specialized Programs
    if row.get('gifted_talented_indicator') == 'Yes':
        parts.append("Our Gifted and Talented program provides additional challenges and opportunities for high-achieving students.")
    
    if row.get('sch_dual_indicator') == 'Yes':
        parts.append("We also offer dual enrollment programs, allowing students to earn college credits while still in high school.")

    # Academic Support
    st_ratio = row.get('student_teacher_ratio', 20)
    if pd.notna(st_ratio) and st_ratio < 15:
        parts.append(f"With a student-to-teacher ratio of {st_ratio:.1f}:1, we prioritize personalized attention and academic support for every individual.")

    return " ".join(parts)

if __name__ == "__main__":
    AGENT_INPUT = "data/agent_input.csv"
    CRDC_DATA = "data/raw/crdc_offerings_2017.csv"
    OUT_DIR = "data/raw/brochures"
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    print("🚀 Loading data for informed synthetic generation...")
    df_agent = pd.read_csv(AGENT_INPUT, dtype={'ncessch': str})
    df_crdc = pd.read_csv(CRDC_DATA, dtype={'ncessch': str})
    
    # Merge on ncessch
    df = df_agent.merge(df_crdc, on='ncessch', how='left')
    
    print(f"Generating informed synthetic brochures for {len(df)} schools...")
    count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        school_id = str(row['ncessch']).zfill(12)
        target = os.path.join(OUT_DIR, f"{school_id}.txt")
        
        # Only generate if it doesn't exist OR is very small (potentially old synthetic)
        if not os.path.exists(target) or os.path.getsize(target) < 100:
            content = generate_informed_brochure(row['school_name'], row)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            count += 1
                
    print(f"Done! Generated/Updated {count} brochures in {OUT_DIR}")
