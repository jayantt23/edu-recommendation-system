import os
import pandas as pd
from tqdm import tqdm

def generate_synthetic_brochure(school_name, athletes_score, st_ratio):
    parts = ["Our mission is to provide a holistic and challenging educational environment."]
    name = str(school_name).lower()
    
    # STEM Heuristic
    if any(kw in name for kw in ['tech', 'science', 'math', 'poly', 'eng', 'prep']):
        parts.append("We specialize in a rigorous STEM curriculum with advanced labs in robotics, coding, and biotechnology.")
    
    # Arts Heuristic
    if any(kw in name for kw in ['art', 'music', 'theater', 'design', 'perform']):
        parts.append("Our vibrant arts program offers students deep immersion in visual arts, classical music, and theater production.")
        
    # Athletics Heuristic (Using the data)
    if athletes_score > 0.1:
        parts.append("With a high participation rate in athletics, our school fosters leadership and teamwork through competitive sports.")
    else:
        parts.append("We offer a balanced physical education program for all students.")

    # Academic Heuristic
    if pd.notna(st_ratio) and st_ratio < 15:
        parts.append("Our low student-to-teacher ratio allows for personalized academic support and advanced AP course offerings.")

    return " ".join(parts)

if __name__ == "__main__":
    INPUT_PATH = "data/agent_input.csv"
    OUT_DIR = "data/raw/brochures"
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        exit(1)
        
    df = pd.read_csv(INPUT_PATH)
    
    print(f"Generating informed synthetic brochures for {len(df)} schools...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        school_id = str(row['ncessch']).zfill(12)
        target = os.path.join(OUT_DIR, f"{school_id}.txt")
        
        # Only generate if it doesn't exist to save time
        if not os.path.exists(target):
            content = generate_synthetic_brochure(
                row['school_name'], 
                row.get('norm_total_athletes', 0), 
                row.get('student_teacher_ratio', 20)
            )
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
                
    print(f"Done! Brochures saved in {OUT_DIR}")
