import os
import pandas as pd
import random
from tqdm import tqdm

TEMPLATES = {
    "STEM": [
        "Our school focuses heavily on STEM education. We have advanced robotics labs and coding classes starting from elementary school.",
        "We pride ourselves on our rigorous science and mathematics curriculum. Students participate in national engineering competitions.",
        "A leader in technology-integrated learning, offering specialized tracks in computer science and biotechnology."
    ],
    "ARTS": [
        "A vibrant community of artists and performers. Our theater and music programs are renowned for their creativity.",
        "We offer comprehensive arts education, including digital media, ceramics, and classical orchestra.",
        "Dedicated to fostering creativity through a strong focus on the performing and visual arts."
    ],
    "ATHLETICS": [
        "Home of the champions! Our athletic program offers competitive teams in over 15 sports.",
        "Strong focus on physical education and team sports, with state-of-the-art training facilities.",
        "We believe in character building through sportsmanship and competitive athletics."
    ],
    "ACADEMIC": [
        "Dedicated to academic excellence and preparing students for the top universities in the country.",
        "Offering a wide range of AP courses and honors programs to challenge our students.",
        "A supportive environment focused on holistic development and lifelong learning."
    ]
}

def generate_synthetic_brochure():
    parts = []
    # Pick a random set of focuses
    focuses = random.sample(list(TEMPLATES.keys()), k=random.randint(2, 4))
    for f in focuses:
        parts.append(random.choice(TEMPLATES[f]))
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
    
    # Process only a subset (e.g., 500 schools) for demonstration
    limit = 500
    df_subset = df.head(limit)
    
    print(f"Generating {len(df_subset)} synthetic brochures...")
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        school_id = str(row['ncessch']).zfill(12)
        target = os.path.join(OUT_DIR, f"{school_id}.txt")
        
        if not os.path.exists(target):
            with open(target, "w", encoding="utf-8") as f:
                f.write(generate_synthetic_brochure())
                
    print(f"Done! Brochures saved in {OUT_DIR}")
