import random

TEMPLATES = {
    "STEM": [
        "Our school focuses heavily on STEM education with advanced robotics labs and coding classes.",
        "We pride ourselves on rigorous science and mathematics. Students compete in national engineering competitions.",
        "A leader in technology-integrated learning, offering specialized tracks in computer science and biotechnology.",
    ],
    "ARTS": [
        "A vibrant community of artists and performers. Our theater and music programs are renowned for creativity.",
        "We offer comprehensive arts education including digital media, ceramics, and classical orchestra.",
        "Dedicated to fostering creativity through a strong focus on performing and visual arts.",
    ],
    "ATHLETICS": [
        "Home of the champions! Our athletic program offers competitive teams in over 15 sports.",
        "Strong focus on physical education and team sports with state-of-the-art training facilities.",
        "We believe in character building through sportsmanship and competitive athletics.",
    ],
    "ACADEMIC": [
        "Dedicated to academic excellence and preparing students for top universities.",
        "Offering a wide range of AP courses and honors programs to challenge our students.",
        "A supportive environment focused on holistic development and lifelong learning.",
    ],
    "CULTURAL": [
        "A rich multicultural environment celebrating diversity through events and language programs.",
        "We offer immersive cultural education including Spanish, French, and Mandarin.",
        "Our school embraces global citizenship through international exchange programs.",
    ],
}

def generate_brochure_from_metrics(
    school_name="",
    norm_enrollment=0.5,
    norm_athletes=0.0,
    norm_str=0.5
):
    """
    Hybrid generator:
    - Uses metrics (athletes, enrollment, STR)
    - Uses name-based hints (STEM/ARTS keywords)
    """

    name = str(school_name).lower()

    weights = {
        "STEM": max(0.1, 1.0 - norm_str),
        "ARTS": random.uniform(0.2, 0.6),
        "ATHLETICS": max(0.1, norm_athletes),
        "ACADEMIC": max(0.1, norm_enrollment),
        "CULTURAL": random.uniform(0.1, 0.4),
    }

    # 🔥 Inject name-based bias (best of old version)
    if any(kw in name for kw in ['tech', 'science', 'math', 'poly', 'eng', 'prep']):
        weights["STEM"] += 0.5

    if any(kw in name for kw in ['art', 'music', 'theater', 'design', 'perform']):
        weights["ARTS"] += 0.5

    total = sum(weights.values())
    topics = list(weights.keys())
    probs = [weights[t] / total for t in topics]

    # Pick topics
    chosen = list(set(random.choices(topics, weights=probs, k=random.randint(2, 3))))

    parts = [random.choice(TEMPLATES[t]) for t in chosen]
    return " ".join(parts)


def generate_synthetic_brochure():
    """Fallback random brochure."""
    focuses = random.sample(list(TEMPLATES.keys()), k=random.randint(2, 4))
    return " ".join(random.choice(TEMPLATES[f]) for f in focuses)