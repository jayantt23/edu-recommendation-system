"""
generate_synthetic_brochures.py
================================
Generates synthetic school brochures from structured metrics.

Topic weights are now driven by REAL + ENRICHED data fields:
  STEM      ← stem_score  (CS + Calc + magnet + name keywords)
  ARTS      ← norm_arts   (art + music + dance + theater headcount)
  ATHLETICS ← norm_total_athletes (sports participants — real CRDC data)
  ACADEMIC  ← norm_ap_count (AP course count)
  CULTURAL  ← norm_world_lang (world language participants)

All 5 topics now have a meaningful, deterministic signal per school.
"""
import random

TEMPLATES = {
    "STEM": [
        "Our school focuses heavily on STEM education with advanced robotics labs and coding classes.",
        "We pride ourselves on rigorous science and mathematics. Students compete in national engineering competitions.",
        "A leader in technology-integrated learning, offering specialized tracks in computer science and biotechnology.",
        "Our computer science and calculus programs prepare students for careers in engineering and technology.",
        "Award-winning science labs and a dedicated coding curriculum set our students apart.",
    ],
    "ARTS": [
        "A vibrant community of artists and performers. Our theater and music programs are renowned for creativity.",
        "We offer comprehensive arts education including digital media, ceramics, and classical orchestra.",
        "Dedicated to fostering creativity through a strong focus on performing and visual arts.",
        "Our award-winning fine arts department includes studio art, choir, band, and drama productions.",
        "Students express themselves through a rich variety of artistic disciplines from dance to filmmaking.",
    ],
    "ATHLETICS": [
        "Home of the champions! Our athletic program offers competitive teams in over 15 sports.",
        "Strong focus on physical education and team sports with state-of-the-art training facilities.",
        "We believe in character building through sportsmanship and competitive athletics.",
        "Our student-athletes thrive in football, basketball, swimming, track, and many more sports.",
        "A championship tradition built on dedication, teamwork, and athletic excellence.",
    ],
    "ACADEMIC": [
        "Dedicated to academic excellence and preparing students for top universities.",
        "Offering a wide range of AP courses and honors programs to challenge our students.",
        "A supportive environment focused on holistic development and lifelong learning.",
        "With over a dozen Advanced Placement courses, our students are college-ready from day one.",
        "Rigorous coursework, college counseling, and a culture of intellectual curiosity define our school.",
    ],
    "CULTURAL": [
        "A rich multicultural environment celebrating diversity through events and language programs.",
        "We offer immersive cultural education including Spanish, French, and Mandarin.",
        "Our school embraces global citizenship through international exchange programs.",
        "Students explore world cultures through language immersion, cultural festivals, and global partnerships.",
        "A diverse student body and multilingual staff create a truly international learning community.",
    ],
}


def generate_brochure_from_metrics(
    school_name="",
    norm_enrollment=0.5,
    norm_athletes=0.0,
    norm_str=0.5,
    stem_score=None,
    norm_arts=None,
    norm_ap_count=None,
    norm_world_lang=None,
    is_magnet=0,
):
    """
    Generates a synthetic school brochure.
    Uses enriched synthetic fields if available; falls back to old proxies.

    Priority:
      STEM      → stem_score (enriched) else (1 - norm_str)
      ARTS      → norm_arts  (enriched) else random
      ATHLETICS → norm_total_athletes (real CRDC data, always present)
      ACADEMIC  → norm_ap_count (enriched) else norm_enrollment
      CULTURAL  → norm_world_lang (enriched) else random
    """
    name = str(school_name).lower()

    # ── Topic weights ──────────────────────────────────────────────────────
    w_stem     = stem_score      if stem_score      is not None else max(0.1, 1.0 - norm_str)
    w_arts     = norm_arts       if norm_arts        is not None else random.uniform(0.15, 0.55)
    w_athletic = norm_athletes
    w_academic = norm_ap_count   if norm_ap_count    is not None else max(0.1, norm_enrollment)
    w_cultural = norm_world_lang if norm_world_lang  is not None else random.uniform(0.1, 0.4)

    # ── Name-based boosts (uses actual school name for extra signal) ───────
    if any(kw in name for kw in ["tech", "science", "math", "stem", "poly",
                                  "engineering", "computer", "robotics"]):
        w_stem = min(1.0, w_stem + 0.30)

    if any(kw in name for kw in ["art", "music", "theater", "design",
                                  "perform", "creative", "conservatory"]):
        w_arts = min(1.0, w_arts + 0.30)

    if any(kw in name for kw in ["international", "global", "world",
                                  "multicultural", "language"]):
        w_cultural = min(1.0, w_cultural + 0.30)

    if any(kw in name for kw in ["prep", "college", "university", "academy",
                                  "scholars", "excellence"]):
        w_academic = min(1.0, w_academic + 0.25)

    # Magnet school gets a STEM + ACADEMIC boost
    if is_magnet:
        w_stem     = min(1.0, w_stem + 0.15)
        w_academic = min(1.0, w_academic + 0.10)

    # Ensure all weights are positive
    weights = {
        "STEM":      max(0.05, w_stem),
        "ARTS":      max(0.05, w_arts),
        "ATHLETICS": max(0.05, w_athletic),
        "ACADEMIC":  max(0.05, w_academic),
        "CULTURAL":  max(0.05, w_cultural),
    }

    total  = sum(weights.values())
    topics = list(weights.keys())
    probs  = [weights[t] / total for t in topics]

    # Pick 2–3 topics, weighted by school profile
    n_picks = random.randint(2, 3)
    chosen  = list(set(random.choices(topics, weights=probs, k=n_picks)))

    parts = [random.choice(TEMPLATES[t]) for t in chosen]
    return " ".join(parts)


def generate_synthetic_brochure():
    """Fallback: purely random brochure (used when no metrics available)."""
    focuses = random.sample(list(TEMPLATES.keys()), k=random.randint(2, 4))
    return " ".join(random.choice(TEMPLATES[f]) for f in focuses)