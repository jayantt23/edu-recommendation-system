"""
enrich_synthetic_fields.py
==========================
Adds synthetic but DETERMINISTIC fields to agent_input.csv for all 9,813 schools.

Why deterministic? Each field is seeded by the school's NCES ID, so:
  - The same school always gets the same synthetic scores
  - Re-running the pipeline gives identical results
  - LDA sees consistent patterns → learns real topic clusters

Fields added:
  - ap_count          : Number of AP courses (0–20), higher for high schools
  - norm_ap_count     : Normalized 0–1
  - stem_score        : CS + Calc + advanced science proxy (0–1)
  - arts_participants : Art + Music + Dance + Theater participants
  - norm_arts         : Normalized 0–1
  - world_lang        : World language program participants
  - norm_world_lang   : Normalized 0–1
  - is_magnet         : 1 if magnet school (13% of schools), else 0
"""

import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "data/agent_input.csv"


def det_score(ncessch: str, field: str, low: float = 0.0, high: float = 1.0) -> float:
    """
    Deterministic pseudo-random score for a school+field pair.
    Uses MD5 hash of (ncessch + field) → always same value for same school.
    """
    raw = int(hashlib.md5(f"{ncessch}_{field}".encode()).hexdigest(), 16)
    normalized = (raw % 100_000) / 100_000.0     # 0.0 → 1.0
    return low + normalized * (high - low)


def get_ap_count(row) -> int:
    """AP courses: only high/secondary schools have them, scaled by school size."""
    level = str(row.get("school_level", "")).lower()
    if "high" in level or "secondary" in level:
        base = int(det_score(row["ncessch"], "ap_base", 0, 16))
        bonus = int(row.get("norm_enrollment", 0.5) * 6)   # bigger school → more AP
        magnet_bonus = 3 if row.get("is_magnet", 0) == 1 else 0
        return min(20, base + bonus + magnet_bonus)
    elif "middle" in level:
        return int(det_score(row["ncessch"], "ap_mid", 0, 3))
    else:
        return 0   # elementary: no AP


def get_stem_score(row) -> float:
    """
    STEM score = proxy for CS courses, calculus, physics.
    High schools score higher. Magnet schools score highest.
    Name-based boost for tech/science/math keywords.
    """
    base = det_score(row["ncessch"], "stem_raw", 0.05, 0.85)
    level = str(row.get("school_level", "")).lower()
    name  = str(row.get("school_name", "")).lower()

    # Level multiplier
    if "high" in level or "secondary" in level:
        base *= 1.0
    elif "middle" in level:
        base *= 0.5
    else:
        base *= 0.25   # elementary

    # Magnet bonus
    if row.get("is_magnet", 0) == 1:
        base = min(1.0, base + 0.25)

    # Name keyword boost
    stem_kw = ["tech", "science", "math", "stem", "polytechnic", "engineering",
                "computer", "robotics", "academy", "magnet"]
    if any(kw in name for kw in stem_kw):
        base = min(1.0, base + 0.20)

    return round(min(1.0, base), 4)


def get_arts_participants(row) -> int:
    """
    Arts participants = art + music + dance + theater headcount.
    Largely independent of school size (arts is a choice).
    Magnet schools with arts names score higher.
    """
    base = int(det_score(row["ncessch"], "arts_raw", 10, 300))
    name = str(row.get("school_name", "")).lower()

    # Magnet bonus
    if row.get("is_magnet", 0) == 1:
        base = int(base * 1.4)

    # Name keyword boost
    arts_kw = ["art", "music", "theater", "dance", "perform", "creative",
                "conservatory", "fine arts"]
    if any(kw in name for kw in arts_kw):
        base = int(base * 1.6)

    return min(600, base)


def get_world_lang(row) -> int:
    """
    World language participants (Spanish, French, Mandarin, etc.).
    Scales loosely with enrollment. Urban/large schools more likely.
    """
    enroll = row.get("enrollment", 300)
    if pd.isna(enroll):
        enroll = 300
    pct = det_score(row["ncessch"], "lang_pct", 0.05, 0.40)
    return int(enroll * pct)


def is_magnet(row) -> int:
    """
    ~13% of US public schools are magnet schools.
    Seeded by school ID so it's deterministic.
    Also flag schools with 'magnet' in their name.
    """
    name = str(row.get("school_name", "")).lower()
    if "magnet" in name:
        return 1
    seed_val = int(hashlib.md5(f"{row['ncessch']}_magnet".encode()).hexdigest(), 16) % 100
    return 1 if seed_val < 13 else 0


def enrich(data_path=DATA_PATH):
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run prepare_agent_data.py first.")
        return

    df = pd.read_csv(data_path, dtype={"ncessch": str})
    df["ncessch"] = df["ncessch"].str.zfill(12)
    print(f"Loaded {len(df)} schools from {data_path}")

    # ── 1. Magnet status (needed by other fields, so compute first) ────────
    print("Computing is_magnet...")
    df["is_magnet"] = df.apply(is_magnet, axis=1)
    print(f"  Magnet schools: {df['is_magnet'].sum()} ({100*df['is_magnet'].mean():.1f}%)")

    # ── 2. AP course count ─────────────────────────────────────────────────
    print("Computing ap_count...")
    df["ap_count"] = df.apply(get_ap_count, axis=1)

    # ── 3. STEM score ──────────────────────────────────────────────────────
    print("Computing stem_score...")
    df["stem_score"] = df.apply(get_stem_score, axis=1)

    # ── 4. Arts participants ───────────────────────────────────────────────
    print("Computing arts_participants...")
    df["arts_participants"] = df.apply(get_arts_participants, axis=1)

    # ── 5. World language participants ─────────────────────────────────────
    print("Computing world_lang...")
    df["world_lang"] = df.apply(get_world_lang, axis=1)

    # ── 6. Normalize new fields ────────────────────────────────────────────
    scaler = MinMaxScaler()
    df[["norm_ap_count", "norm_arts", "norm_world_lang"]] = scaler.fit_transform(
        df[["ap_count", "arts_participants", "world_lang"]]
    )
    # stem_score is already 0–1

    # ── 7. Save back to agent_input.csv ───────────────────────────────────
    df.to_csv(data_path, index=False)
    print(f"\n✅ Enriched dataset saved → {data_path}")
    print(f"   New columns: is_magnet, ap_count, norm_ap_count,")
    print(f"                stem_score, arts_participants, norm_arts,")
    print(f"                world_lang, norm_world_lang")
    print(f"\n   Summary:")
    print(f"   AP courses (mean):        {df['ap_count'].mean():.1f}")
    print(f"   STEM score (mean):        {df['stem_score'].mean():.3f}")
    print(f"   Arts participants (mean): {df['arts_participants'].mean():.0f}")
    print(f"   World lang (mean):        {df['world_lang'].mean():.0f}")


if __name__ == "__main__":
    enrich()
