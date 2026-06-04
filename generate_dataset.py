"""Generate synthetic career recommendation dataset."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 3000
NOISE_RATE = 0.10

INTEREST_AREAS = [
    "AI/ML",
    "Coding",
    "Business",
    "Networking",
    "Design",
    "Data",
    "Security",
    "Management",
]

CAREERS = [
    "Machine Learning",
    "Software Development",
    "Data Science",
    "Product Management",
    "Cybersecurity",
    "Cloud Engineering",
    "UI/UX Design",
    "Business Analytics",
]


def _assign_career(row: pd.Series) -> str:
    """Assign career label using rule-based scoring (highest score wins)."""
    prog = row["programming_skill"]
    math = row["math_skill"]
    comm = row["communication_skill"]
    logic = row["logic_score"]
    apt = row["aptitude_score"]
    interest = row["interest_area"]

    scores = {}

    # Machine Learning
    if prog >= 7 and math >= 7 and interest in ("AI/ML", "Data"):
        scores["Machine Learning"] = (prog - 6) + (math - 6) + 2

    # Software Development
    if prog >= 7 and logic >= 6 and interest == "Coding":
        scores["Software Development"] = (prog - 6) + (logic - 5) + 2

    # Data Science
    if math >= 7 and prog >= 6 and interest in ("Data", "AI/ML"):
        scores["Data Science"] = (math - 6) + (prog - 5) + 2

    # Product Management
    if comm >= 7 and apt >= 65 and interest in ("Business", "Management"):
        scores["Product Management"] = (comm - 6) + (apt - 64) / 10 + 2

    # Cybersecurity
    if logic >= 7 and interest in ("Security", "Networking"):
        scores["Cybersecurity"] = (logic - 6) + 2

    # Cloud Engineering
    if prog >= 6 and logic >= 6 and interest == "Networking":
        scores["Cloud Engineering"] = (prog - 5) + (logic - 5) + 2

    # UI/UX Design
    if comm >= 6 and interest == "Design":
        scores["UI/UX Design"] = (comm - 5) + 2

    # Business Analytics
    if math >= 6 and comm >= 6 and interest in ("Business", "Data"):
        scores["Business Analytics"] = (math - 5) + (comm - 5) + 2

    if scores:
        return max(scores, key=scores.get)

    # Fallback: nearest career by dominant traits
    fallback_scores = {
        "Machine Learning": prog + math + (2 if interest in ("AI/ML", "Data") else 0),
        "Software Development": prog + logic + (2 if interest == "Coding" else 0),
        "Data Science": math + prog + (2 if interest in ("Data", "AI/ML") else 0),
        "Product Management": comm + apt / 20 + (2 if interest in ("Business", "Management") else 0),
        "Cybersecurity": logic + (2 if interest in ("Security", "Networking") else 0),
        "Cloud Engineering": prog + logic + (2 if interest == "Networking" else 0),
        "UI/UX Design": comm + (2 if interest == "Design" else 0),
        "Business Analytics": math + comm + (2 if interest in ("Business", "Data") else 0),
    }
    return max(fallback_scores, key=fallback_scores.get)


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    programming_skill = rng.integers(1, 11, size=n_samples)
    math_skill = rng.integers(1, 11, size=n_samples)
    communication_skill = rng.integers(1, 11, size=n_samples)
    logic_score = rng.integers(1, 11, size=n_samples)
    cgpa = np.round(rng.uniform(5.0, 10.0, size=n_samples), 1)
    aptitude_score = rng.integers(40, 101, size=n_samples)
    interest_area = rng.choice(INTEREST_AREAS, size=n_samples)

    df = pd.DataFrame(
        {
            "programming_skill": programming_skill,
            "math_skill": math_skill,
            "communication_skill": communication_skill,
            "logic_score": logic_score,
            "cgpa": cgpa,
            "aptitude_score": aptitude_score,
            "interest_area": interest_area,
        }
    )

    df["career"] = df.apply(_assign_career, axis=1)

    # ~10% controlled label noise
    n_flip = int(n_samples * NOISE_RATE)
    flip_idx = rng.choice(n_samples, size=n_flip, replace=False)
    for idx in flip_idx:
        current = df.at[idx, "career"]
        others = [c for c in CAREERS if c != current]
        df.at[idx, "career"] = rng.choice(others)

    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    out_path = data_dir / "career_dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} records -> {out_path}")
    print("\nCareer distribution:")
    print(df["career"].value_counts().sort_index())
    print("\nInterest area distribution:")
    print(df["interest_area"].value_counts().sort_index())


if __name__ == "__main__":
    main()
