#!/usr/bin/env python3
# divergence_KL_cursus.py

"""
Compute Kullback–Leibler (KL) divergence between each student distribution and each cursus profile.

Two modes are available (mutually exclusive):

1) --ressource
   - Uses L/P transitions columns (L-L, L-P, P-L, P-P)
   - Inputs:
     docs/proportion-KL-ressource-theme/ressource/proportion_ressource_students.csv
     docs/proportion-KL-ressource-theme/ressource/proportion_ressource_cursus.csv
   - Output:
     docs/proportion-KL-ressource-theme/ressource/divergence_KL_ressource_cursus.csv

2) --theme
   - Uses theme transitions columns (data, archi_web, tech_env, data-archi_web, etc.)
     Note: same-theme transitions are stored as "data" (not "data-data"), etc.
   - Inputs:
     docs/proportion-KL-ressource-theme/theme/proportion_theme_students.csv
     docs/proportion-KL-ressource-theme/theme/proportion_theme_cursus.csv
   - Output:
     docs/proportion-KL-ressource-theme/theme/divergence_KL_theme_cursus.csv

Student-like code style: simple, clear, commented in English.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from scipy.stats import entropy


# -------------------------
# Paths (relative to repo)
# -------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESS_DIR = os.path.join(PROJECT_ROOT, "docs", "proportion-KL-ressource-theme", "ressource")
THEME_DIR = os.path.join(PROJECT_ROOT, "docs", "proportion-KL-ressource-theme", "theme")

EPSILON = 1e-9  # small value to avoid log(0) / division by 0


# -------------------------
# CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute KL divergence between each student and each cursus profile (resource or theme mode)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ressource", action="store_true", help="Compute KL on L/P transitions.")
    mode.add_argument("--theme", action="store_true", help="Compute KL on theme transitions.")
    return parser.parse_args()


def get_io_paths(mode: str):
    """
    Return (students_csv, profils_csv, output_csv) depending on the selected mode.
    """
    if mode == "ressource":
        students = os.path.join(RESS_DIR, "proportion_ressource_students.csv")
        profils = os.path.join(RESS_DIR, "proportion_ressource_cursus.csv")
        out = os.path.join(RESS_DIR, "divergence_KL_ressource_cursus.csv")
        return students, profils, out

    if mode == "theme":
        students = os.path.join(THEME_DIR, "proportion_theme_students.csv")
        profils = os.path.join(THEME_DIR, "proportion_theme_cursus.csv")
        out = os.path.join(THEME_DIR, "divergence_KL_theme_cursus.csv")
        return students, profils, out

    raise ValueError(f"Unknown mode: {mode}")


# -------------------------
# Column detection helpers
# -------------------------

def find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Find the first column whose name contains one of the keywords (case-insensitive)."""
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in keywords):
            return c
    return None


def is_transition_column(col_name: str) -> bool:
    """
    Detect if a column looks like a transition/proportion column.
    Works for both modes:
      - resource: L-L, L-P, P-L, P-P (or L_L etc.)
      - theme: data, archi_web, tech_env, data-archi_web, etc.
    """
    name = col_name.strip()
    low = name.lower()

    # Exclude obvious metadata columns
    if "login" in low or "student" in low or "id" in low:
        return False
    if "cohort" in low or "cohorte" in low:
        return False
    if "cursus" in low:
        return False
    if "note" in low or "tp" in low:
        return False
    if "effectif" in low:
        return False

    # Resource transitions (very explicit)
    if name in {"L-L", "L-P", "P-L", "P-P", "L_L", "L_P", "P_L", "P_P"}:
        return True

    # Theme transitions (allow: letters/digits/_ and optional "-other")
    # Examples: "data", "archi_web", "tech_env", "data-archi_web"
    if re.fullmatch(r"[a-zA-Z0-9_]+(-[a-zA-Z0-9_]+)?", name):
        return True

    return False


def detect_transition_columns(df_students: pd.DataFrame) -> list[str]:
    """Detect transition columns in the students CSV."""
    cols = [c for c in df_students.columns if is_transition_column(c)]
    return cols


def normalize_distribution(arr: np.ndarray) -> np.ndarray:
    """Add epsilon and normalize so the sum is 1."""
    arr = arr.astype(float) + EPSILON
    s = arr.sum()
    if s <= 0:
        # fallback: uniform distribution if something is really wrong
        return np.ones_like(arr) / len(arr)
    return arr / s


def short_cursus_name(c_name: str) -> str:
    """
    Keep the same renaming idea as your original script:
    - 'Réseaux ...' -> RT
    - 'Mesures Physiques' -> MP
    - otherwise keep as-is (cleaned)
    """
    if not isinstance(c_name, str):
        return "Unknown"
    if "Réseaux" in c_name:
        return "RT"
    if "Mesures Physiques" in c_name:
        return "MP"
    return c_name.strip()


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    mode = "ressource" if args.ressource else "theme"

    students_csv, profils_csv, output_csv = get_io_paths(mode)

    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Students CSV: {students_csv}")
    print(f"[INFO] Profils CSV : {profils_csv}")

    # Load data
    df_students = pd.read_csv(students_csv, sep=None, engine="python")
    df_profils = pd.read_csv(profils_csv, sep=None, engine="python")

    # Clean column names
    df_students.columns = [c.strip() for c in df_students.columns]
    df_profils.columns = [c.strip() for c in df_profils.columns]

    # Find metadata columns in students CSV
    col_cursus = find_col(df_students, ["cursus"])
    col_note = find_col(df_students, ["note", "tp"])
    col_id = find_col(df_students, ["login", "student", "id"])

    if not col_cursus or not col_note or not col_id:
        print("[ERROR] Could not find cursus/note/login columns in the students CSV.")
        print(f"[DEBUG] Columns: {list(df_students.columns)}")
        return

    # Detect transition columns dynamically (resource or theme)
    cols_prop = detect_transition_columns(df_students)

    if not cols_prop:
        print("[ERROR] Could not detect transition columns in students CSV.")
        print(f"[DEBUG] Columns: {list(df_students.columns)}")
        return

    # Ensure numeric for transition columns (students)
    for c in cols_prop:
        df_students[c] = pd.to_numeric(df_students[c], errors="coerce")

    # Ensure numeric for transition columns (profiles) too
    # NOTE: profile file also contains "Effectif", so we only take common columns.
    cols_prop_in_profils = [c for c in cols_prop if c in df_profils.columns]
    if len(cols_prop_in_profils) != len(cols_prop):
        missing = sorted(set(cols_prop) - set(cols_prop_in_profils))
        print("[ERROR] Some transition columns are missing in profils CSV.")
        print(f"[DEBUG] Missing columns in profils: {missing}")
        return

    for c in cols_prop_in_profils:
        df_profils[c] = pd.to_numeric(df_profils[c], errors="coerce")

    # Build reference distributions (one per cursus profile)
    profils_dict = {}
    col_cursus_prof = find_col(df_profils, ["cursus"])
    if not col_cursus_prof:
        print("[ERROR] Could not find cursus column in profils CSV.")
        print(f"[DEBUG] Columns: {list(df_profils.columns)}")
        return

    for _, row in df_profils.iterrows():
        c_name = row[col_cursus_prof]
        if pd.isna(c_name):
            continue
        dist = np.array([row[c] for c in cols_prop_in_profils], dtype=float)
        if np.any(np.isnan(dist)):
            continue
        profils_dict[str(c_name)] = normalize_distribution(dist)

    valid_cursuses = list(profils_dict.keys())
    if not valid_cursuses:
        print("[ERROR] No valid cursus profiles found (all NaN or empty).")
        return

    print(f"[INFO] Detected {len(cols_prop)} transition columns.")
    print(f"[INFO] Computing KL for {len(df_students)} students vs {len(valid_cursuses)} cursus profiles...")

    results = []

    for _, row in df_students.iterrows():
        # Skip if student transitions are missing
        if pd.isna(row[cols_prop[0]]):
            continue

        sid = row[col_id]
        own_cursus = row[col_cursus]

        # parse grade
        try:
            grade = float(str(row[col_note]).replace(",", "."))
        except Exception:
            grade = None

        # Student distribution p
        p_student = np.array([row[c] for c in cols_prop], dtype=float)
        if np.any(np.isnan(p_student)):
            continue
        p_student = normalize_distribution(p_student)

        row_res = {
            "Student_ID": sid,
            "Note_TP": grade,
            "Cursus_Origine": own_cursus
        }

        # KL(Student || Profil) for each cursus
        for c_name in valid_cursuses:
            q_profil = profils_dict[c_name]
            kl_div = entropy(pk=p_student, qk=q_profil)

            col_short = short_cursus_name(c_name)
            row_res[f"Dist_KL_{col_short}"] = round(float(kl_div), 4)

        results.append(row_res)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False, sep=";")

    print("\n[SUCCESS] KL divergence CSV generated!")
    print(f"File saved here: {output_csv}")


if __name__ == "__main__":
    main()