#!/usr/bin/env python3
# proportion_success.py

"""
Compute mean transition proportions for two grade groups:
- Success  : Note_TP >= 12
- Failure  : Note_TP < 12

Two modes are available:

1) --ressource
   - Uses L/P transition columns: L-L, L-P, P-L, P-P
   - Input:
     docs/proportion-KL-ressource-theme/ressource/proportion_ressource_students.csv
   - Output:
     docs/proportion-KL-ressource-theme/ressource/proportion_ressource_sucess.csv

2) --theme
   - Uses theme transition columns (auto-detected):
     data, archi_web, tech_env, data-archi_web, etc.
   - Input:
     docs/proportion-KL-ressource-theme/proportion_theme_students.csv
   - Output:
     docs/proportion-KL-ressource-theme/proportion_theme_success.csv

Student-like style: simple, clear, and commented in English.
"""

import os
import re
import argparse
import pandas as pd


SUCCESS_THRESHOLD = 12


# -------------------------
# Paths
# -------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESS_DIR = os.path.join(PROJECT_ROOT, "docs", "proportion-KL-ressource-theme", "ressource")
THEME_DIR = os.path.join(PROJECT_ROOT, "docs", "proportion-KL-ressource-theme", "theme")


# -------------------------
# CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mean transition proportions by success/failure (resource or theme mode)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ressource", action="store_true", help="Use L/P transition columns.")
    mode.add_argument("--theme", action="store_true", help="Use theme transition columns.")
    return parser.parse_args()


def get_io_paths(mode: str):
    """Return (input_csv, output_csv) depending on the mode."""
    if mode == "ressource":
        inp = os.path.join(RESS_DIR, "proportion_ressource_students.csv")
        out = os.path.join(RESS_DIR, "proportion_theme_sucess.csv")
        return inp, out

    if mode == "theme":
        inp = os.path.join(THEME_DIR, "proportion_theme_students.csv")
        out = os.path.join(THEME_DIR, "proportion_theme_sucess.csv")
        return inp, out

    raise ValueError(f"Unknown mode: {mode}")


# -------------------------
# Column detection (theme mode)
# -------------------------

def is_transition_column(col_name: str) -> bool:
    """
    Detect if a column looks like a transition/proportion column.

    We exclude obvious metadata columns, then accept:
    - resource transitions: L-L, L-P, P-L, P-P (also L_L etc.)
    - theme transitions: data, archi_web, tech_env, data-archi_web, etc.
    """
    name = col_name.strip()
    low = name.lower()

    # Exclude metadata
    if "login" in low or "ldap" in low:
        return False
    if "cohort" in low or "cohorte" in low:
        return False
    if "cursus" in low:
        return False
    if "note" in low or "tp" in low:
        return False
    if "groupe" in low or "effectif" in low:
        return False

    # Resource transitions (explicit)
    if name in {"L-L", "L-P", "P-L", "P-P", "L_L", "L_P", "P_L", "P_P"}:
        return True

    # Theme transitions (letters/digits/_ with optional "-other")
    if re.fullmatch(r"[a-zA-Z0-9_]+(-[a-zA-Z0-9_]+)?", name):
        return True

    return False


def detect_transition_columns(df: pd.DataFrame) -> list[str]:
    """Detect transition columns automatically (useful for theme mode)."""
    return [c for c in df.columns if is_transition_column(c)]


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    mode = "ressource" if args.ressource else "theme"

    input_csv, output_csv = get_io_paths(mode)

    if not os.path.exists(input_csv):
        print(f"[ERROR] Input file not found: {input_csv}")
        print("[HINT] Generate it first with proportion_L-P_shuffle.py using the same mode.")
        return

    df = pd.read_csv(input_csv, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Clean / parse grade
    if "Note_TP" not in df.columns:
        print("[ERROR] Column 'Note_TP' not found in the input CSV.")
        print(f"[DEBUG] Columns: {list(df.columns)}")
        return

    df["Note_TP"] = pd.to_numeric(df["Note_TP"], errors="coerce")

    # Create success/failure group (spec: success if >= 12)
    df["Groupe"] = df["Note_TP"].apply(
        lambda x: "Réussite" if pd.notna(x) and x >= SUCCESS_THRESHOLD else "Échec"
    )

    # Transition columns
    if mode == "ressource":
        cols_prop = ["L-L", "L-P", "P-L", "P-P"]
        missing = [c for c in cols_prop if c not in df.columns]
        if missing:
            print("[ERROR] Missing resource transition columns in the input CSV:", missing)
            return
    else:
        cols_prop = detect_transition_columns(df)
        if not cols_prop:
            print("[ERROR] Could not detect any theme transition columns.")
            print(f"[DEBUG] Columns: {list(df.columns)}")
            return

    # Convert transition columns to numeric
    for c in cols_prop:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove rows with missing values
    df = df.dropna(subset=cols_prop + ["Groupe"])

    # Mean proportions by group
    df_profils = df.groupby("Groupe")[cols_prop].mean().reset_index()

    # Add group sizes
    effectifs = df.groupby("Groupe").size().reset_index(name="Effectif")
    df_profils = pd.merge(df_profils, effectifs, on="Groupe")

    # Round
    df_profils[cols_prop] = df_profils[cols_prop].round(4)

    # Save
    df_profils.to_csv(output_csv, index=False, sep=";")
    
    print(f"\n[SUCCESS] File generated: {output_csv}")


if __name__ == "__main__":
    main()