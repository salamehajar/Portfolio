#!/usr/bin/env python3
# proportion_cursus.py

"""
This script aggregates (by "Cursus") the transition proportions computed previously.

Two modes are available (mutually exclusive):

1) --ressource
   - Reads:
     docs/proportion-KL-ressource-theme/proportion_L-P-ressource_shuffle.csv
   - Writes:
     docs/proportion-KL-ressource-theme/proportion_L-P-ressource_shuffle_par_cursus.csv

2) --theme
   - Reads:
     docs/proportion-KL-ressource-theme/proportion_L-P-theme_shuffle.csv
   - Writes:
     docs/proportion-KL-ressource-theme/proportion_L-P-theme_shuffle_par_cursus.csv

The output file contains, for each cursus:
- the mean proportion of each transition column
- the number of students (Effectif)

Style goal: simple, clean, student-like, and commented in English.
"""

import os
import re
import argparse
import pandas as pd


# -------------------------
# Paths (relative to repo)
# -------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESS_DIR = os.path.join(PROJECT_ROOT, "docs", "proportion-KL-ressource-theme", "ressource")
THEME_DIR = os.path.join(PROJECT_ROOT, "docs", "proportion-KL-ressource-theme", "theme")

# -------------------------
# Helpers
# -------------------------

def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate transition proportions by cursus (resource or theme mode)."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--ressource",
        action="store_true",
        help="Use the CSV produced by proportion_L-P_shuffle.py --ressource."
    )
    mode.add_argument(
        "--theme",
        action="store_true",
        help="Use the CSV produced by proportion_L-P_shuffle.py --theme."
    )

    return parser.parse_args()


def get_io_paths(selected_mode: str):
    """
    Return (input_csv, output_csv) depending on the mode.
    """
    if selected_mode == "ressource":
        input_csv = os.path.join(RESS_DIR, "proportion_ressource_students.csv")
        output_csv = os.path.join(RESS_DIR, "proportion_ressource_cursus.csv")
        return input_csv, output_csv

    if selected_mode == "theme":
        input_csv = os.path.join(THEME_DIR, "proportion_theme_students.csv")
        output_csv = os.path.join(THEME_DIR, "proportion_theme_cursus.csv")
        return input_csv, output_csv

    raise ValueError(f"Unknown mode: {selected_mode}")


def find_cursus_column(df: pd.DataFrame) -> str | None:
    """
    Try to find the cursus column name dynamically.
    We keep it flexible because column names may differ slightly.
    """
    for c in df.columns:
        if "cursus" in c.lower():
            return c
    return None


def is_transition_column(col_name: str) -> bool:
    """
    Detect if a column looks like a transition/proportion column.

    In resource mode: columns like "L-L", "L-P", "P-L", "P-P" (or sometimes "L_L"...).
    In theme mode: columns like "data", "archi_web", "tech_env", "data-archi_web", etc.

    We keep it fairly permissive but avoid obvious metadata columns.
    """
    name = col_name.strip()
    low = name.lower()

    # Exclude obvious metadata columns
    if low in {"login_ldap", "cohorte", "note_tp", "note", "effectif"}:
        return False
    if "login" in low or "cohort" in low or "cohorte" in low or "note" in low:
        return False
    if "cursus" in low:
        return False

    # Resource transitions
    if name in {"L-L", "L-P", "P-L", "P-P"}:
        return True
    if name in {"L_L", "L_P", "P_L", "P_P"}:
        return True

    # Theme transitions (examples: "data", "archi_web", "tech_env", "data-archi_web", etc.)
    # We accept letters, digits, underscore, and optional "-<same pattern>".
    # This fits "data-archi_web" and also "archi_web-data".
    if re.fullmatch(r"[a-zA-Z0-9_]+(-[a-zA-Z0-9_]+)?", name):
        return True

    return False


def detect_transition_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns considered as transition proportion columns."""
    cols = []
    for c in df.columns:
        if is_transition_column(c):
            cols.append(c)
    return cols


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    mode = "ressource" if args.ressource else "theme"

    input_csv, output_csv = get_io_paths(mode)

    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Reading input CSV: {input_csv}")

    if not os.path.exists(input_csv):
        print(f"[ERROR] File not found: {input_csv}")
        print("[HINT] Make sure you generated it using proportion_L-P_shuffle.py with the same mode.")
        return

    # Read CSV (delimiter can vary, so we let pandas detect it)
    df = pd.read_csv(input_csv, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Find cursus column
    col_cursus = find_cursus_column(df)
    if not col_cursus:
        print("[ERROR] Could not find the cursus column in the CSV.")
        print(f"[DEBUG] Available columns: {list(df.columns)}")
        return

    # Detect transition columns dynamically (works for both modes)
    cols_transitions = detect_transition_columns(df)
    if not cols_transitions:
        print("[ERROR] Could not detect any transition columns to aggregate.")
        print(f"[DEBUG] Available columns: {list(df.columns)}")
        return

    # Convert transition columns to numeric (coerce errors to NaN)
    for col in cols_transitions:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing cursus or missing transition values
    df = df.dropna(subset=[col_cursus] + cols_transitions)

    print(f"[INFO] Computing mean profiles by cursus for {len(df)} students...")

    # Group by cursus and compute mean of transitions
    df_profiles = df.groupby(col_cursus)[cols_transitions].mean().reset_index()

    # Add student counts per cursus
    counts = df.groupby(col_cursus).size().reset_index(name="Effectif")
    df_profiles = pd.merge(df_profiles, counts, on=col_cursus)

    # Round values (3 decimals like your original script)
    df_profiles[cols_transitions] = df_profiles[cols_transitions].round(3)

    # Save
    df_profiles.to_csv(output_csv, index=False, sep=";")

    # Print a small readable summary in terminal
    print("==================================================")
    print(" RESULTS: MEAN TRANSITION PROPORTIONS BY CURSUS")
    print("==================================================\n")

    for _, row in df_profiles.iterrows():
        print(f"--- {row[col_cursus]} ({row['Effectif']} students) ---")
        for col in cols_transitions:
            print(f"  {col} : {row[col]}")
        print("")

    print(f"[SUCCESS] Saved output here: {output_csv}")


if __name__ == "__main__":
    main()
    