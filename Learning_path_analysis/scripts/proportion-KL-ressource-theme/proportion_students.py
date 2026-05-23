#!/usr/bin/env python3
# proportion_students.py

"""
This script computes transition proportions from student logs.

Two modes are available (mutually exclusive):

1) --ressource
   - Transitions between resource types: L -> L, L -> P, P -> L, P -> P
   - Output:
     docs/proportion-KL-ressource-theme/proportion_ressource_students.csv

2) --theme
   - Transitions between pedagogical themes (data, archi_web, tech_env)
   - If a transition stays in the same theme (e.g., data -> data),
     we store only "data" (not "data-data").
   - Output:
     docs/proportion-KL-ressource-theme/proportion_theme_students.csv
"""

import os
import re
import argparse
from datetime import datetime
import pandas as pd

# -------------------------
# Configuration / constants
# -------------------------

# Limits (inclusive): ignore log lines AFTER these dates for each group.
LIMIT_GROUP_1 = datetime.strptime("21/02/2024", "%d/%m/%Y").date()
LIMIT_GROUP_2 = datetime.strptime("21/02/2025", "%d/%m/%Y").date()

PEDAGOGICAL_STRUCTURE = {
    "data": {
        "lecture": [
            "prerequisite_data", "prerequisite_data_2025",
            "lecture_relational_model", "lecture_db_integrity",
            "lecture_relational_algebra", "lecture_modeling_normalization",
            "lecture_conceptual_modeling", "lecture_normalization",
            "lecture_independence_views", "lecture_indexation",
            "lecture_transaction", "lecture_isolation", "lecture_db_tools"
        ],
        "practice": [
            "activity_relational_model", "activity_relational_model_correction",
            "activity_conceptual_modeling", "activity_conceptual_modeling_correction",
            "activity_conceptual_modeling_advanced",
            "activity_conceptual_modeling_advanced_correction",
            "activity_normalization", "activity_normalization_correction",
            "activity_decomposition", "activity_decomposition_correction",
            "activity_sql", "activity_sql_correction",
            "practice_independence_views", "practice_independence_views_correction",
            "practice_sql", "practice_sql_correction",
            "practice_advanced_sql", "practice_advanced_sql_correction",
            "practice_indexation", "practice_indexation_correction",
            "practice_transaction", "practice_transaction_correction",
            "practice_isolation", "practice_isolation_correction"
        ]
    },

    "archi_web": {
        "lecture": [
            "prerequisite_archiweb", "prerequisite_archiweb_2025",
            "web_architecture_home", "lecture_technical_architecture",
            "lecture_software_architecture", "lecture_dto",
            "lecture_mvc_spring", "lecture_jpa", "lecture_xml",
            "lecture_validity", "lecture_xpath", "lecture_parser",
            "lecture_jaxb", "lecture_json"
        ],
        "practice": [
            "practice_spring_2tiers", "practice_spring_2tiers_ext",
            "practice_jpa", "practice_jpa_advanced",
            "practice_xml", "practice_schema_xml",
            "practice_xpath", "practice_apis", "practice_json"
        ]
    },

    "tech_env": {
        "lecture": [
            "prerequisite_techenv", "prerequisite_techenv_2025",
            "technical_environment_home", "documentation_docker",
            "documentation_rdbms", "documentation_programming",
            "spring_bean", "inversion_of_control", "documentation_si"
        ],
        "practice": [
            "practice_spring", "practice_postgresql",
            "practice_docker", "practice_docker_advanced"
        ]
    }
}

# Log extraction: resource name from something like: GET /.../resource.html HTTP/1.1
GET_RE = re.compile(r'GET\s+[^"\s]*/([^/?"]+)\.html', re.IGNORECASE)


# -------------------------
# Small helper functions
# -------------------------

def cursus_category(last_degree: str | None) -> str | None:
    """Simple mapping based on string inclusion."""
    if not last_degree:
        return None

    s = last_degree.strip().lower()

    if "réseaux" in s and "télé" in s:
        return "Réseaux & Télécommunications"
    if "mesures physiques" in s:
        return "Mesures Physiques"
    if "génie électrique" in s or "geii" in s:
        return "GEII"
    if "informatique" in s:
        return "Informatique"
    if "cpge" in s or "ats" in s:
        return "CPGE/ATS"

    return None


def extract_resource(line: str) -> str | None:
    """Extract the resource name (without .html) from a log line."""
    m = GET_RE.search(line)
    if not m:
        return None
    return m.group(1)


def extract_log_date(line: str):
    """
    Expected log format example:
    2414\t2024/01/29 09:33:52\t"GET ..."

    Returns a date object, or None if parsing fails.
    """
    parts = line.split("\t")
    if len(parts) < 2:
        return None

    dt_str = parts[1].strip()  # "2024/01/29 09:33:52"
    try:
        return datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S").date()
    except ValueError:
        return None


def load_metadata(csv_path: str) -> dict:
    """
    Load a CSV and return a dict keyed by Login_LDAP for quick lookup.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    meta = {}
    for _, row in df.iterrows():
        login = str(row["Login_LDAP"])
        meta[login] = row
    return meta


# -------------------------
# Maps (resource -> type/theme)
# -------------------------

def build_resource_type_map() -> dict:
    """
    Build a dict:
      resource_name -> "L" or "P"
    """
    mp = {}
    for _, block in PEDAGOGICAL_STRUCTURE.items():
        for r in block["lecture"]:
            mp[r] = "L"
        for r in block["practice"]:
            mp[r] = "P"
    return mp


def build_resource_theme_map() -> dict:
    """
    Build a dict:
      resource_name -> theme_name
    Example:
      "lecture_relational_model" -> "data"
    """
    mp = {}
    for theme_name, block in PEDAGOGICAL_STRUCTURE.items():
        for r in block["lecture"]:
            mp[r] = theme_name
        for r in block["practice"]:
            mp[r] = theme_name
    return mp


RESOURCE_TYPE = build_resource_type_map()
RESOURCE_THEME = build_resource_theme_map()


# -------------------------
# Core computation
# -------------------------

def compute_transition_proportions(
    log_path: str,
    limit_date,
    resource_to_value: dict,
    possible_labels: list[str],
    same_value_label_is_single: bool
) -> dict[str, float]:
    """
    Generic transition computation.

    - resource_to_value maps a resource name to a value:
        * mode ressource: value in {"L", "P"}
        * mode theme: value in {"data", "archi_web", "tech_env"}

    - possible_labels defines output columns ordering.

    - same_value_label_is_single:
        * False: "L-L" is stored as "L-L"
        * True:  "data -> data" is stored as "data" (not "data-data")
    """
    counts = {lab: 0 for lab in possible_labels}
    prev_value = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            d = extract_log_date(line)
            if d is None:
                continue
            if d > limit_date:
                continue  # ignore after limit_date

            res = extract_resource(line)
            if not res:
                continue

            cur_value = resource_to_value.get(res)
            if not cur_value:
                continue  # unknown resource (not in our pedagogical structure)

            if prev_value is not None:
                if same_value_label_is_single and prev_value == cur_value:
                    label = prev_value
                else:
                    label = f"{prev_value}-{cur_value}"

                # Just in case: only count labels we planned to output
                if label in counts:
                    counts[label] += 1

            prev_value = cur_value

    total = sum(counts.values())
    if total == 0:
        return {lab: 0.0 for lab in possible_labels}

    return {lab: counts[lab] / total for lab in possible_labels}


def process_group(
    group_dir: str,
    cohort_name: str,
    meta: dict,
    limit_date,
    mode: str
) -> list[dict]:
    """
    Process all student logs in one directory (group1 or group2) and return rows for the output CSV.
    """
    rows = []

    # Define mode-specific settings
    if mode == "ressource":
        possible_labels = ["L-L", "L-P", "P-L", "P-P"]
        resource_to_value = RESOURCE_TYPE
        same_value_label_is_single = False
    elif mode == "theme":
        # For 3 themes, we output 3 "same-theme" labels + 6 cross labels = 9 columns.
        themes = list(PEDAGOGICAL_STRUCTURE.keys())  # ["data", "archi_web", "tech_env"]
        cross = []
        for a in themes:
            for b in themes:
                if a != b:
                    cross.append(f"{a}-{b}")

        possible_labels = themes + cross
        resource_to_value = RESOURCE_THEME
        same_value_label_is_single = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Process each .log file
    for fname in os.listdir(group_dir):
        if not fname.endswith(".log"):
            continue
        if fname == "students.log":
            continue

        login = os.path.splitext(fname)[0]
        row = meta.get(login)
        if row is None:
            continue

        # Keep same filtering rule as your original script
        if row["Campus"] == "Rennes":
            continue

        cursus = cursus_category(row["Dernier diplôme obtenu"])
        note_tp = row["Note test TP"]

        proportions = compute_transition_proportions(
            log_path=os.path.join(group_dir, fname),
            limit_date=limit_date,
            resource_to_value=resource_to_value,
            possible_labels=possible_labels,
            same_value_label_is_single=same_value_label_is_single
        )

        out = {
            "Login_LDAP": login,
            "Cohorte": cohort_name,
            "Cursus": cursus,
            "Note_TP": note_tp
        }
        out.update(proportions)
        rows.append(out)

    return rows


# -------------------------
# CLI + main
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute transition proportions from logs (resource-type or theme transitions)."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ressource",
        action="store_true",
        help="Compute L/P transitions (L-L, L-P, P-L, P-P)."
    )
    group.add_argument(
        "--theme",
        action="store_true",
        help="Compute theme transitions (data, archi_web, tech_env, and cross-theme transitions)."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    mode = "ressource" if args.ressource else "theme"

    # Input paths (same as your current script)
    group1_dir = "data/clean/logs/group1"
    group2_dir = "data/clean/logs/group2"

    csv1 = "data/clean/csv/FIP-Group1_anonymized_cleaned.csv"
    csv2 = "data/clean/csv/FIP-Group2_anonymized_cleaned.csv"

    meta1 = load_metadata(csv1)
    meta2 = load_metadata(csv2)

    out_rows = []
    out_rows += process_group(group1_dir, "Brest 1", meta1, LIMIT_GROUP_1, mode)
    out_rows += process_group(group2_dir, "Brest 2", meta2, LIMIT_GROUP_2, mode)

    # Build output columns based on mode
    base_cols = ["Login_LDAP", "Cohorte", "Cursus", "Note_TP"]

    if mode == "ressource":
        trans_cols = ["L-L", "L-P", "P-L", "P-P"]
        out_path = "docs/proportion-KL-ressource-theme/ressource/proportion_ressource_students.csv"
    else:
        themes = list(PEDAGOGICAL_STRUCTURE.keys())
        cross = [f"{a}-{b}" for a in themes for b in themes if a != b]
        trans_cols = themes + cross
        out_path = "docs/proportion-KL-ressource-theme/theme/proportion_theme_students.csv"

    out_df = pd.DataFrame(out_rows, columns=base_cols + trans_cols)

    # Make sure output folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()