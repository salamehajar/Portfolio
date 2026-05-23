import os
import re
import pandas as pd
from datetime import datetime

# ===============================
# LOGS
# ===============================

LOG_DIRS = [
    "./data/clean/logs/group1",
    "./data/clean/logs/group2"
]

OUTPUT = "./scripts/pattern_mining/student_sequences.csv"

# ===============================
# REGEX LOG
# ===============================

pattern = re.compile(
    r"(\d+)\s+(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}).*?/(.*?)\.html"
)

# ===============================
# STRUCTURE PEDAGOGIQUE
# ===============================

PEDAGOGICAL_STRUCTURE = {
    "data": {
        "lecture": [
            "prerequisite_data","prerequisite_data_2025",
            "lecture_relational_model","lecture_db_integrity",
            "lecture_relational_algebra","lecture_modeling_normalization",
            "lecture_conceptual_modeling","lecture_normalization",
            "lecture_independence_views","lecture_indexation",
            "lecture_transaction","lecture_isolation","lecture_db_tools"
        ],
        "practice": [
            "activity_relational_model","activity_relational_model_correction",
            "activity_conceptual_modeling","activity_conceptual_modeling_correction",
            "activity_conceptual_modeling_advanced",
            "activity_conceptual_modeling_advanced_correction",
            "activity_normalization","activity_normalization_correction",
            "activity_decomposition","activity_decomposition_correction",
            "activity_sql","activity_sql_correction",
            "practice_independence_views","practice_independence_views_correction",
            "practice_sql","practice_sql_correction",
            "practice_advanced_sql","practice_advanced_sql_correction",
            "practice_indexation","practice_indexation_correction",
            "practice_transaction","practice_transaction_correction",
            "practice_isolation","practice_isolation_correction"
        ]
    },

    "archi_web": {
        "lecture": [
            "prerequisite_archiweb","prerequisite_archiweb_2025",
            "web_architecture_home","lecture_technical_architecture",
            "lecture_software_architecture","lecture_dto",
            "lecture_mvc_spring","lecture_jpa","lecture_xml",
            "lecture_validity","lecture_xpath","lecture_parser",
            "lecture_jaxb","lecture_json"
        ],
        "practice": [
            "practice_spring_2tiers","practice_spring_2tiers_ext",
            "practice_jpa","practice_jpa_advanced",
            "practice_xml","practice_schema_xml",
            "practice_xpath","practice_apis","practice_json"
        ]
    },

    "tech_env": {
        "lecture": [
            "prerequisite_techenv","prerequisite_techenv_2025",
            "technical_environment_home","documentation_docker",
            "documentation_rdbms","documentation_programming",
            "spring_bean","inversion_of_control","documentation_si"
        ],
        "practice": [
            "practice_spring","practice_postgresql",
            "practice_docker","practice_docker_advanced"
        ]
    }
}

# ===============================
# CREER SET DES RESSOURCES
# ===============================

VALID_RESOURCES = set()

for theme in PEDAGOGICAL_STRUCTURE.values():
    for part in theme.values():
        VALID_RESOURCES.update(part)


# ===============================
# EXTRACTION RESSOURCE
# ===============================

def extract_resource(page):

    parts = page.split("/")

    for p in reversed(parts):

        if p in VALID_RESOURCES:
            return p

    return None


# ===============================
# PARSER LOGS
# ===============================

records = []

for folder in LOG_DIRS:

    for file in os.listdir(folder):

        if not file.endswith(".log"):
            continue

        path = os.path.join(folder, file)

        with open(path, "r", encoding="utf8") as f:

            for line in f:

                m = pattern.search(line)

                if not m:
                    continue

                sid = m.group(1)
                time = datetime.strptime(m.group(2), "%Y/%m/%d %H:%M:%S")
                page = m.group(3)

                resource = extract_resource(page)

                if resource:

                    records.append({
                        "student_id": sid,
                        "timestamp": time,
                        "page": resource
                    })


df = pd.DataFrame(records)

print("Pages pédagogiques trouvées :", len(df))

# ===============================
# TRIER
# ===============================

df = df.sort_values(["student_id", "timestamp"])

# ===============================
# CREER SEQUENCES
# ===============================

seqs = df.groupby("student_id")["page"].apply(list).reset_index()

# ===============================
# SUPPRIMER DOUBLONS CONSECUTIFS
# ===============================

def remove_duplicates(seq):

    new = []

    for s in seq:

        if not new or new[-1] != s:
            new.append(s)

    return new


seqs["sequence"] = seqs["page"].apply(remove_duplicates)

seqs = seqs.drop(columns="page")

# ===============================
# FILTRE LONGUEUR
# ===============================

seqs = seqs[seqs["sequence"].apply(len) >= 3]

# ===============================
# SAVE
# ===============================

seqs.to_csv(OUTPUT, index=False)

print("Sequences saved:", OUTPUT)
print("Nombre étudiants :", len(seqs))