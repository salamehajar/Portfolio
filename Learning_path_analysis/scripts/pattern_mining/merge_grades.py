import pandas as pd

# ==============================
# FICHIERS
# ==============================

SEQUENCES_FILE = "./scripts/pattern_mining/student_sequences.csv"

STUDENTS_CSV_G1 = "./data/clean/csv/FIP-Group1_anonymized_cleaned.csv"
STUDENTS_CSV_G2 = "./data/clean/csv/FIP-Group2_anonymized_cleaned.csv"

OUTPUT = "./scripts/pattern_mining/sequences_with_grades.csv"

# ==============================
# CHARGEMENT DES SEQUENCES
# ==============================

seq = pd.read_csv(SEQUENCES_FILE)

# harmonisation du type
seq["student_id"] = seq["student_id"].astype(str)

# ==============================
# CHARGEMENT DES ETUDIANTS
# ==============================

g1 = pd.read_csv(STUDENTS_CSV_G1)
g2 = pd.read_csv(STUDENTS_CSV_G2)

students = pd.concat([g1, g2], ignore_index=True)

# ==============================
# FILTRE CAMPUS BREST
# ==============================

students["Campus"] = students["Campus"].str.lower()

students = students[
    students["Campus"].str.contains("brest", na=False)
]

# ==============================
# NETTOYAGE NOTE
# ==============================

students["Note test TP"] = pd.to_numeric(
    students["Note test TP"],
    errors="coerce"
)

# ==============================
# CREATION GROUPE
# ==============================

students["group"] = students["Note test TP"].apply(
    lambda x: "success" if x > 12 else "failure"
)

# ==============================
# NORMALISATION ID
# ==============================

students = students.rename(columns={
    "Login_LDAP": "student_id"
})

students["student_id"] = students["student_id"].astype(str)

# ==============================
# MERGE
# ==============================

df = seq.merge(
    students[["student_id", "Note test TP", "group"]],
    on="student_id",
    how="inner"
)

# ==============================
# SAUVEGARDE
# ==============================

df.to_csv(OUTPUT, index=False)

print("Dataset final sauvegardé :", OUTPUT)

print("\nNombre étudiants :", df["student_id"].nunique())

print("\nRépartition groupes :")
print(df["group"].value_counts())