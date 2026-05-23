# -*- coding: utf-8 -*-
"""
Script pour séparer les logs étudiants des groupes 1 et 2.
Chaque groupe est traité séparément, et les fichiers étudiants sont placés
dans procom-lpa/data/clean/logs/<nom_du_groupe>/
"""

import os

# --- CONFIGURATION DES CHEMINS ---
# Chemin relatif vers les logs
BASE_DIR = "data/clean/logs"

# --- TRAITEMENT DES GROUPES ---
groups = ["group1", "group2"]

for group in groups:
    input_file = os.path.join(BASE_DIR, group, "students.log")
    output_dir = os.path.join(BASE_DIR, group)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Traitement de {group} ...")
    print(f"  Lecture : {input_file}")
    print(f"  Sortie  : {output_dir}")

    files = {}

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                student_id = line.split('\t', 1)[0]

                if student_id not in files:
                    files[student_id] = open(os.path.join(output_dir, f"{student_id}.log"), "w", encoding="utf-8")

                files[student_id].write(line + "\n")

        print(f"Groupe {group} traité avec succès ({len(files)} fichiers créés).")

    except FileNotFoundError:
        print(f"Le fichier {input_file} est introuvable, groupe ignoré.")
    finally:
        for f in files.values():
            f.close()
