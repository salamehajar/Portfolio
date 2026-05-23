"""
Script : clean-move-logs.py

Usage :
    python3 scripts/clean-move-logs.py data/raw/groupX chemin/vers/etudiants.csv

Fonctions :
- Parcourt tous les .log dans data/raw/groupX/logs/
- Extrait uniquement les lignes contenant ".html" ET dont la 1ère colonne (ID) est un login présent dans le CSV (colonne "Login")
- Écrit ces lignes dans data/clean/log/groupX/students.log
- Nettoie students.log pour supprimer l'IP et les 3 valeurs qui suivent (ne garde que : ID, date/heure, "GET ...")
- Affiche à la fin la liste des logins du CSV n’ayant aucune ligne dans students.log
"""

import os
import sys
import csv

def get_group_name(input_path):
    """
    Extrait le nom du groupe à partir du chemin donné.
    Exemple : data/raw/group1 -> group1
    """
    return os.path.basename(os.path.normpath(input_path))


def find_log_files(group_path):
    """
    Parcourt le dossier du groupe et retourne la liste complète
    des chemins de fichiers .log présents dans le dossier logs/.
    """
    logs_dir = os.path.join(group_path, "logs")
    if not os.path.exists(logs_dir):
        print(f"Le dossier {logs_dir} n'existe pas.")
        return []
    return [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")]

def load_logins_from_csv(csv_path):
    """
    Charge la liste des logins depuis un CSV séparé par des virgules.
    Attend une colonne nommée 'Login'. Retourne un set.
    """
    logins = set()
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=",")
        if "Login_LDAP" not in (reader.fieldnames or []):
            print("La colonne 'Login' est introuvable dans le CSV.")
            sys.exit(1)
        for row in reader:
            login = (row.get("Login_LDAP") or "").strip()
            if login:
                logins.add(login)

    return logins


def extract_html_lines(log_file, allowed_logins):
    """
    Lit un fichier .log et retourne la liste des lignes contenant '.html'.
    """
    html_lines = []
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            first_field = parts[0].strip()
            # Vérifie que .html est présent et que la ligne commence par l'ID affectée à un étudiant
            if len(parts) > 0 and first_field in allowed_logins and ".html" in line:
                html_lines.append(line)
    return html_lines


def save_cleaned_logs(group_name, html_lines):
    """
    Sauvegarde les lignes extraites dans le dossier de sortie correspondant.
    Exemple : data/clean/log/group1/students.log
    """
    output_dir = os.path.join("data", "clean", "logs", group_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "students.log")

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(html_lines)

    print(f"Fichier brut sauvegardé dans : {output_file}")
    return output_file

def clean_final_log(file_path):
    """
    Nettoie le fichier students.log en supprimant l'adresse IP et les 3 valeurs qui suivent.
    Exemple :
    Entrée  : ID    Date Heure    192.168.1.1    200    3    4442    "GET url.html HTTP/1.1"
    Sortie  : ID    Date Heure    "GET url.html HTTP/1.1"
    """
    cleaned_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            # Vérifie qu'il y a assez de colonnes avant de nettoyer
            if len(parts) >= 7:
                cleaned_line = "\t".join(parts[:2] + [parts[-1]]) + "\n"
                cleaned_lines.append(cleaned_line)
            else:
                # Si la ligne ne correspond pas, on la garde telle quelle
                cleaned_lines.append(line)

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Fichier students.log nettoyé : {file_path}")

def list_missing_logins(students_log_path, all_logins):
    """
    Affiche les logins du CSV qui n'ont aucune ligne dans students.log.
    """
    present = set()
    if os.path.exists(students_log_path):
        with open(students_log_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    present.add(parts[0].strip())

    missing = sorted(all_logins - present)
    if missing:
        print("\nÉtudiants sans aucune ligne dans students.log :")
        for m in missing:
            print(m)
    else:
        print("\nTous les logins du CSV sont présents au moins une fois dans students.log.")


if __name__ == "__main__":
    # Vérification du nombre d'arguments passés en entrée
    if len(sys.argv) != 3:
        print("Utilisation : python3 scripts/clean-move-logs.py data/raw/groupX chemin/vers/etudiants.csv")
        sys.exit(1)

    # Arguments
    input_path = sys.argv[1]
    csv_path = sys.argv[2]

    # Outputs des fonctions
    group_name = get_group_name(input_path)
    allowed_logins = load_logins_from_csv(csv_path)
    log_files = find_log_files(input_path)

    if not log_files:
        print("Aucun fichier .log trouvé.")
        sys.exit(0)

    # Extraction des lignes contenant ".html"
    all_html_lines = []
    for log_file in log_files:
        html_lines = extract_html_lines(log_file, allowed_logins)
        all_html_lines.extend(html_lines)

    output_file = save_cleaned_logs(group_name, all_html_lines)

    # Nettoyage du fichier final students.log
    clean_final_log(output_file)

    # Liste des étudiants qui ne sont pas présents dans students.log
    list_missing_logins(output_file, allowed_logins)
