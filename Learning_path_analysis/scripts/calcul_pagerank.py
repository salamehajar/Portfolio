"""
======================================================================
ANALYSE PAGERANK : IMPORTANCE DES PAGES & PROFILAGE ÉTUDIANT
INCLUT LE CLASSEMENT PAR GROUPE, PAR CURSUS ET PAR PERFORMANCE (RÉUSSITE VS ÉCHEC)
======================================================================
"""

import os
import re
import pandas as pd
import networkx as nx
from datetime import datetime
from collections import Counter

# ==========================================
# 1. CONFIGURATION
# ==========================================

DIR_GROUP_1 = "./data/clean/logs/group1"
DIR_GROUP_2 = "./data/clean/logs/group2"

# Fichiers étudiants (pour récupérer les cursus et les notes)
STUDENTS_CSV_G1 = "./data/clean/csv/FIP-Group1_anonymized_cleaned.csv"
STUDENTS_CSV_G2 = "./data/clean/csv/FIP-Group2_anonymized_cleaned.csv"

# DATES LIMITES (CUTOFF)
CUTOFF_G1 = datetime(2024, 2, 20, 23, 59, 59)
CUTOFF_G2 = datetime(2025, 2, 21, 23, 59, 59)

# Fichiers de sortie
OUTPUT_PAGES_CSV = "./docs/résultats pagerank/pagerank_pages_importance.csv"
OUTPUT_STUDENTS_CSV = "./docs/résultats pagerank/pagerank_students_scores.csv"
OUTPUT_CURSUS_PAGES_CSV = "./docs/résultats pagerank/pagerank_pages_importantes_cursus.csv"
OUTPUT_PERF_PAGES_CSV = "./docs/résultats pagerank/pagerank_pages_importantes_performance.csv" 

MIN_STUDY_TIME = 2
MAX_SESSION_GAP = 3600

# Mots clés à exclure des noms de pages
MOTS_EXCLUS = ['home', 'project', 'index', 'favicon','prerequisite']

# ==========================================
# 2. CHARGEMENT DES DONNÉES ÉTUDIANTS (CURSUS + NOTES)
# ==========================================

def load_student_data():
    """Charge les CSV et associe chaque ID étudiant à son diplôme et sa note."""
    student_data = {}
    
    for file in [STUDENTS_CSV_G1, STUDENTS_CSV_G2]:
        if os.path.exists(file):
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                student_id = str(row['Login_LDAP'])
                cursus = str(row['Dernier diplôme obtenu'])
                
                # Récupération de la note avec gestion des virgules et valeurs nulles
                try:
                    note_str = str(row.get('Note test TP', -1)).replace(',', '.')
                    note = float(note_str)
                except ValueError:
                    note = -1.0 # Valeur par défaut si la note est absente ou illisible
                
                student_data[student_id] = {
                    'cursus': cursus,
                    'note': note
                }
        else:
            print(f"⚠️ Fichier étudiant introuvable : {file}")
            
    return student_data

# ==========================================
# 3. LECTURE DES LOGS
# ==========================================

def get_transitions_from_log(file_path, cutoff_date=None):
    """
    Lit un log et retourne les transitions (source -> target) et les pages.
    Applique le filtrage temporel et l'exclusion de mots-clés.
    """
    transitions = []
    logs = []
    log_pattern = re.compile(r'^\d+\s+([\d/:\s]+)\s+"GET\s(.*?)\sHTTP.*"')
    timestamp_format = '%Y/%m/%d %H:%M:%S'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = log_pattern.match(line.strip())
                if m:
                    try:
                        ts = datetime.strptime(m.group(1), timestamp_format)
                        
                        if cutoff_date and ts > cutoff_date:
                            continue
                        res = os.path.splitext(os.path.basename(m.group(2)))[0]
                        
                        if res:
                            res_lower = res.lower()
                            if not any(mot in res_lower for mot in MOTS_EXCLUS):
                                logs.append((ts, res))
                                
                    except: 
                        pass
    except: 
        return [], []

    if not logs: 
        return [], []
    
    logs.sort(key=lambda x: x[0])
    all_pages = [x[1] for x in logs]

    for i in range(len(logs)-1):
        t1, n1 = logs[i]
        t2, n2 = logs[i+1]
        delta = (t2 - t1).total_seconds()
        
        if MIN_STUDY_TIME <= delta <= MAX_SESSION_GAP:
            transitions.append((n1, n2))
            
    return transitions, all_pages

# ==========================================
# 4. ORCHESTRATION PRINCIPALE
# ==========================================

def main():
    print("--- DÉMARRAGE DE L'ANALYSE GLOBALE ---")
    
    student_data_map = load_student_data()
    
    all_pages_stats = []
    all_students_stats = []
    all_cursus_stats = []
    all_perf_stats = [] 
    
    # Dictionnaire de Graphes par cursus et par performance
    graphs_by_cursus = {}
    graphs_by_performance = {
        'Réussite': nx.DiGraph(),
        'Échec': nx.DiGraph()
    }

    def process_directory(group_name, directory, cutoff_date):
        print(f"\n--- Traitement {group_name} ---")
        if not os.path.exists(directory):
            print(f"   Dossier introuvable : {directory}")
            return

        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.log')]
        if not files:
            print("   Aucun fichier log trouvé.")
            return

        G_group = nx.DiGraph()
        student_logs_data = {}

        # 1. Extraction des données et construction des graphes
        for f in files:
            student_id = os.path.splitext(os.path.basename(f))[0]
    
            if not student_id.isdigit() :
                continue
            
            student_info = student_data_map.get(student_id, {"cursus": "Inconnu", "note": -1.0})
            cursus = student_info["cursus"]
            note = student_info["note"]
            
            transitions, pages = get_transitions_from_log(f, cutoff_date)
            student_logs_data[student_id] = {"pages": pages, "cursus": cursus}
            
            if transitions:
                # Ajout au Graphe du Groupe
                for u, v in transitions:
                    if G_group.has_edge(u, v):
                        G_group[u][v]['weight'] += 1
                    else:
                        G_group.add_edge(u, v, weight=1)
                        
                # Ajout au Graphe du Cursus
                if cursus not in graphs_by_cursus:
                    graphs_by_cursus[cursus] = nx.DiGraph()
                    
                G_cursus = graphs_by_cursus[cursus]
                for u, v in transitions:
                    if G_cursus.has_edge(u, v):
                        G_cursus[u][v]['weight'] += 1
                    else:
                        G_cursus.add_edge(u, v, weight=1)

                # Ajout au Graphe de Performance (Réussite / Échec)
                perf_category = "Réussite" if note >= 12 else "Échec"
                G_perf = graphs_by_performance[perf_category]
                
                for u, v in transitions:
                    if G_perf.has_edge(u, v):
                        G_perf[u][v]['weight'] += 1
                    else:
                        G_perf.add_edge(u, v, weight=1)

        # 2. PageRank du Groupe et export des pages
        if len(G_group.nodes) == 0:
            print("   Pas assez de données pour le PageRank de ce groupe.")
            return

        pr_scores = nx.pagerank(G_group, weight='weight', alpha=0.85)
        
        sorted_pages = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
        rank = 1
        for page, score in sorted_pages:
            all_pages_stats.append({
                "Group": group_name,
                "Rank": rank,
                "Page": page,
                "PageRank_Score": round(score, 4),
                "In_Degree": G_group.in_degree(page, weight='weight')
            })
            rank += 1

        # 3. Évaluation des étudiants (basée sur le PageRank du Groupe)
        max_pr_value = max(pr_scores.values()) if pr_scores else 1
        
        for student_id, data in student_logs_data.items():
            pages = data["pages"]
            if not pages: 
                continue

            page_counts = Counter(pages)
            most_frequent_page = page_counts.most_common(1)[0][0]
            fav_page_pr_score = pr_scores.get(most_frequent_page, 0)
            
            total_pr_accumulated = sum([pr_scores.get(p, 0) for p in pages])
            avg_nav_quality = total_pr_accumulated / len(pages) if len(pages) > 0 else 0

            all_students_stats.append({
                "Student_ID": student_id,
                "Group": group_name,
                "Cursus": data["cursus"],
                "Total_Pages_Viewed": len(pages),
                "Most_Frequent_Page": most_frequent_page,
                "Freq_Page_Global_Rank": round(fav_page_pr_score, 4),
                "Avg_Navigation_Value": round(avg_nav_quality, 4),
                "Normalized_Score": round(avg_nav_quality / max_pr_value, 2)
            })

    # Lancement des deux groupes
    process_directory("Group_1", DIR_GROUP_1, CUTOFF_G1)
    process_directory("Group_2", DIR_GROUP_2, CUTOFF_G2)

    # 4. PageRank par Cursus
    print("\n--- Calcul du PageRank par Cursus ---")
    for cursus, G_cursus in graphs_by_cursus.items():
        if len(G_cursus.nodes) == 0 or G_cursus.number_of_edges() < 5:
            continue
            
        pr_scores_cursus = nx.pagerank(G_cursus, weight='weight', alpha=0.85)
        sorted_pages_cursus = sorted(pr_scores_cursus.items(), key=lambda x: x[1], reverse=True)
        
        rank = 1
        for page, score in sorted_pages_cursus:
            all_cursus_stats.append({
                "Cursus": cursus,
                "Rank": rank,
                "Page": page,
                "PageRank_Score": round(score, 4),
                "In_Degree": G_cursus.in_degree(page, weight='weight')
            })
            rank += 1

    # 5. PageRank par Performance (Réussite / Échec)
    print("--- Calcul du PageRank par Performance (Réussite vs Échec) ---")
    for perf_category, G_perf in graphs_by_performance.items():
        if len(G_perf.nodes) == 0 or G_perf.number_of_edges() < 5:
            continue
            
        pr_scores_perf = nx.pagerank(G_perf, weight='weight', alpha=0.85)
        sorted_pages_perf = sorted(pr_scores_perf.items(), key=lambda x: x[1], reverse=True)
        
        rank = 1
        for page, score in sorted_pages_perf:
            all_perf_stats.append({
                "Performance_Group": perf_category,
                "Rank": rank,
                "Page": page,
                "PageRank_Score": round(score, 4),
                "In_Degree": G_perf.in_degree(page, weight='weight')
            })
            rank += 1

    # ==========================================
    # 6. SAUVEGARDE
    # ==========================================
    os.makedirs(os.path.dirname(OUTPUT_PAGES_CSV), exist_ok=True)
    
    if all_pages_stats:
        pd.DataFrame(all_pages_stats).to_csv(OUTPUT_PAGES_CSV, index=False, sep=';')
        print(f"✅ Classement des pages (par Groupe) : {OUTPUT_PAGES_CSV}")

    if all_cursus_stats:
        pd.DataFrame(all_cursus_stats).to_csv(OUTPUT_CURSUS_PAGES_CSV, index=False, sep=';')
        print(f"✅ Classement des pages (par Cursus) : {OUTPUT_CURSUS_PAGES_CSV}")

    if all_students_stats:
        pd.DataFrame(all_students_stats).to_csv(OUTPUT_STUDENTS_CSV, index=False, sep=';')
        print(f"✅ Comparatif étudiants : {OUTPUT_STUDENTS_CSV}")
        
    if all_perf_stats:
        pd.DataFrame(all_perf_stats).to_csv(OUTPUT_PERF_PAGES_CSV, index=False, sep=';')
        print(f"✅ Classement des pages (Réussite vs Échec) : {OUTPUT_PERF_PAGES_CSV}")

if __name__ == "__main__":
    main()