"""
======================================================================
Application d'analyse et de visualisation des scores PageRank
pour les ressources, les cursus et les performances étudiantes.
======================================================================
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Tes fichiers de résultats
PAGERANK_PAGES_CSV = "./docs/résultats pagerank/pagerank_pages_importance.csv"
PAGERANK_CURSUS_CSV = "./docs/résultats pagerank/pagerank_pages_importantes_cursus.csv"
PAGERANK_STUDENTS_CSV = "./docs/résultats pagerank/pagerank_students_scores.csv"
PAGERANK_PERF_CSV = "./docs/résultats pagerank/pagerank_pages_importantes_performance.csv"

# Les fichiers d'origine pour récupérer les notes 
STUDENTS_CSV_G1 = "./data/clean/csv/FIP-Group1_anonymized_cleaned.csv"
STUDENTS_CSV_G2 = "./data/clean/csv/FIP-Group2_anonymized_cleaned.csv"

# Dossier de sortie pour les images
OUTPUT_DIR = "./docs/résultats pagerank/visualisations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration du style visuel
sns.set_theme(style="whitegrid", context="talk")

def load_and_merge_student_data():
    """Charge les scores PageRank et y fusionne les notes académiques"""
    df_scores = pd.read_csv(PAGERANK_STUDENTS_CSV, sep=';')
    df_scores['Student_ID'] = df_scores['Student_ID'].astype(str)

    df_g1 = pd.read_csv(STUDENTS_CSV_G1)
    df_g1['Group'] = 'Group_1'
    df_g2 = pd.read_csv(STUDENTS_CSV_G2)
    df_g2['Group'] = 'Group_2'
    
    df_profils = pd.concat([df_g1, df_g2], ignore_index=True)
    df_profils.rename(columns={'Login_LDAP': 'Student_ID'}, inplace=True)
    df_profils['Student_ID'] = df_profils['Student_ID'].astype(str)
    
    df_merged = pd.merge(df_scores, df_profils[['Student_ID', 'Group', 'Note test TP']], on=['Student_ID', 'Group'], how='inner')
    return df_merged

def main():
    print("--- GÉNÉRATION DES VISUALISATIONS ---")
    
    # =========================================================
    # A1. LE PALMARÈS GLOBAL DES RESSOURCES (BAR CHART)
    # =========================================================
    print("Génération du graphique : Top 10 des ressources globales...")
    df_pages = pd.read_csv(PAGERANK_PAGES_CSV, sep=';')
    
    top_global = df_pages.groupby('Page')['PageRank_Score'].mean().reset_index()
    top_global = top_global.sort_values(by='PageRank_Score', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_global, x='PageRank_Score', y='Page', palette='viridis')
    plt.title("Top 10 des ressources les plus consultées (Score Global)")
    plt.xlabel("Importance (Score PageRank moyen)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/A1_Top_10_Ressources.png")
    plt.close()

    # =========================================================
    # A2. HEATMAP DES RANGS PAR CURSUS
    # =========================================================
    print("Génération de la Heatmap des parcours...")
    df_cursus_pages = pd.read_csv(PAGERANK_CURSUS_CSV, sep=';')
    
    top_15_pages = df_pages.groupby('Page')['PageRank_Score'].mean().nlargest(15).index
    df_heatmap = df_cursus_pages[df_cursus_pages['Page'].isin(top_15_pages)]
    pivot_heatmap = df_heatmap.pivot(index='Page', columns='Cursus', values='Rank')
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_heatmap, annot=True, cmap="YlGn_r", cbar_kws={'label': 'Rang (1 = Meilleur)'}, fmt="g", linewidths=.5)
    plt.title("Classement des 15 meilleures ressources selon le cursus d'origine")
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/A2_Heatmap_Ressources_Cursus.png")
    plt.close()

    # =========================================================
    # B. INTENSITÉ DE L'EFFORT VS CURSUS (BOXPLOTS)
    # =========================================================
    print("Génération des profils de navigation (Boxplots)...")
    df_merged = load_and_merge_student_data()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.boxplot(data=df_merged, x='Total_Pages_Viewed', y='Cursus', ax=axes[0], palette='pastel')
    axes[0].set_title("Volume d'activité (Pages Vues)")
    axes[0].set_xlabel("Nombre total de pages consultées")
    axes[0].set_ylabel("")

    sns.boxplot(data=df_merged, x='Avg_Navigation_Value', y='Cursus', ax=axes[1], palette='pastel')
    axes[1].set_title("Qualité de la navigation (Score PageRank)")
    axes[1].set_xlabel("Valeur moyenne des ressources consultées")
    axes[1].set_ylabel("")

    plt.suptitle("Profils de navigation selon le cursus", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/B_Boxplots_Profils_Navigation.png")
    plt.close()

    # =========================================================
    # C. COMPARAISON TOP 10 PAR CURSUS (BAR CHARTS MULTIPLES)
    # =========================================================
    print("Génération du graphique : Top 10 par Cursus...")
    if os.path.exists(PAGERANK_CURSUS_CSV):
        df_cursus = pd.read_csv(PAGERANK_CURSUS_CSV, sep=';')
        
        # Identifier tous les cursus disponibles
        cursus_list = df_cursus['Cursus'].unique()
        n_cursus = len(cursus_list)
        
        # Calculer le nombre de lignes nécessaires (2 graphiques par ligne)
        cols = 2
        rows = math.ceil(n_cursus / cols)
        
        # Créer une figure dynamique
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows), squeeze=False)
        axes = axes.flatten()
        
        # Palette de couleurs distinctes pour les cursus
        palette = sns.color_palette("husl", n_cursus)
        
        for i, cursus in enumerate(cursus_list):
            top_cursus = df_cursus[df_cursus['Cursus'] == cursus].nlargest(10, 'PageRank_Score')
            
            sns.barplot(data=top_cursus, x='PageRank_Score', y='Page', ax=axes[i], color=palette[i])
            axes[i].set_title(f"Top 10 des pages : {cursus}", fontsize=16, fontweight='bold', color=palette[i])
            axes[i].set_xlabel("Importance (Score PageRank)")
            axes[i].set_ylabel("")
            
        # Effacer les cadres vides si le nombre de cursus est impair
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/C_Top10_par_Cursus.png")
        plt.close()
    else:
        print(f"⚠️ Le fichier {PAGERANK_CURSUS_CSV} est introuvable. Graphique C ignoré.")

    # =========================================================
    # D. COMPARAISON RÉUSSITE VS ÉCHEC (TOP 10)
    # =========================================================
    print("Génération du graphique : Top 10 Réussite vs Échec...")
    if os.path.exists(PAGERANK_PERF_CSV):
        df_perf = pd.read_csv(PAGERANK_PERF_CSV, sep=';')

        # Extraction du Top 10 pour chaque groupe
        top_reussite = df_perf[df_perf['Performance_Group'] == 'Réussite'].nlargest(10, 'PageRank_Score')
        top_echec = df_perf[df_perf['Performance_Group'] == 'Échec'].nlargest(10, 'PageRank_Score')

        # Création de la figure (2 graphiques côte à côte)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True)

        # Graphique de Gauche : RÉUSSITE
        sns.barplot(data=top_reussite, x='PageRank_Score', y='Page', ax=axes[0], color='#2ca02c') 
        axes[0].set_title("Top 10 des pages : RÉUSSITE", fontsize=16, fontweight='bold', color='#2ca02c')
        axes[0].set_xlabel("Importance (Score PageRank)")
        axes[0].set_ylabel("")

        # Graphique de Droite : ÉCHEC
        sns.barplot(data=top_echec, x='PageRank_Score', y='Page', ax=axes[1], color='#d62728') 
        axes[1].set_title("Top 10 des pages : ÉCHEC", fontsize=16, fontweight='bold', color='#d62728')
        axes[1].set_xlabel("Importance (Score PageRank)")
        axes[1].set_ylabel("")

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/D_Top10_Reussite_vs_Echec.png")
        plt.close()
    else:
        print(f"⚠️ Le fichier {PAGERANK_PERF_CSV} est introuvable. Graphique D ignoré.")

    print(f"\n✅ TERMINÉ ! 5 graphiques ont été générés dans le dossier : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()