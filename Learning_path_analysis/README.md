# PROCOM - Learning Path Analysis

## Présentation générale

Ce projet vise à analyser les parcours d’apprentissage d’étudiants au sein d’une matière composée de multiples ressources pédagogiques (cours, vidéos, exercices, évaluations, etc.).

Lien du Cloud Google Drive : https://drive.google.com/drive/folders/1iW8s6ryGjrmPwcdUC-vZ-BXaBNdVFuj_?usp=sharing

---

## Objectifs du projet

Les trois principaux objectifs sont :

- Représenter graphiquement les logs pour visualiser les parcours d'apprentissage.
- Relier le parcours d'apprentissage des étudiants à des indicateurs de réussite (ou d'échec).
- Faire un retour aux enseignants sur la manière dont les étudiants appréhendent l'UE afin de mieux les guider.

---

## Technologies utilisées

- **Python 3**
- **uv** pour la gestion des dépendances et des environnements
- Bibliothèques d’analyse de données et de graphes (ex. : pandas, networkx, numpy – selon configuration)
- Librairie **Streamlit** pour créer les différentes applications

---

## Prérequis

- Python 3.10 ou supérieur
- Git
- uv (gestionnaire d’environnement et de dépendances)

---

## 📂 Structure du projet

```text
procom-lpa/
├── Bibliography/       # Sources et références bibliographiques
├── data/               # Données brutes ou traitées
├── docs/               # Fichiers csv utiles pour les applications
├── graphs/             # Graphiques et visualisations générés
├── reports/            # Rapports finaux ou intermédiaires
├── scripts/            # Scripts Python (incluant les 2 applications)
├── .gitignore          # Fichiers et dossiers à exclure de Git
├── .python-version     # Version de Python utilisée par uv
├── LICENSE             # Licence du projet
├── pyproject.toml      # Configuration et dépendances du projet
├── README.md           # Documentation principale
└── uv.lock             # Fichier de verrouillage des dépendances
```

---

## Installation et mise en place de l’environnement

### 1. Installation de uv

Installer `uv` à l’aide du script officiel :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Récupération du projet

Cloner le dépôt Git puis se placer dans le répertoire du projet :
```bash
git clone https://gitlab.imt-atlantique.fr/h23salam/procom-lpa.git
cd procom-lpa
```

### 3. Création de l’environnement et installation des dépendances

Le projet est configuré via pyproject.toml et uv.lock. Pour récupérer les dépendances associées au projet :
```bash
uv sync --dev
```
uv crée automatiquement un environnement virtuel isolé dans le dossier caché `.venv`.
 
---

## Visualisation des parcours (application Streamlit)

Une application Streamlit est fournie pour **visualiser le parcours d’un étudiant sous forme de graphe orienté**, avec :
- un **filtre par groupe** et **sélection d’étudiant** ;
- une **fenêtre temporelle** (plage de dates) ;
- un **curseur d’étapes** pour rejouer progressivement le parcours.
- un **choix** sur la disposition du graphe (layout) : layout classique, ou layout chronologique (comme une frise).

Le script se trouve ici : `scripts/app_graphs_chrono_layout.py`

Pour lancer le script, effectuer la commande suivante depuis la racine du dépôt :
```
uv run streamlit run scripts/app_graphs_chrono_layout.py
```
Streamlit affichera ensuite une URL locale (http://localhost:8501) à ouvrir dans le navigateur. 

---

## Analyse des séquences ressource-ressource et thème-thème (application Streamlit)

Le script se trouve ici : `scripts/app_passage_ressource_theme.py`

Pour lancer le script, effectuer la commande suivante depuis la racine du dépôt :
```
uv run streamlit run scripts/app_passage_ressource_theme.py
```
Streamlit affichera ensuite une URL locale à ouvrir dans le navigateur. 
