"""
======================================================================
Application pour visualiser les parcours des étudiants, avec deux curseurs : un pour sélectionner la plage temporelle,
puis un pour visualiser les étapes du parcours au sein de cette plage. Particularité : layout chronologique (de gauche à droite)
Pour run le script : uv run streamlit run scripts/app_graphs_chrono_layout.py
======================================================================
"""

import os
import re
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from collections import defaultdict
import pandas as pd
import networkx as nx
from pyvis.network import Network

# ==========================================
# 1. CONFIGURATION
# ==========================================

INPUT_DIRS = [
    "data/clean/logs/group1", 
    "data/clean/logs/group2" 
]

MIN_STUDY_TIME_SECONDS = 2
MAX_SESSION_GAP_SECONDS = 60 * 60 # on suppose qu'un élève reste au maximum 1h sur une page avant d'aller sur une autre page (d'après les logs)

theme_dict = {
    "tech_env": ["prerequisite_techenv", "prerequisite_techenv_2025", "technical_environment_home", "documentation_docker", "practice_docker", "practice_docker_advanced", "documentation_rdbms", "practice_postgresql", "documentation_programming", "spring_bean", "practice_spring", "inversion_of_control", "documentation_si"],
    "archi_web": ["prerequisite_archiweb", "prerequisite_archiweb_2025", "web_architecture_home", "lecture_technical_architecture", "lecture_software_architecture", "lecture_dto", "lecture_mvc_spring", "practice_spring_2tiers", "practice_spring_2tiers_ext", "lecture_jpa", "practice_jpa", "practice_jpa_advanced", "lecture_xml", "lecture_validity", "practice_xml", "lecture_xpath", "lecture_parser", "lecture_jaxb", "practice_schema_xml", "practice_xpath", "practice_apis", "lecture_json", "practice_json"],
    "data": ["prerequisite_data", "prerequisite_data_2025", "data_home", "lecture_relational_model", "lecture_db_integrity", "activity_relational_model", "activity_relational_model_correction", "lecture_relational_algebra", "lecture_modeling_normalization", "lecture_conceptual_modeling", "activity_conceptual_modeling", "activity_conceptual_modeling_correction", "activity_conceptual_modeling_advanced", "activity_conceptual_modeling_advanced_correction", "lecture_normalization", "activity_normalization", "activity_normalization_correction", "activity_decomposition", "activity_decomposition_correction", "lecture_independence_views", "practice_independence_views", "practice_independence_views_correction", "activity_sql", "activity_sql_correction", "practice_sql", "practice_sql_correction", "practice_advanced_sql", "practice_advanced_sql_correction", "lecture_indexation", "practice_indexation", "practice_indexation_correction", "lecture_transaction", "practice_transaction", "practice_transaction_correction", "lecture_isolation", "practice_isolation", "practice_isolation_correction", "lecture_db_tools"],
    "project": ["project_evaluation", "project_home", "project_part1", "project_part2", "project_part3", "project_presentation", "project_specifications", "project_structureui", "project_task_time"],
    "other": ["index", "licence"]
}

theme_color_map = {
    "tech_env": "#ADD8E6", 
    "archi_web": "#90EE90", 
    "data": "#FA8072",
    "project": "#9370DB", 
    "other": "#D3D3D3"
}

def get_node_theme(node_name, themes):
    for theme, resources in themes.items():
        if node_name in resources: return theme
    return "other" 

# ==========================================
# 2. PARSING
# ==========================================

@st.cache_data
def parse_all_logs(input_dirs):
    log_pattern = re.compile(r'^\d+\s+([\d/:\s]+)\s+"GET\s(.*?)\sHTTP.*"')
    timestamp_format = '%Y/%m/%d %H:%M:%S'
    all_events = []

    for input_dir in input_dirs:
        if not os.path.exists(input_dir): continue
        group_name = os.path.basename(input_dir)
        filenames = [f for f in os.listdir(input_dir) if f.endswith(('.log', '.txt'))]
        
        for filename in filenames:
            student_id = os.path.splitext(filename)[0]
            file_path = os.path.join(input_dir, filename)
            
            parsed_logs = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = log_pattern.match(line.strip())
                        if not match: continue
                        try:
                            timestamp = datetime.strptime(match.group(1), timestamp_format)
                            resource_path = match.group(2)
                            node_name = os.path.splitext(os.path.basename(resource_path))[0]
                            if node_name:
                                parsed_logs.append((timestamp, node_name))
                        except Exception: pass
            except Exception: continue
                
            if len(parsed_logs) < 2: continue
            parsed_logs.sort(key=lambda x: x[0])

            for i in range(len(parsed_logs) - 1):
                timestamp_a, node_a = parsed_logs[i]
                timestamp_b, node_b = parsed_logs[i+1]
                time_delta = (timestamp_b - timestamp_a).total_seconds()
                
                if MIN_STUDY_TIME_SECONDS <= time_delta <= MAX_SESSION_GAP_SECONDS:
                    all_events.append({
                        "student_id": student_id,
                        "group": group_name,
                        "timestamp": timestamp_b,
                        "source": node_a,
                        "target": node_b,
                        "duration_on_source_sec": time_delta
                    })
    
    if not all_events: return pd.DataFrame()
    df = pd.DataFrame(all_events)
    df['student_id'] = df['student_id'].astype(str)
    return df

# ==========================================
# 3. LOGIQUE GRAPHIQUE (DATA & LAYOUTS)
# ==========================================

def calculate_fixed_layout(student_df):
    """
    Layout 1 : organique
    Calcule une position fixe optimisée pour l'espace.
    """
    G_full = nx.from_pandas_edgelist(student_df, 'source', 'target', create_using=nx.DiGraph())
    
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G_full, prog='dot')
    except ImportError:
        pos = nx.spring_layout(G_full, k=0.9, iterations=50, seed=42)
    
    SCALE_FACTOR = 600 
    pos_scaled = {node: (x * SCALE_FACTOR, y * SCALE_FACTOR) for node, (x, y) in pos.items()}
    
    return pos_scaled

def calculate_chronological_layout(student_df):
    """
    Layout 2 : chronologique
    Basé sur la date de première visite.
    """
    sources = student_df[['source', 'timestamp']].rename(columns={'source': 'node'})
    targets = student_df[['target', 'timestamp']].rename(columns={'target': 'node'})
    all_visits = pd.concat([sources, targets])
    
    first_seen = all_visits.groupby('node')['timestamp'].min().sort_values()
    nodes = first_seen.index.tolist()
    
    pos = {}
    X_SPACING = 250 
    Y_AMPLITUDE = 400 
    
    for i, node in enumerate(nodes):
        x = i * X_SPACING
        y = (i % 2 * 2 - 1) * (Y_AMPLITUDE * (0.5 + (i % 3) * 0.2)) 
        
        x -= (len(nodes) * X_SPACING) / 2
        pos[node] = (x, y)
        
    return pos

def get_graph_data(journal_df_filtered):
    if journal_df_filtered.empty:
        return {}, {}

    edge_frequencies = journal_df_filtered.groupby(['source', 'target']).size().to_dict()
    node_total_duration = journal_df_filtered.groupby('source')['duration_on_source_sec'].sum().to_dict(into=defaultdict(float))
    
    all_nodes = set()
    for s, t in edge_frequencies.keys():
        all_nodes.add(s)
        all_nodes.add(t)
    
    for node in all_nodes:
        if node not in node_total_duration:
            node_total_duration[node] = 0.0
            
    return edge_frequencies, node_total_duration

# ==========================================
# 4. VISUALISATION (PYVIS)
# ==========================================

def render_pyvis_graph(edge_frequencies, node_total_duration, fixed_pos):
    
    nt = Network(height="750px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")
    nt.toggle_physics(False) 

    # --- A. NOEUDS ---
    if node_total_duration:
        durations = list(node_total_duration.values())
        min_d, max_d = min(durations), max(durations)
    else:
        min_d, max_d = 0, 0

    for node, duration in node_total_duration.items():
        theme = get_node_theme(node, theme_dict)
        color = theme_color_map.get(theme, "#D3D3D3")
        
        if max_d > min_d:
            size = 15 + 30 * (duration - min_d) / (max_d - min_d)
        else:
            size = 25
        
        x, y = 0, 0
        if node in fixed_pos:
            x, y = fixed_pos[node]

        nt.add_node(
            node,
            label=node,
            title=f"{node}\nTemps total: {duration:.0f}s",
            color=color,
            size=size,
            x=x, 
            y=y,
            borderWidth=1,
            font={'size': 20, 'face': 'arial', 'color': 'black', 'strokeWidth': 3, 'strokeColor': 'white'} 
        )

    # --- B. ARÊTES ---
    if edge_frequencies:
        freqs = list(edge_frequencies.values())
        min_f, max_f = min(freqs), max(freqs)
    else:
        min_f, max_f = 0, 0

    for (source, target), freq in edge_frequencies.items():
        if max_f > min_f:
            width = 1 + 7 * (freq - min_f) / (max_f - min_f)
        else:
            width = 2

        nt.add_edge(
            source,
            target,
            width=width,
            title=f"Fréquence: {freq}",
            color='#555555',
            smooth={'type': 'curvedCW', 'roundness': 0.2} 
        )
    
    # --- C. OPTIONS ---
    nt.set_options("""
    var options = {
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      },
      "physics": {
        "enabled": false
      }
    }
    """)
    
    path = 'graphs/temp_graph.html'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    nt.save_graph(path)
    return path

# ==========================================
# 5. MAIN
# ==========================================

def main():
    st.set_page_config(page_title="Parcours Étudiants", layout="wide")
    st.title("Parcours dynamique des étudiants")

    # 1. Chargement
    with st.spinner("Chargement des logs..."):
        journal_df = parse_all_logs(INPUT_DIRS)
    
    if journal_df is None or journal_df.empty:
        st.error("Aucun log trouvé.")
        return

    # 2. Filtres
    with st.sidebar:
        st.header("Filtres")

        group_filter = st.radio("Groupe :", ["Tous", "Groupe 1", "Groupe 2"], index=0)
        if group_filter == "Groupe 1":
            filtered_df = journal_df[journal_df['group'] == 'group1']
        elif group_filter == "Groupe 2":
            filtered_df = journal_df[journal_df['group'] == 'group2']
        else:
            filtered_df = journal_df

        student_list = sorted(filtered_df['student_id'].unique())
        if not student_list:
            st.warning("Aucun étudiant pour ce filtre.")
            return

        selected_student_id = st.selectbox("Étudiant :", student_list, key="student_select")
        
        st.markdown("---")
        st.subheader("Disposition")
        layout_type = st.radio("Type de carte :", ["Organique (Optimisé)", "Chronologique (Découverte)"])
        
    # 3. Calcul du Layout
    student_journal = filtered_df[filtered_df['student_id'] == selected_student_id].copy()
    student_journal = student_journal.sort_values("timestamp").reset_index(drop=True)
    n_events = len(student_journal)
    
    if ('current_student' not in st.session_state or 
        st.session_state.current_student != selected_student_id or
        'current_layout' not in st.session_state or
        st.session_state.current_layout != layout_type):
        
        st.session_state.current_student = selected_student_id
        st.session_state.current_layout = layout_type
        
        with st.spinner(f"Calcul de la carte ({layout_type})..."):
            if layout_type == "Organique (Optimisé)":
                st.session_state.fixed_pos = calculate_fixed_layout(student_journal)
            else:
                st.session_state.fixed_pos = calculate_chronological_layout(student_journal)
    
    fixed_pos = st.session_state.fixed_pos

    min_date_student = student_journal["timestamp"].min().to_pydatetime()
    max_date_student = student_journal["timestamp"].max().to_pydatetime()

    with st.sidebar:
        st.markdown("---")
        st.subheader("Fenêtre temporelle")

        date_min, date_max = st.slider(
            "Plage de dates :",
            min_value=min_date_student,
            max_value=max_date_student,
            value=(min_date_student, max_date_student),
            format="DD/MM/YY"
        )

        start_ts = pd.to_datetime(date_min)
        end_ts = pd.to_datetime(date_max)

        journal_in_range = student_journal[
            (student_journal["timestamp"] >= start_ts) &
            (student_journal["timestamp"] <= end_ts)
        ].reset_index(drop=True)

        n_events_range = len(journal_in_range)

        if n_events_range == 0:
            st.info("Aucune activité pour cet étudiant dans cette plage.")
            return

        st.subheader("Étapes dans la plage")
        step_index = st.slider("Étape :", 1, n_events_range, n_events_range, step=1)

        current_event = journal_in_range.iloc[step_index - 1]
        st.caption(f"**{current_event['timestamp'].strftime('%d/%m %H:%M')}** : {current_event['source']} → {current_event['target']}")

    # 4. Rendu
    journal_at_time = journal_in_range.iloc[:step_index]
    edge_freqs, node_durs = get_graph_data(journal_at_time)

    if not node_durs:
        st.info("Pas d'activité à cette étape.")
        return

    html_file = render_pyvis_graph(edge_freqs, node_durs, fixed_pos)
    with open(html_file, 'r', encoding='utf-8') as f:
        source_code = f.read()
    components.html(source_code, height=750, scrolling=False)

if __name__ == "__main__":
    main()
