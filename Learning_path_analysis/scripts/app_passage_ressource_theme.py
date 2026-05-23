"""
======================================================================
Application pour visualiser les transitions ressource / thème,
avec une vue d'ensemble et une vue étudiant.
Pour run le script : uv run streamlit run scripts/app_passage_ressource_theme.py
======================================================================
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

st.set_page_config(page_title="Resource / Theme transitions", layout="wide")

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_ROOT = "docs/proportion-KL-ressource-theme"
META_COLUMNS = {"Login_LDAP", "Cohorte", "Cursus", "Note_TP", "Student_ID", "Groupe", "Effectif"}
PATTERNS = ["", "/", "\\", "x", ".", "|", "+", "-"]

# ==========================================
# 2. HELPERS / CHARGEMENT DES DONNÉES
# ==========================================

@st.cache_data
def read_csv_auto(path):
    """Read a CSV file with automatic separator detection."""
    return pd.read_csv(path, sep=None, engine="python")


def get_file_paths(mode):
    """Return the 4 file paths for the selected mode."""
    mode = mode.lower()

    if mode == "resource":
        folder = "ressource"
        return {
            "students": f"{DATA_ROOT}/{folder}/proportion_ressource_students.csv",
            "cursus": f"{DATA_ROOT}/{folder}/proportion_ressource_cursus.csv",
            "kl_cursus": f"{DATA_ROOT}/{folder}/divergence_KL_ressource_cursus.csv",
            "success": f"{DATA_ROOT}/{folder}/proportion_theme_sucess.csv",
        }

    folder = "theme"
    return {
        "students": f"{DATA_ROOT}/{folder}/proportion_theme_students.csv",
        "cursus": f"{DATA_ROOT}/{folder}/proportion_theme_cursus.csv",
        "kl_cursus": f"{DATA_ROOT}/{folder}/divergence_KL_theme_cursus.csv",
        "success": f"{DATA_ROOT}/{folder}/proportion_theme_sucess.csv",
    }


@st.cache_data
def load_mode_data(mode):
    """Load all files for the selected mode."""
    paths = get_file_paths(mode)

    df_students = read_csv_auto(paths["students"])
    df_cursus = read_csv_auto(paths["cursus"])
    df_kl = read_csv_auto(paths["kl_cursus"])
    df_success = read_csv_auto(paths["success"])

    return df_students, df_cursus, df_kl, df_success


def get_transition_columns(df):
    """Keep only transition columns."""
    return [col for col in df.columns if col not in META_COLUMNS]


def kl_to_cursus_name(name):
    """Rename short cursus names used in KL files."""
    mapping = {
        "MP": "Mesures Physiques",
        "RT": "Réseaux & Télécommunications",
        "CPGE/ATS": "CPGE/ATS",
        "GEII": "GEII",
        "Informatique": "Informatique",
    }
    return mapping.get(name, name)


@st.cache_data
def build_kl_table(df_kl):
    """Convert the KL file to a wide table with one column per cursus/theme."""
    wide_cols = [col for col in df_kl.columns if str(col).startswith("Dist_KL_")]

    if wide_cols:
        out = df_kl[["Student_ID"] + wide_cols].copy()
        out = out.rename(columns={col: kl_to_cursus_name(col.replace("Dist_KL_", "")) for col in wide_cols})
        return out

    if {"Student_ID", "Cursus", "Dist_KL"}.issubset(df_kl.columns):
        out = df_kl[["Student_ID", "Cursus", "Dist_KL"]].copy()
        out["Cursus"] = out["Cursus"].astype(str).map(kl_to_cursus_name)
        out = out.pivot(index="Student_ID", columns="Cursus", values="Dist_KL").reset_index()
        return out

    return pd.DataFrame()


@st.cache_data
def compute_closest_group(kl_wide):
    """Find the closest group from the KL distance."""
    if kl_wide.empty or "Student_ID" not in kl_wide.columns:
        return pd.DataFrame(columns=["Student_ID", "Closest_Group"])

    value_cols = [col for col in kl_wide.columns if col != "Student_ID"]
    if not value_cols:
        return pd.DataFrame(columns=["Student_ID", "Closest_Group"])

    out = kl_wide.copy()
    out["Closest_Group"] = out[value_cols].idxmin(axis=1)
    return out[["Student_ID", "Closest_Group"]]


def build_color_map(values):
    """Create a stable color for each cursus/theme."""
    palette = qualitative.Plotly + qualitative.D3 + qualitative.Set2 + qualitative.Set3
    values = sorted(pd.Series(list(values)).dropna().astype(str).unique())
    return {value: palette[i % len(palette)] for i, value in enumerate(values)}


def make_hover_students(df_students, group_name, transition):
    """Build the hover text with students from one group."""
    sub = df_students[df_students["Cursus"].astype(str) == str(group_name)][["Login_LDAP", transition]].copy()
    sub = sub.sort_values(by=transition, ascending=False)

    lines = []
    for _, row in sub.iterrows():
        lines.append(f"{row['Login_LDAP']}: {row[transition]:.3f}")

    return "<br>".join(lines)


def highlight_student_trace(x_value, y_value):
    """Marker used to highlight the selected student."""
    return go.Scatter(
        x=[x_value],
        y=[y_value],
        mode="markers",
        marker=dict(size=14, symbol="diamond", line=dict(width=2, color="black")),
        name="Selected student",
        showlegend=False,
        hovertemplate="Selected student<br>x=%{x:.3f}<br>Note_TP=%{y:.2f}<extra></extra>",
    )


def show_scatter_grid(figures):
    """Display figures on 2 columns."""
    col1, col2 = st.columns(2)

    for i, fig in enumerate(figures):
        if i % 2 == 0:
            col1.plotly_chart(fig, use_container_width=True)
        else:
            col2.plotly_chart(fig, use_container_width=True)

# ==========================================
# 3. GÉNÉRATION DES FIGURES
# ==========================================

def make_all_students_scatter(df_students, transitions, color_map):
    """Create one scatter plot per transition for all students."""
    figures = []

    for transition in transitions:
        fig = px.scatter(
            df_students,
            x=transition,
            y="Note_TP",
            color="Cursus",
            color_discrete_map=color_map,
            hover_data=["Login_LDAP", "Cohorte"],
        )
        fig.update_layout(
            title=f"{transition} - all students",
            xaxis_title=f"Proportion {transition}",
            yaxis_title="Note_TP",
        )
        figures.append(fig)

    return figures


def make_student_comparison_scatter(df_group, student, transitions, color_map, title_suffix):
    """Create one scatter plot per transition and highlight the selected student."""
    figures = []

    for transition in transitions:
        fig = px.scatter(
            df_group,
            x=transition,
            y="Note_TP",
            color="Cursus",
            color_discrete_map=color_map,
            hover_data=["Login_LDAP"],
        )
        fig.update_layout(
            title=f"{transition} - {title_suffix}",
            xaxis_title=f"Proportion {transition}",
            yaxis_title="Note_TP",
            showlegend=False,
        )
        fig.add_trace(highlight_student_trace(float(student[transition]), float(student["Note_TP"])))
        figures.append(fig)

    return figures


def make_student_vs_success_chart(student, df_success, transitions):
    """Compare the selected student with success/fail averages."""
    rows = []

    for transition in transitions:
        rows.append(
            {
                "Group": "Selected student",
                "Transition": transition,
                "Average": float(student[transition]),
            }
        )

    for _, row in df_success.iterrows():
        group_name = str(row["Groupe"])
        for transition in transitions:
            rows.append(
                {
                    "Group": group_name,
                    "Transition": transition,
                    "Average": float(row[transition]),
                }
            )

    chart_df = pd.DataFrame(rows)

    fig = px.bar(
        chart_df,
        x="Transition",
        y="Average",
        color="Group",
        barmode="group",
        category_orders={"Transition": transitions},
    )
    fig.update_layout(
        title="Selected student vs success / fail averages",
        xaxis_title="Transition",
        yaxis_title="Average proportion",
    )

    return fig

# ==========================================
# 4. SIDEBAR / NAVIGATION
# ==========================================

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Data view", ["Resource", "Theme"], index=0)
page = st.sidebar.radio("Page", ["Overview", "Student view"], index=0)

# ==========================================
# 5. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================

df_students, df_cursus, df_kl, df_success = load_mode_data(mode)
transitions = get_transition_columns(df_students)

df_students = df_students.copy()
df_students["Student_ID"] = pd.to_numeric(df_students["Login_LDAP"], errors="coerce")

kl_wide = build_kl_table(df_kl)
df_closest = compute_closest_group(kl_wide)

all_groups = set(df_students["Cursus"].astype(str).unique())
all_groups |= set(df_cursus["Cursus"].astype(str).unique())

if not kl_wide.empty:
    all_groups |= set([col for col in kl_wide.columns if col != "Student_ID"])

color_map = build_color_map(all_groups)
mode_title = mode.lower()

# ==========================================
# 6. PAGE OVERVIEW
# ==========================================

if page == "Overview":
    st.title(f"Overview - {mode_title}")

    st.subheader("Average proportions by group")
    df_cursus_long = df_cursus.melt(
        id_vars=["Cursus", "Effectif"],
        value_vars=transitions,
        var_name="Transition",
        value_name="Average",
    )

    df_cursus_long["Students"] = df_cursus_long.apply(
        lambda row: make_hover_students(df_students, row["Cursus"], row["Transition"]),
        axis=1,
    )

    group_order = sorted(df_cursus_long["Cursus"].astype(str).unique())
    pattern_map = {transition: PATTERNS[i % len(PATTERNS)] for i, transition in enumerate(transitions)}

    fig_cursus = go.Figure()

    for transition in transitions:
        sub = df_cursus_long[df_cursus_long["Transition"] == transition].set_index("Cursus").reindex(group_order)

        fig_cursus.add_trace(
            go.Bar(
                x=group_order,
                y=sub["Average"].tolist(),
                name=transition,
                marker=dict(
                    color=[color_map[group] for group in group_order],
                    pattern=dict(shape=pattern_map[transition]),
                ),
                customdata=list(zip([transition] * len(group_order), sub["Students"].fillna("").tolist())),
                hovertemplate=(
                    "Group=%{x}<br>"
                    "Transition=%{customdata[0]}<br>"
                    "Average=%{y:.3f}<br><br>"
                    "%{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )

    fig_cursus.update_layout(
        barmode="group",
        xaxis_title="Group",
        yaxis_title="Average proportion",
        legend_title_text="Transition",
    )
    st.plotly_chart(fig_cursus, use_container_width=True)

    st.subheader("Success vs fail per transition")
    df_success_long = df_success.melt(
        id_vars=["Groupe", "Effectif"],
        value_vars=transitions,
        var_name="Transition",
        value_name="Average",
    )

    fig_success = px.bar(
        df_success_long,
        x="Transition",
        y="Average",
        color="Groupe",
        barmode="group",
        category_orders={"Transition": transitions},
        hover_data=["Effectif"],
    )
    fig_success.update_layout(
        xaxis_title="Transition",
        yaxis_title="Average proportion",
    )
    st.plotly_chart(fig_success, use_container_width=True)

    st.subheader("All students - scatter plots per transition")
    overview_figures = make_all_students_scatter(df_students, transitions, color_map)
    show_scatter_grid(overview_figures)

# ==========================================
# 7. PAGE STUDENT VIEW
# ==========================================

else:
    st.title(f"Student view - {mode_title}")

    df_students["Login_LDAP_str"] = df_students["Login_LDAP"].astype(str)

    selected_login = st.selectbox(
        "Select a student (Login_LDAP)",
        sorted(df_students["Login_LDAP_str"].unique()),
    )

    student = df_students[df_students["Login_LDAP_str"] == selected_login].iloc[0]

    student_id = student["Student_ID"]
    student_group = str(student["Cursus"])
    student_grade = float(student["Note_TP"])

    closest_row = df_closest[df_closest["Student_ID"] == student_id]
    if not closest_row.empty:
        closest_group = str(closest_row["Closest_Group"].iloc[0])
    else:
        closest_group = student_group

    st.caption(
        f"Selected student: **{selected_login}** · "
        f"Own group: **{student_group}** · "
        f"Note_TP: **{student_grade:.2f}** · "
        f"Closest group (KL): **{closest_group}**"
    )

    st.subheader("Student vs own group")
    df_own_group = df_students[df_students["Cursus"].astype(str) == student_group].copy()
    own_figures = make_student_comparison_scatter(
        df_own_group,
        student,
        transitions,
        color_map,
        student_group,
    )
    show_scatter_grid(own_figures)

    st.subheader("Student vs closest group (KL)")
    df_closest_group = df_students[df_students["Cursus"].astype(str) == closest_group].copy()
    closest_figures = make_student_comparison_scatter(
        df_closest_group,
        student,
        transitions,
        color_map,
        closest_group,
    )
    show_scatter_grid(closest_figures)

    st.subheader("Student vs success / fail averages")
    fig_student_vs_success = make_student_vs_success_chart(student, df_success, transitions)
    st.plotly_chart(fig_student_vs_success, use_container_width=True)