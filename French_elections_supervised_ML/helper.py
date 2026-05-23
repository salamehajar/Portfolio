# This files contains functions used across multiple notebooks

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import defaultdict
import math  
import ast
from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random


def plot_missing_values(datasets):
    """Plot percent of missing values per column for multiple datasets."""
    # plt.figure(figsize=(14, 10))
    # number of figures depends on number of datasets
    n_datasets = len(datasets)
    plt.figure(figsize=(14, 5 * ((n_datasets + 1) // 2)))
    for i, (name, df) in enumerate(datasets.items(), 1):
        plt.subplot(2, 2, i)
        # percent of missing values by column (0-100)
        na_pct = df.isna().mean() * 100
        na_pct = na_pct[na_pct > 0].sort_values(ascending=False)
        if na_pct.empty:
            plt.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=12)
            plt.title(f'{name}: 0 missing columns')
            plt.xlabel('')
            plt.yticks([])
        else:
            sns.barplot(x=na_pct.values, y=na_pct.index, palette='viridis')
            plt.xlabel('Percent missing (%)')
            plt.title(f'{name}: Missing values by column')
            # annotate bars with percent values
            for j, v in enumerate(na_pct.values):
                plt.text(v + 0.5, j, f'{v:.1f}%', va='center')
    plt.suptitle('Percent Missing Values per Column', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def replace_titles_with_ids(df, column, genres_id):
    """
    This is a function that takes a colomn where features are titles 
    and replace them with their corresponding ids from the genres_id
    """
    title_to_id = dict(zip(genres_id['genre_title'], genres_id['genre_id']))
    df[column] = df[column].map(title_to_id)
    return df

def replace_ids_with_titles(df, column, genres_id):
    id_to_title = dict(zip(genres_id['genre_id'], genres_id['genre_title']))
    df = df.copy()

    def map_ids(x):
        # treat iterables (lists, tuples, sets, etc.) as collections of ids
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return [id_to_title.get(i) for i in x]
        # treat scalars (single id) as one id
        return id_to_title.get(x, x)

    df[column] = df[column].apply(map_ids)
    return df

def build_genre_hierarchy(genres):
    """
    Builds a genre hierarchy from the genres DataFrame.

    Expects columns:
      - 'genre_id'
      - 'genre_parent_id'
      - 'genre_title'

    Returns:
      - parent_to_children: dict[parent_id] -> list[child_id]
      - mapped_parent_to_children: dict[parent_title] -> list[child_title]
      - id_to_title: dict[genre_id] -> genre_title
      - root_ids: list of root genre_ids (no parent)
      - root_titles: list of root genre_titles
    """
    # parent_id -> list of child_ids
    parent_to_children = defaultdict(list)
    id_to_title = dict(zip(genres['genre_id'], genres['genre_title']))

    # fill parent_to_children (by ids)
    for _, row in genres.iterrows():
        gid = row['genre_id']
        pid = row['genre_parent_id']
        # treat None / NaN as "no parent" (root)
        if pid is None or (isinstance(pid, float) and math.isnan(pid)):
            continue
        parent_to_children[pid].append(gid)

    # roots = ids that never appear as a child
    all_ids = set(genres['genre_id'])
    all_child_ids = set(genres['genre_parent_id'].dropna())
    root_ids = list(all_ids - all_child_ids)
    root_titles = [id_to_title[rid] for rid in root_ids]

    # same mapping, but with titles instead of ids
    mapped_parent_to_children = {}
    for pid, children in parent_to_children.items():
        parent_title = id_to_title.get(pid, f"Unknown({pid})")
        child_titles = [id_to_title.get(cid, f"Unknown({cid})") for cid in children]
        mapped_parent_to_children[parent_title] = child_titles

    return parent_to_children, mapped_parent_to_children, id_to_title, root_ids, root_titles

def fill_top_genre(df, parent_genre_map):
    
    df = df.copy()

    # Identify root genres (parent = itself)
    # root_genres = [gid for gid, root in parent_genre_map.items() if gid == root]

    for idx, row in df.iterrows():

        # Ad a treatment for rows where the genre_top is not na
        # we make sure they refer to the root parent and not the child
        if not pd.isna(row['genre_top']):
            gid = row['genre_top']
            root_parent = parent_genre_map.get(gid, gid)
            df.at[idx, 'genre_top'] = root_parent
            continue

        # Only fill missing values
        if pd.isna(row['genre_top']):
            # STEP 1: Dominant parent in genres
            genre_counts = {}
            if isinstance(row['genres'], list):
                for gid in row['genres']:
                    parent = parent_genre_map.get(gid)
                    if parent is not None:
                        genre_counts[parent] = genre_counts.get(parent, 0) + 1

            if genre_counts:
                df.at[idx, 'genre_top'] = max(genre_counts, key=genre_counts.get)
                continue

            # STEP 2: Dominant parent in genres_all
            genre_all_counts = {}
            if isinstance(row['genres_all'], list):
                for gid in row['genres_all']:
                    parent = parent_genre_map.get(gid)
                    if parent is not None:
                        genre_all_counts[parent] = genre_all_counts.get(parent, 0) + 1

            if genre_all_counts:
                df.at[idx, 'genre_top'] = max(genre_all_counts, key=genre_all_counts.get)
                continue

            # STEP 3: Use first available genre (genres, then genres_all)
            genres_list = row['genres'] if isinstance(row['genres'], list) else []
            genres_all_list = row['genres_all'] if isinstance(row['genres_all'], list) else []

            # If both empty: drop row
            if len(genres_list) == 0 and len(genres_all_list) == 0:
                df = df.drop(index=idx)
                continue

            # Else pick first valid gid
            if len(genres_list) > 0:
                first_gid = genres_list[0]
            else:
                first_gid = genres_all_list[0]

            root_parent = parent_genre_map.get(first_gid, first_gid)
            df.at[idx, 'genre_top'] = root_parent

    return df

def pca_group(df, cols, name):
    Xg = df[cols].dropna()
    scaler = StandardScaler()
    Xg_scaled = scaler.fit_transform(Xg)

    pca = PCA()           # toutes les composantes
    pca.fit(Xg_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nGroupe {name} ({len(cols)} variables)")
    print("Variance expliquée par composante :", pca.explained_variance_ratio_)
    print("Variance cumulée :", cum_var)

def make_pcs(df, cols, n_comp, prefix):
    Xg = df[cols].dropna()
    scaler = StandardScaler()
    Xg_scaled = scaler.fit_transform(Xg)
    pca = PCA(n_components=n_comp)
    Z = pca.fit_transform(Xg_scaled)
    for j in range(n_comp):
        df.loc[Xg.index, f'{prefix}_pc{j+1}'] = Z[:, j]
    return df

def evaluate_model(model, X_test, Y_test):
    
    Y_pred = model.predict(X_test)

    y_true_seconds = np.expm1(Y_test)
    y_pred_seconds = np.expm1(Y_pred)

    r2 = model.score(X_test, Y_test)
    rmse_log = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae_log = mean_absolute_error(Y_test, Y_pred)
    mae_seconds_real = mean_absolute_error(y_true_seconds, y_pred_seconds)

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE (Log-Seconds): {rmse_log:.4f}")
    print(f"MAE (Log-Seconds): {mae_log:.4f}")
    print(f"MAE (Seconds): {mae_seconds_real:.2f} seconds")
    
def target_encode(train_df, test_df, target_series, col_name, m=10):
    """
    This function applies Target Encoding with Smoothing.
    We smooth the encoding to avoid overfitting on categories with few samples.
    """
    # Calculate Global Mean (of the target)
    global_mean = target_series.mean()

    # Create a temporary dataframe for calculation
    train = train_df.copy()
    train['target'] = target_series

    # Calculate Count and Mean for each category
    count = train.groupby(col_name)['target'].agg(['count', 'mean'])

    # Calculate Smoothed Score
    # Formula: (n * mean + m * global_mean) / (n + m)
    count['smoothed'] = (count['count'] * count['mean'] + m * global_mean) / (count['count'] + m)

    # Create mapping dictionary
    dict = count['smoothed'].to_dict()

    # Map to Train and Test
    train_df[col_name + '_encoded'] = train_df[col_name].map(dict)
    test_df[col_name + '_encoded'] = test_df[col_name].map(dict)

    # Fill unknowns in Test with Global Mean
    test_df[col_name + '_encoded'] = test_df[col_name + '_encoded'].fillna(global_mean)

    # Drop the original string column
    train_df = train_df.drop(columns=[col_name])
    test_df = test_df.drop(columns=[col_name])

    return train_df, test_df
