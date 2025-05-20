
########################################################################

# Code to perform permutation test
# Python version of imcRtools TestInteractions
# Reference: https://github.com/BodenmillerGroup/imcRtools/blob/devel/man/testInteractions.Rd 

########################################################################



import sys
import os
import matplotlib.pyplot as plt 
from Graph_builder import *
from SimData_Generator import *
from glob import glob
from CellECMGraphs_multiple import *
from tqdm import tqdm 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ast
import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

def graph_to_df(cmg): 
    '''
    Converts G to DF
    '''
    node_id = []
    unique_id = []
    cell_or_ecm = []
    labels = []
    neighbors = []
    unique_id_counter = 0 


    for n,attri in cmg.G.nodes(data=True):
        node_id.append(n)
        unique_id.append(unique_id_counter)
        unique_id_counter+=1 
        neighbors.append(list(cmg.G.neighbors(n)))

        if 'cell' in n:
            labels.append(attri['cell_type'])
            cell_or_ecm.append('cell')
        if 'ecm' in n:
            cell_or_ecm.append('ecm')
            labels.append('ecm_' + str(attri['ecm_labels']))



    node_id_to_labels_dict = {}

    for n,l in zip(node_id, labels):
        node_id_to_labels_dict[n] = l
        

    neighbors_labels = [ ]
    for n_list in neighbors:
        temp_n_l = []
        for j in n_list: 
            temp_n_l.append(node_id_to_labels_dict[j])
        neighbors_labels.append(temp_n_l)

    df = pd.DataFrame((node_id, cell_or_ecm, labels, neighbors_labels), index=['node_id', 'cell_or_ecm', 'labels', 'neighbor_labels']).T

    return df

def test_interactions_from_neighbor_labels(
    df,
    group_by="all",
    label_col="labels",
    neighbor_col="neighbor_labels",
    iter=1000,
    p_threshold=0.05,
    return_samples=False,
    tolerance=1e-8,
    n_jobs=1,
):
    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    
    # If group_by is "all", set a single group label for all rows
    if group_by == "all":
        df["__group__"] = "all"
        group_by = "__group__"
    else:
        df[group_by] = df[group_by].astype(str)

    observed = count_interactions_from_neighbors(df, group_by, label_col, neighbor_col)

    all_permuted = Parallel(n_jobs=n_jobs)(
        delayed(_permute_and_count_from_neighbors)(df, group_by, label_col, neighbor_col)
        for _ in tqdm(range(iter), desc="Permutations")
    )

    result = _calc_p_vals(observed, all_permuted, iter, p_threshold, return_samples, tolerance)
    return result


def count_interactions_from_neighbors(df, group_by, label_col, neighbor_col):
    interaction_counts = []
    all_labels = df[label_col].unique()

    for group in df[group_by].unique():
        df_group = df[df[group_by] == group]
        all_pairs = [(a, b) for a in all_labels for b in all_labels]

        counts = defaultdict(int)
        for idx, row in df_group.iterrows():
            source_label = row[label_col]
            for neighbor_label in row[neighbor_col]:
                counts[(source_label, neighbor_label)] += 1

        for a, b in all_pairs:
            from_count = (df_group[label_col] == a).sum()
            count = counts.get((a, b), 0)
            norm_ct = count / from_count if from_count > 0 else np.nan
            interaction_counts.append({
                'group_by': group,
                'from_label': a,
                'to_label': b,
                'ct': norm_ct
            })

    return pd.DataFrame(interaction_counts)


def _permute_and_count_from_neighbors(df, group_by, label_col, neighbor_col):
    df = df.copy()
    df[label_col] = df.groupby(group_by)[label_col].transform(lambda x: np.random.permutation(x.values))
    return count_interactions_from_neighbors(df, group_by, label_col, neighbor_col)


def _calc_p_vals(observed, permutations, n_perm, p_thres, return_samples, tolerance):
    permutations_df = pd.concat(permutations)
    merged = observed.copy()
    merged['p_gt'] = 0
    merged['p_lt'] = 0

    for i, row in observed.iterrows():
        group = row['group_by']
        a = row['from_label']
        b = row['to_label']
        ct = row['ct']
        sub = permutations_df[
            (permutations_df['group_by'] == group) &
            (permutations_df['from_label'] == a) &
            (permutations_df['to_label'] == b)
        ]
        perm_vals = sub['ct'].values
        p_gt = np.mean(perm_vals >= ct - tolerance)
        p_lt = np.mean(perm_vals <= ct + tolerance)
        merged.loc[i, 'p_gt'] = p_gt
        merged.loc[i, 'p_lt'] = p_lt
        merged.loc[i, 'p'] = min(p_gt, p_lt)
        interaction = p_lt > p_gt
        merged.loc[i, 'interaction'] = interaction
        sig = merged.loc[i, 'p'] < p_thres
        merged.loc[i, 'sig'] = sig
        merged.loc[i, 'sigval'] = (
            1 if interaction and sig else -1 if not interaction and sig else 0
        )

    return (merged, permutations_df) if return_samples else merged

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_interaction_heatmap(result_df, value_col='ct', sig_col='sigval', cmap='bwr', figsize=(10, 8), annot=False):
    # Pivot the result to a matrix form for heatmap
    heatmap_data = result_df.pivot_table(
        index='from_label',
        columns='to_label',
        values=value_col,
        aggfunc='mean'
    )

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"label": value_col},
        mask=heatmap_data.isna()
    )

    plt.title("Interaction Strength Heatmap")
    plt.xlabel("To Label")
    plt.ylabel("From Label")
    plt.tight_layout()
    plt.show()
