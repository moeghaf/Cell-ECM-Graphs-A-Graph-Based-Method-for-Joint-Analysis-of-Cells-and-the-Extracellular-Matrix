'''
Helper functions to perform permutation test on cell-ECM graphs 

'''

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

def select_rows_with_ecm_and_columns_without_ecm(df):
    filtered_rows = df[df.index.str.contains('ecm', case=False, na=False)]
    filtered_columns = filtered_rows.loc[:, ~filtered_rows.columns.str.contains('ecm', case=False)]
    return filtered_columns
def select_rows_without_ecm_and_columns_with_ecm(df):
    """
    Selects all rows where the index does not contain 'ecm' 
    and all columns that contain 'ecm' in their name.
    """
    filtered_rows = df[~df.index.str.contains('ecm', case=False, na=False)]
    filtered_columns = filtered_rows.loc[:, filtered_rows.columns.str.contains('ecm', case=False)]
    return filtered_columns


def graph_to_df(cmg): 
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

def count_interactions_from_df(df):
    
    # Create df to store counts 
    unique_labels = np.sort(df['labels'].unique())
    interaction_counts = np.zeros((len(unique_labels), len(unique_labels)))
    interaction_counts_df = pd.DataFrame((interaction_counts), index= unique_labels, columns=unique_labels)

    for l, n in zip(df['labels'], df['neighbor_labels']):
        for j in n:
            if j in list(interaction_counts_df.columns):
                interaction_counts_df.loc[l,j] += 1 
    return interaction_counts_df

def permute_labels_in_df(df):
    permuted_df = df.copy()
    permuted_df['labels'] = permuted_df['labels'].sample(frac=1).reset_index(drop=True)
    return permuted_df 

def classic_interaction_count(cmg, original_matrix, cell_or_ecm): 
    '''
    The count is divided by the total number of cells of type A . 
    '''
    
    # Get cell counts from graph 
    if cell_or_ecm == 'cell': 
        unique_ct, uni_ct_counts = np.unique(cmg.cell_y_str,return_counts=True)
    if cell_or_ecm == 'ecm':
        labels = []
        for n,attri in cmg.G.nodes(data=True):
            if 'ecm' in n:
                labels.append('ECM ' + str(attri['ecm_labels']))
        unique_ct, uni_ct_counts = np.unique(labels,return_counts=True)
    if cell_or_ecm == 'both':
        unique_ct, uni_ct_counts = np.unique(cmg.cell_y_str,return_counts=True)
        
    ct_counts = pd.DataFrame((uni_ct_counts), index=unique_ct)
    
    classic_matrix = original_matrix.copy()
    # divided each row by cell type A count 
    for i in unique_ct:
        classic_matrix.loc[i, :]= classic_matrix.loc[i, :] / ct_counts.loc[i].values[0]
    
    return classic_matrix 

# Shuffle function
#def shuffle_group(group):
#    shuffled_group = np.random.permutation(group.values)
#    return shuffled_group

#def permute_cell_ecm_labels_in_df(df):
#    # Shuffle labels based on 'cell_or_ecm' groups
#    shuffled_labels = df.groupby('cell_or_ecm')['labels'].apply(shuffle_group).reset_index(drop=True)
#    df['labels'] = shuffled_labels
#    return df

def shuffle_group(group):
    shuffled_group = group.copy()
    np.random.shuffle(shuffled_group.values)
    return shuffled_group

def permute_cell_ecm_labels_in_df(df):
    # Shuffle labels based on 'cell_or_ecm' groups
    shuffled_labels = df.groupby('cell_or_ecm')['labels'].apply(shuffle_group).reset_index(drop=True)
    # Add shuffled labels back to the DataFrame
    df['labels'] = shuffled_labels
    return df 

def permutation_test(cmg, cell_or_ecm, iter):
    df = graph_to_df(cmg)
    if cell_or_ecm != 'both': 
                    cell_df = df[df['cell_or_ecm'] == cell_or_ecm].reset_index(drop=True)
                    count_interactions = count_interactions_from_df(cell_df)

    else:
        cell_df = df 
        count_interactions = count_interactions_from_df(cell_df)
        count_interactions= select_rows_without_ecm_and_columns_with_ecm(count_interactions)

    if cell_or_ecm == 'ecm': 
        rename_map ={f'ecm_{i}': f'ECM {i}' for i in range(cmg.ecm_KNN)}
        old_names = list(count_interactions.columns)
        new_names = [rename_map[i] for i in old_names]

        count_interactions.columns = new_names
        count_interactions.index = new_names

    ct = classic_interaction_count(cmg, count_interactions, cell_or_ecm=cell_or_ecm)
    pertubations = []
    print(iter)

    for _ in range(iter): 
        if cell_or_ecm != 'both': 
            permuted_cell_df = permute_labels_in_df(cell_df)
            count_interactions = count_interactions_from_df(permuted_cell_df)
        else: 
            permuted_cell_df = permute_cell_ecm_labels_in_df(cell_df)
            count_interactions = count_interactions_from_df(permuted_cell_df)
            count_interactions = select_rows_without_ecm_and_columns_with_ecm(count_interactions)

        if cell_or_ecm == 'ecm': 
            rename_map ={f'ecm_{i}': f'ECM {i}' for i in range(cmg.ecm_KNN)}


            old_names = list(count_interactions.columns)
            new_names = [rename_map[i] for i in old_names]

            count_interactions.columns = new_names
            count_interactions.index = new_names
        permuted_ct = classic_interaction_count( cmg, count_interactions, cell_or_ecm)
        pertubations.append(np.array(permuted_ct))

    pertubations = np.array(pertubations)

    sigval_matrix = np.zeros_like(ct)
    print(pertubations.shape)
    n, r,c = pertubations.shape
    p_thresh = 0.05

    for i in range(r):
        for j in range(c): 

            observed = ct.iloc[i, j]
            permuted = pertubations[:, i, j]

            p_gt = np.mean(permuted >= observed)
            p_lt = np.mean(permuted <= observed)

            interaction = p_lt > p_gt
            p = min(p_gt, p_lt)
            print('p over o', permuted >=observed)
            print('permuted:', permuted)
            print('observed:', observed)

            print('interaction:', interaction)

            print('p_lt', p_lt)    
            print('p_gt', p_gt)    
            print('p_val: ', p)

            sig = p < p_thresh
            print('sig', sig)

                        
            if (interaction == False) & (sig == True): 
                sigval = -1 
                print('sigval:', sigval)
            elif (interaction == True) & (sig == True): 
                sigval = 1
                print('sigval:', sigval)
            else:
                sigval= 0
                print('sigval:', sigval)


            
            sigval_matrix[i,j] = sigval

    if cell_or_ecm != 'both':
        names = ct.columns
        sigval_matrix = pd.DataFrame((sigval_matrix), columns=names, index=names)
    else: 
        col_names = permuted_ct.columns
        row_names = permuted_ct.index
        rename_map ={f'ecm_{i}': f'ECM {i}' for i in range(cmg.ecm_KNN)}
        old_names = list(col_names)
        new_names = [rename_map[i] for i in old_names]
        sigval_matrix = pd.DataFrame((sigval_matrix), columns=new_names, index=row_names)

    plt.figure(figsize=(10, 8),dpi=300)
    sns.heatmap(sigval_matrix,cbar_kws={'ticks': [-1, 0, 1]}, cmap='bwr')
    if cell_or_ecm == 'cell': 
        plt.title('Significant Cell-Cell Interactions ')
    if cell_or_ecm == 'ecm': 
        plt.title('Significant ECM-ECM Interactions ')
    if cell_or_ecm == 'both': 
        plt.title('Significant Cell-ECM Interactions ')

    plt.show()

