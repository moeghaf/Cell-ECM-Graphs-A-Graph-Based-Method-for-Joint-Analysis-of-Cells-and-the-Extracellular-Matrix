import matplotlib.pyplot as plt
import numpy as np
import dill

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

# Load the CellECMGraph object saved from Figure S1 notebook
def load_ceg(savename):
    """
    Load the CellECMGraph object from a file using dill.
    """
    with open(savename, "rb") as f:
        return dill.load(f)

def plot_36_images(images, titles=None, cmap='jet', figsize=(10, 10), save_path=None):
    
    assert len(images) == 36, "You must provide exactly 36 images."

    fig, axs = plt.subplots(6, 6, figsize=figsize, dpi=300)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i], cmap=cmap)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i], fontsize=6, pad=2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()

def embeddings_neighbourhood_feature_vector(ceg_dict, g_type):
    all_node_features = []
    all_node_labels = []
    cell_or_ecm = []
    temp_G = ceg_dict.cell_G if g_type == 'cellgraph' else ceg_dict.G

    for n, attri in temp_G.nodes(data=True):
        if 'cell' in n:
            all_node_labels.append(attri['cell_type'])
            cell_or_ecm.append('cell')
        else:
            all_node_labels.append(attri['ecm_labels'])
            cell_or_ecm.append('ecm')

    unique_labels = np.unique(all_node_labels)
    nodes = list(temp_G.nodes)
    for n in nodes:
        neighbours = list(temp_G.neighbors(n))
        frequency_count_per_node = pd.DataFrame(np.zeros((len(unique_labels))), index=unique_labels)
        for neigh in neighbours:
            if 'cell' in neigh:
                n_ct = temp_G.nodes[neigh]['cell_type']
            else:
                n_ct = str(temp_G.nodes[neigh]['ecm_labels'])
            frequency_count_per_node.loc[n_ct] += 1
        all_node_features.append(frequency_count_per_node.values.flatten())
    return np.array(all_node_features), np.array(all_node_labels), np.array(cell_or_ecm)

def node_classification_knn_cv(ceg, g_type, n_neighbors=5, n_splits=5):
    if g_type == 'ecmgraph':
        x, y, cell_or_ecm = embeddings_neighbourhood_feature_vector(ceg, 'cellecmgraph')
        x = x[cell_or_ecm == 'cell']
        y = y[cell_or_ecm == 'cell']
        x = x[:, :3]
    else:
        x, y, cell_or_ecm = embeddings_neighbourhood_feature_vector(ceg, g_type)
        x = x[cell_or_ecm == 'cell'].astype(int)
        y = y[cell_or_ecm == 'cell']

    # Filter classes with at least 2 samples
    unique_classes, class_counts = np.unique(y, return_counts=True)
    sufficient_classes = unique_classes[class_counts >= 2]
    x_filtered = x[np.isin(y, sufficient_classes)]
    y_filtered = y[np.isin(y, sufficient_classes)]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    reports = []
    f1_per_fold = []

    for train_index, test_index in skf.split(x_filtered, y_filtered):
        X_train, X_test = x_filtered[train_index], x_filtered[test_index]
        y_train, y_test = y_filtered[train_index], y_filtered[test_index]

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cityblock')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        reports.append(report_df)
        f1_per_fold.append(report_df.loc['weighted avg']['f1-score'])

    reports_df = pd.concat(reports)
    return reports_df, f1_per_fold

def get_macro_metrics(report_df, name):
    macro_metrics = pd.DataFrame(
        [[
            report_df.loc['accuracy']['precision'] if 'accuracy' in report_df.index else np.nan,
            report_df.loc['weighted avg']['precision'],
            report_df.loc['weighted avg']['recall'],
            report_df.loc['weighted avg']['f1-score']
        ]],
        columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        index=[name]
    )
    return macro_metrics

import matplotlib.pyplot as plt

def visualize_cell_ground_truths(self):
    # Extracting node attributes for nodes with 'cell' in ID
    cell_nodes = [node for node in self.G.nodes() if 'cell' in node]
    node_positions = {node: attributes['centroid'] for node, attributes in self.G.nodes(data=True) if node in cell_nodes}
    node_ground_truth = {node: attributes['ground_truth_label'] for node, attributes in self.G.nodes(data=True) if node in cell_nodes}
    unique_regions = sorted(np.unique(list(node_ground_truth.values())))
    # Set up a new colormap for regions
    region_colors = sns.color_palette("hls", n_colors=len(unique_regions))
    color_map = {region: region_colors[i % len(region_colors)] for i, region in enumerate(unique_regions)}

    # Generate node colors based on ground truth region
    node_colors = [color_map[node_ground_truth[node]] for node in node_positions]

    # Extract edges where 'cell-cell' interaction exists
    edges_to_plot = [(u, v) for u, v, attr in self.G.edges(data=True) if 'cell-cell' in attr['interaction']]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.axis('off')

    # Draw nodes and edges
    nx.draw_networkx_nodes(self.G, pos=node_positions, nodelist=cell_nodes, 
                        node_color=node_colors, node_size=self.node_size, ax=ax, 
                        edgecolors='black', linewidths=self.node_linewidth)
    nx.draw_networkx_edges(self.G, pos=node_positions, edgelist=edges_to_plot, 
                        width=1, alpha=0.6, ax=ax, edge_color='black')

    # Create legend handles for ground truth regions
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=region, 
                                markersize=18, markerfacecolor=color_map[region]) 
                    for region in unique_regions]

    # Add legend for cell-cell interaction
    legend_handles.append(plt.Line2D([0], [0], color='darkblue', lw=1, alpha=1, label='Cell-Cell Interaction'))

    plt.legend(
        handles=legend_handles, 
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        ncol=5,  
        fontsize=18,  
        handletextpad=0.5,
        frameon=False  
    )

    plt.show()

    # convert to percentage
def convert_to_percentage(df):
    return df.apply(lambda x: x / x.sum() * 100, axis=1)


def embeddings_neighbourhood_feature_vector_gt(ceg_dict, g_type):
    all_node_features = []
    all_node_labels = []
    cell_or_ecm = []

    if g_type == 'cellgraph':
        temp_G = ceg_dict.cell_G   
    else:
        # Default to the full graph if g_type is not 'cellgraph'    
        temp_G = ceg_dict.G

    for n, attri in temp_G.nodes(data=True):
        if 'cell' in n:
            all_node_labels.append(str(attri['cell_type']))
            cell_or_ecm.append('cell')
        else:
            if g_type == 'cellecmgraph':
                all_node_labels.append(str(attri['ecm_labels']))
                cell_or_ecm.append('ecm')

    unique_labels = np.unique(all_node_labels)

    nodes = list(temp_G.nodes)
    for n in nodes:
        neighbours = list(temp_G.neighbors(n))
        frequency_count_per_node = pd.DataFrame(np.zeros((len(unique_labels))), index=unique_labels)
        for neigh in neighbours:
            if 'cell' in neigh:
                n_ct = str(temp_G.nodes[neigh]['cell_type'])
            else:
                if g_type == 'cellecmgraph':
                    n_ct = str(temp_G.nodes[neigh]['ecm_labels'])
            frequency_count_per_node.loc[n_ct] += 1
        all_node_features.append(frequency_count_per_node.values.flatten())

    return np.array(all_node_features), np.array(all_node_labels), np.array(cell_or_ecm)

import matplotlib.pyplot as plt

def visualize_cell_labels(self, label):
    
    # Extracting node attributes for nodes with 'cell' in ID
    cell_nodes = [node for node in self.G.nodes() if 'cell' in node]
    node_positions = {node: attributes['centroid'] for node, attributes in self.G.nodes(data=True) if node in cell_nodes}
    node_ground_truth = {node: attributes[label] for node, attributes in self.G.nodes(data=True) if node in cell_nodes}
    unique_regions = sorted(np.unique(list(node_ground_truth.values())))
    # Set up a new colormap for regions
    region_colors = sns.color_palette("hls", n_colors=len(unique_regions))
    color_map = {region: region_colors[i % len(region_colors)] for i, region in enumerate(unique_regions)}

    # Generate node colors based on ground truth region
    node_colors = [color_map[node_ground_truth[node]] for node in node_positions]

    # Extract edges where 'cell-cell' interaction exists
    edges_to_plot = [(u, v) for u, v, attr in self.G.edges(data=True) if 'cell-cell' in attr['interaction']]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.axis('off')

    # Draw nodes and edges
    nx.draw_networkx_nodes(self.G, pos=node_positions, nodelist=cell_nodes, 
                        node_color=node_colors, node_size=self.node_size, ax=ax, 
                        edgecolors='black', linewidths=self.node_linewidth)
    nx.draw_networkx_edges(self.G, pos=node_positions, edgelist=edges_to_plot, 
                        width=1, alpha=0.6, ax=ax, edge_color='black')

    # Create legend handles for ground truth regions
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=region, 
                                markersize=18, markerfacecolor=color_map[region]) 
                    for region in unique_regions]

    # Add legend for cell-cell interaction
    legend_handles.append(plt.Line2D([0], [0], color='darkblue', lw=1, alpha=1, label='Cell-Cell Interaction'))

    plt.legend(
        handles=legend_handles, 
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        ncol=5,  
        fontsize=18,  
        handletextpad=0.5,
        frameon=False  
    )

    plt.show()

def visualize_cell_ecm_neighbourhood(G):
    pos = {}
    neighbourhood_labels = []
    cell_nodes = []
    ecm_nodes = []

    for node, attr in G.nodes(data=True):
        if 'ecm_coords' in attr:
            pos[node] = attr['ecm_coords']
            ecm_nodes.append(node)
            neighbourhood_labels.append(attr['cell_neighbourhood'])
        if 'centroid' in attr:
            pos[node] = attr['centroid']
            cell_nodes.append(node)
            neighbourhood_labels.append(attr['cell_neighbourhood'])

        # Collect cell_neighbourhood label for coloring

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.axis('off')

    cell_cell_edges = []
    ecm_ecm_edges = []
    cell_ecm_edges = []

    for u, v, attr in G.edges(data=True):
        if 'cell-cell' in attr['interaction']:
            cell_cell_edges.append((u, v))
        elif 'ecm-ecm' in attr['interaction']:
            ecm_ecm_edges.append((u, v))
        elif 'cell-ecm' in attr['interaction']:
            cell_ecm_edges.append((u, v))

    # Get unique neighbourhood labels and assign colors
    unique_neigh_labels = sorted(set(neighbourhood_labels))
    cmap = plt.get_cmap('tab10')
    neigh_color_map = {lbl: cmap(i % 10) for i, lbl in enumerate(unique_neigh_labels)}
    node_colors = [neigh_color_map[lbl] for lbl in neighbourhood_labels]

    # Plot edges
    nx.draw_networkx_edges(G, pos=pos, edgelist=cell_cell_edges, width=1,
                          alpha=0.3, ax=ax, edge_color='darkblue')
    nx.draw_networkx_edges(G, pos=pos, edgelist=ecm_ecm_edges, width=1,
                          alpha=0.3, ax=ax, edge_color='darkgreen')
    nx.draw_networkx_edges(G, pos=pos, edgelist=cell_ecm_edges, width=1,
                          alpha=0.6, ax=ax, edge_color='darkred')

    # Plot all nodes colored by neighbourhood label
    nx.draw_networkx_nodes(G, pos=pos, nodelist=list(G.nodes), node_color=node_colors,
                          node_size=10, ax=ax, edgecolors='black',
                          linewidths=1, alpha=0.8)

    # Create legend for neighbourhood clusters
    legend_handles = []
    for lbl in unique_neigh_labels:
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', label=f'Neighbourhood {lbl}',
                       markersize=18, markerfacecolor=neigh_color_map[lbl])
        )

    # Add edge legends
    legend_handles.append(plt.Line2D([0], [0], color='darkblue', lw=1, alpha=1, label='Cell-Cell Interaction'))
    legend_handles.append(plt.Line2D([0], [0], color='darkgreen', lw=1, alpha=1, label='ECM-ECM Interaction'))
    legend_handles.append(plt.Line2D([0], [0], color='darkred', lw=1, alpha=1, label='Cell-ECM Interaction'))

    plt.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=6,
        fontsize=12,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=1.2
    )

