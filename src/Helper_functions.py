import matplotlib.pyplot as plt
import numpy as np
import dill

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd


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