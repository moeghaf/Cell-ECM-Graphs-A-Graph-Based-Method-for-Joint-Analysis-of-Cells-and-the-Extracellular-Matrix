import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from glob import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, GCNConv, global_add_pool
from torch_geometric.data import Data

from captum.attr import Saliency, IntegratedGradients


import random
from collections import defaultdict


import networkx as nx
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import seaborn as sns


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.5):
        super(GNN, self).__init__()
        self.convs = ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Fully connected layers
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)


    def forward(self, x, edge_index, batch, edge_weight=None, return_pooled=False):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight).relu()

        x = global_add_pool(x, batch)

        if return_pooled:
            return x  # For UMAP visualization

        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
    
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F

# Custom transformation to drop random edges
class RandomEdgeDrop(T.BaseTransform):
    def __init__(self, p=0.2):
        self.p = p  # Probability of dropping an edge

    def __call__(self, data):
        # Randomly drop edges based on probability p
        edge_mask = torch.rand(data.edge_index.shape[1]) > self.p
        data.edge_index = data.edge_index[:, edge_mask]
        return data

# Custom transformation to drop random nodes
class RandomNodeDrop(T.BaseTransform):
    def __init__(self, p=0.2):
        self.p = p  # Probability of dropping a node

    def __call__(self, data):
        # Randomly drop nodes based on probability p
        num_nodes = data.num_nodes
        drop_mask = torch.rand(num_nodes) > self.p
        data.x = data.x[drop_mask]  # Apply mask to node features
        # Modify the edge index to only include edges that are within the remaining nodes
        edge_mask = (data.edge_index[0] >= drop_mask.sum()).logical_and(drop_mask[data.edge_index[0]]).logical_and(drop_mask[data.edge_index[1]])
        data.edge_index = data.edge_index[:, edge_mask]
        return 
    

from torch_geometric.datasets import CoraFull
import grafog.transforms as T
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import torch
import numpy as np
from captum.attr import IntegratedGradients, Saliency
from collections import defaultdict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def train(model, optimizer, train_loader, device, class_weights=None, edge_dropout_prob=0.2):
    """
    Train the GNN model with optional random edge dropout for data augmentation.

    Parameters:
    - model (torch.nn.Module): The GNN model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - train_loader (DataLoader): DataLoader for training data.
    - device (str): Device to run the model on ('cpu' or 'cuda').
    - class_weights (torch.Tensor, optional): Class weights for weighted loss.
    - edge_dropout_prob (float, optional): Probability of dropping edges for data augmentation.

    Returns:
    - float: Average training loss.
    """
    model.train()
    total_loss = 0

    if class_weights is not None:
        class_weights = class_weights.to(device)  # Ensure weights are on the same device

    for data in train_loader:
        data = data.to(device)

        # Apply random edge dropout
        edge_mask = torch.rand(data.edge_index.shape[1]) > edge_dropout_prob
        data.edge_index = data.edge_index[:, edge_mask]

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y[0], weight=class_weights)  # Weighted loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y[0], reduction='sum')  # Sum loss for accuracy calculation
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)  # Return both loss and accuracy

# Plot the training and validation losses
def plot_losses(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 6))
    
    plt.subplot()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("train_val_loss.tiff", bbox_inches="tight", dpi=600)
    
    return 


# Function to optimize the model
def objective(trial):
    print('Hyperparam optimization ... ')
    # Load data
    G_TYPE = 'cell_ecm'

    #### LOAD GRAPH DATA #########

    data_path = glob('C:/Users/Adminn/Documents/GitHub/Cell_ECM_Graphs/real_graph_data/'+G_TYPE+'/*')
    data_list = [torch.load(i, weights_only=False) for i in data_path]
    labels = np.array([i.y[0] for i in data_list])

    print('Using class weights')
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels.reshape(-1))
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    #### SPLIT INTO TRAIN,TEST VAL #########

    # Step 1: Split into train (60%) and temp (40%)
    train_data, temp_data = train_test_split(data_list, test_size=0.3, random_state=5, stratify=labels)
    temp_labels = [i.y[0] for i in temp_data]

    # Step 2: Split temp into val (50%) and test (50%) → 20% each of total
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=5, stratify=temp_labels)


    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)  # Small batch for tiny data
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)
    # Define hyperparameters to optimize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = data_list[0].x.shape[1]
    num_classes = 2
    num_layers = trial.suggest_int('num_layers', 2, 10)  # Number of layers
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)  # Hidden dimension size
    lr = trial.suggest_loguniform('lr', 1e-10, 1e-2,)  # Learning rate in log space

    # Define model and optimizer with sampled hyperparameters
    model = GNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    best_val_acc = 0
    patience = 10
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, 100):
        train_loss = train(model, optimizer, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)  # Evaluate validation loss and accuracy
        
        # Store losses for plotting later
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break

    # Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    final_val_loss, final_val_acc = evaluate(model, val_loader, device)

    return final_val_loss  # Return validation loss as objective metric

def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict

def pyg_to_networkx(data: Data) -> nx.Graph:
    """
    Converts a PyTorch Geometric Data object to a NetworkX graph with node features, positions, and edge attributes.
    
    Parameters:
        data (Data): A PyTorch Geometric Data object with attributes `x`, `edge_index`, `edge_attr`, and `pos`.
        
    Returns:
        G (nx.Graph): A NetworkX graph with node attributes and positions.
    """
    G = nx.Graph()

    # Add nodes with features and positions
    for i in range(data.x.size(0)):
        node_attr = {'feat': data.x[i].tolist()}
        if hasattr(data, 'pos'):
            node_attr['pos'] = data.pos[i].tolist()
        
        node_attr['node_labels'] = data.node_labels[i].tolist()

        G.add_node(i, **node_attr)

    # Add edges with attributes
    edge_index = data.edge_index
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    num_edges = edge_index.size(1)

    for i in range(num_edges):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        if edge_attr is not None:
            G.add_edge(src, dst, attr=edge_attr[i].tolist())
        else:
            G.add_edge(src, dst)

    return G


def model_forward(edge_mask, data, model, device):
    """
    Forward pass for the model with edge masking.

    Parameters:
    - edge_mask (torch.Tensor): Mask for edges.
    - data (torch_geometric.data.Data): Graph data.
    - model (torch.nn.Module): GNN model.
    - device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
    - torch.Tensor: Model output.
    """
    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
    out = model(data.x, data.edge_index, batch, edge_mask)
    return out

def explain(method, data, model, device, target=1):
    """
    Explain the model's predictions using the specified method.

    Parameters:
    - method (str): Explanation method ('ig' for Integrated Gradients, 'saliency' for Saliency).
    - data (torch_geometric.data.Data): Graph data.
    - model (torch.nn.Module): GNN model.
    - device (str): Device to run the model on ('cpu' or 'cuda').
    - target (int): Target class for explanation.

    Returns:
    - np.ndarray: Normalized edge mask.
    """
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    
    # Define a forward function that accepts all inputs as positional arguments
    def forward_func(edge_mask, data):
        return model_forward(edge_mask, data, model, device)
    
    if method == 'ig':
        ig = IntegratedGradients(forward_func)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data,),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(forward_func)
        mask = saliency.attribute(input_mask, target=target,
                                   additional_forward_args=(data,))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # Avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask

def aggregate_edge_directions(edge_mask, data):
    """
    Aggregate edge directions for undirected graphs.

    Parameters:
    - edge_mask (np.ndarray): Edge mask values.
    - data (torch_geometric.data.Data): Graph data.

    Returns:
    - dict: Aggregated edge mask values.
    """
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


def visualize_graph(G: nx.Graph, node_size=20, figsize=(12, 12), edge_cmap=plt.cm.inferno, use_edge_importance=True, savename=None):
    """    
    Parameters:
        G (nx.Graph): Graph to visualize.
        node_size (int): Size of the nodes.
        figsize (tuple): Figure size.
        edge_cmap (matplotlib colormap): Colormap for edge importance.
        use_edge_importance (bool): Whether to use edge importance for coloring.
    """
    # Extract positions
    pos = {i: G.nodes[i]['pos'] for i in G.nodes if 'pos' in G.nodes[i]}

    # Extract node labels and normalize them to a color scale
    labels = [G.nodes[i]['node_labels'] for i in G.nodes if 'node_labels' in G.nodes[i]]
    unique_labels = sorted(set(labels))
    label_to_color = {label: sns.color_palette("husl", 3)[i] for i, label in enumerate(unique_labels)}
    node_colors = [label_to_color[label] for label in labels]

    # Handle edge importance
    if use_edge_importance:
        edge_importance_values = [G[u][v].get('importance', 0.0) for u, v in G.edges]
        norm = mcolors.Normalize(vmin=min(edge_importance_values), vmax=max(edge_importance_values))
        edge_colors = [edge_cmap(norm(i)) for i in edge_importance_values]
        edge_widths = [1 + 3 * norm(i) for i in edge_importance_values]
        al = 0.3
    else:
        edge_colors = "gray"
        edge_widths = 1
        al = 1 

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=edge_widths, alpha=1
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_size, alpha=al, edgecolors="black"
    )

    # Add colorbar for edge importance if enabled
    if use_edge_importance:
        sm = cm.ScalarMappable(cmap=edge_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Edge Importance", fontsize=14)

    # Add horizontal legend at top
    legend_elements = [
        mpatches.Patch(color=color, label='ECM ' + str(label)) 
        for label, color in label_to_color.items()
    ]
    ax.legend(
        handles=legend_elements,
        title="Node Labels",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=14,
        title_fontsize=16
    )

    plt.axis("off")
    plt.tight_layout()
    if savename: 
        plt.savefig(savename + '.tiff', format='tiff', bbox_inches="tight", dpi=600)
    plt.show()



def plot_top_important_edges_heatmap(G: nx.Graph, top_n=30, figsize=(10, 8), cmap="Reds", savename=None):
    """
    Visualize the top N most important edges as a heatmap-style plot.

    Parameters:
        G (nx.Graph): Graph containing edge 'importance' and node 'node_labels'.
        top_n (int): Number of top edges to display.
        figsize (tuple): Size of the figure.
        cmap (str): Colormap to use.
    """
    # Extract edges with importance
    edge_data = [
        (u, v, G.nodes[u].get('node_labels', 'N/A'), G.nodes[v].get('node_labels', 'N/A'), G[u][v].get('importance', 0))
        for u, v in G.edges
    ]
    edge_df = pd.DataFrame(edge_data, columns=["Source", "Target", "Source_Label", "Target_Label", "Importance"])

    # Helper function to add 'ECM ' if label is numeric
    def format_label(label):
        try:
            float(label)
            return f"ECM {label}"
        except ValueError:
            return str(label)

    # Sort and take top N
    top_edges = edge_df.sort_values(by="Importance", ascending=False).head(top_n).reset_index(drop=True)

    # Format labels for the heatmap row labels
    top_edges["Edge_Label"] = top_edges.apply(
        lambda row: f"{format_label(row['Source_Label'])} → {format_label(row['Target_Label'])}", axis=1
    )

    # Prepare heatmap data: single column heatmap
    heatmap_data = pd.DataFrame({
        "Importance": top_edges["Importance"].values
    }, index=top_edges["Edge_Label"])

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, cbar_kws={'label': 'Edge Importance'})
    plt.title(f"Top {top_n} Most Important Edges", fontsize=14)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Edges", fontsize=12)
    plt.tight_layout()
    if savename: 
        plt.savefig(savename + '.tiff', format='tiff', bbox_inches="tight", dpi=600)
    plt.show()
    plt.show()
