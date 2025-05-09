from tifffile import imwrite, imread
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import os 
from sklearn.preprocessing import LabelEncoder
from glob import glob
import seaborn as sns 
import shutil
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.ticker as ticker
import seaborn as sns 
from matplotlib.patches import Patch
import networkx as nx 
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import skew, kurtosis
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import torch 
import random
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from sklearn.neighbors import KDTree

random.seed(42)  
np.random.seed(42)  

class GraphBuilder:
    
    def __init__(self, 
                 full_stack_img_path, 
                 panel_path, cell_data_path, 
                 save_folder,
                 patch_size=5,
                 norm='min-max',
                 Dmax_CE = 7,
                 Dmax_CC = 17,
                 ecm_KNN = 4,
                 single=False,
                 feature_type = 'mean+std',
                 node_linewidth = 0.5,
                 node_size = 10,
                 edge_width = 2,
                 interaction_k = 5,
                 edge_metric = 'knn',
                 save = False
                 ):
        
        # Hyperparams 
        self.norm = norm 
        self.Dmax_CE = Dmax_CE
        self.Dmax_CC = Dmax_CC
        self.ecm_KNN = ecm_KNN 
        self.interaction_k = interaction_k
        self.edge_metric=edge_metric

        # Paths 
        self.full_stack_img_path = full_stack_img_path
        self.panel_path = panel_path
        self.cell_data_path = cell_data_path
        self.patch_size = patch_size
        self.save_folder = save_folder
        self.single = single 
        self.feature_type = feature_type

        # Visualization params 
        self.node_linewidth = node_linewidth
        self.node_size = node_size 
        self.edge_width = edge_width
        self.save = save
            
    def load_imgs(self, normalize=True):
        # Load IMC intensity stacks, panel, cell labels, and cell segmentation masks. 
        self.full_stack_imgs = imread(self.full_stack_img_path)
        self.scaler_min_max = MinMaxScaler()

        # If normalization is required
        if normalize:
            if self.norm == 'znorm':
                self.scaler = StandardScaler()
            
            if self.norm == 'min-max':
                self.scaler = MinMaxScaler()

            self.raw = self.full_stack_imgs.copy()
            normalized_stack = np.zeros_like(self.raw, dtype=np.float32)
            min_normalized_stack = np.zeros_like(self.raw, dtype=np.float32)

            for i in range(self.raw.shape[0]):
                normalized_stack[i] = self.scaler.fit_transform(self.raw[i])
                min_normalized_stack[i] = self.scaler_min_max.fit_transform(self.raw[i])
            
            self.full_stack_imgs = normalized_stack
            self.full_stack_imgs_min_max = min_normalized_stack # Used for plotting protein proportions instead of z-norm
        else:
            # If no normalization, just use the raw images
            self.full_stack_imgs = self.raw = self.full_stack_imgs.copy()
            self.full_stack_imgs_min_max = self.raw.copy()

        # Shape of the image stack
        self.c, self.h, self.w = self.full_stack_imgs.shape

        # Load panel and cell data
        self.panel = pd.read_csv(self.panel_path)
        self.cell_data = pd.read_csv(self.cell_data_path) 
        self.cell_y_str = np.array(self.cell_data['celltype'])  # May need to rename it to celltype 

        # Create directory if not exists
        os.makedirs(self.save_folder, exist_ok=True)

    def find_patch_pixels_using_centre(self, image, patch_centre, patch_size=10):
        mask = np.zeros_like(image, dtype=bool)
        # Calculate the coordinates of the patch region
        top_left_row = patch_centre[0] - patch_size // 2
        top_left_col = patch_centre[1] - patch_size // 2
        bottom_right_row = top_left_row + patch_size
        bottom_right_col = top_left_col + patch_size

        # Set the patch region in the mask to True
        mask[top_left_row:bottom_right_row, top_left_col:bottom_right_col] = True

        # Apply the mask to the original image
        patch_only_img = np.where(mask, image, 0) 
        return patch_only_img

    def find_nonzero_coordinates(self, image):
        nonzero_rows, nonzero_cols = np.nonzero(image)
        return np.column_stack((nonzero_rows, nonzero_cols))

    def find_neighbour_patch_coords(self, coord, coord_list):
        ecm_ecm_distance = self.patch_size 
        within_distance_coords = [] 
        
        for c in coord_list:
            # Calculate the absolute difference between coordinates in both rows and columns
            row_diff = abs(coord[0] - c[0])
            col_diff = abs(coord[1] - c[1])

            # Check if both row and column differences are within the specified distance
            if row_diff <= ecm_ecm_distance and col_diff <= ecm_ecm_distance:
                within_distance_coords.append(c)
        return within_distance_coords
    
    def min_distance_between_patch_coords(self, set1, set2):
        distances = cdist(set1, set2)
        return np.min(distances), distances

    def number_of_ecm_edges(self):
        n_ecm_ecm_edges = 0
        for u,v in self.G.edges():
            if ('ecm' in u) & ('ecm' in v):
                n_ecm_ecm_edges +=1
        return n_ecm_ecm_edges

    def number_of_cell_ecm_edges(self):
        n_cell_ecm_edges = 0
        for u,v in self.G.edges():
            if (('cell' in u) & ('ecm' in v)) | (('ecm' in u) & ('cell' in v)):
                n_cell_ecm_edges +=1
        return n_cell_ecm_edges

    def number_of_cell_edges(self):
        return self.cell_G.number_of_edges()

    def build_cell_ecm_interactions(self):

        if self.edge_metric == 'knn':
            # Set up cellnode id dict 
            self.cell_nodeid_to_cell_centre = {}
            for n, attri in self.G.nodes(data=True):
                if 'cell' in n:
                    self.cell_nodeid_to_cell_centre[attri['centroid']] = n

            # KNN setup
            knn = NearestNeighbors(n_neighbors=self.interaction_k)  # We want to find the 5 nearest neighbors
            knn.fit(list(self.ecm_nodeid_to_patch_centre.keys()))  # Fit KNN on ECM patch coordinates

            # Loop through each cell and find the nearest ECM patches
            for  cell_centroid, cell_id in self.cell_nodeid_to_cell_centre.items():
                # Find 5 nearest ECM patches for this cell using KNN
                distances, indices = knn.kneighbors([cell_centroid])  # Query with the cell centroid

                # Loop through the nearest ECM patches
                for i in range(len(distances[0])):  # Loop through the 5 nearest ECM patches
                    ecm_node_id = list(self.ecm_nodeid_to_patch_centre.values())[indices[0][i]]  # Get ECM node ID
                    ecm_centroid = list(self.ecm_nodeid_to_patch_centre.keys())[indices[0][i]]  # Get ECM centroid

                    # Check if the distance is below the threshold Dmax_CE
                    if distances[0][i] <= self.Dmax_CE:
                        # Add the interaction (edge) between the cell and the ECM node if the distance is within threshold
                        
                        self.G.add_edge(cell_id, ecm_node_id, interaction='cell-ecm', feature=np.array([0.,0.,1.]))

    def create_patches(self, padded_ecm_stack=None):
        if padded_ecm_stack:
            self.padded_ecm_stack
        # Get the shape of the padded image
        rows, cols, channels = self.padded_ecm_stack.shape
        
        # Calculate the number of patches along rows and columns
        num_patches_rows = rows // self.patch_size
        num_patches_cols = cols // self.patch_size
        
        # Initialize arrays to store patches and centers
        self.patches = []
        self.centers = []
        
        # Iterate over patches
        for i in range(num_patches_rows):
            for j in range(num_patches_cols):
                # Calculate patch coordinates
                start_row = i * self.patch_size
                end_row = start_row + self.patch_size
                start_col = j * self.patch_size
                end_col = start_col + self.patch_size
                
                # Extract the patch from the image
                patch = self.padded_ecm_stack[start_row:end_row, start_col:end_col, :]
                
                # Calculate patch center
                center_row = (start_row + end_row) / 2
                center_col = (start_col + end_col) / 2
                
                # Append patch and center to respective arrays
                self.patches.append(patch)
                self.centers.append((center_row, center_col))
        
        # Convert lists to numpy arrays
        self.patches = np.array(self.patches)
        self.centers = np.array(self.centers)
        
        # Reshape patches to desired format
        self.ecm_patches_rs = self.patches.reshape(-1, self.patch_size, self.patch_size, self.c)    
        self.ecm_patches_rs = self.ecm_patches_rs[:,:,:,self.ecm_mask]
        
    def load_ecm_data(self): 
        self.ecm_stack = self.full_stack_imgs.copy()
        self.ecm_mask = self.panel['ecm'].eq(1).to_numpy()
        self.ecm_stack[~self.ecm_mask, :, :] = 0    
    
    def build_cell_ecm_graph(self):
        self.get_ecm_patches()
        self.load_patch_data()
        self.build_cell_cell_graph()
        self.set_up_colors()
        self.build_ecm_ecm_graph()
        self.build_cell_ecm_interactions()

    def pad_ecm_stack(self):
    
        # Calculate the amount of padding required for each dimension
        pad_width = [(0, 0)] 
        pad_h = int(np.ceil(self.h/self.patch_size ) * self.patch_size )
        pad_w = int(np.ceil(self.w/self.patch_size ) * self.patch_size )
        
        for target_dim, stack_dim in zip((pad_h, pad_w), self.ecm_stack.shape[1:]):
            pad_before = (target_dim - stack_dim) // 2
            pad_after = target_dim - stack_dim - pad_before
            pad_width.append((pad_before, pad_after))
        
        # Pad the ecm_stack array
        padded_ecm_stack = np.pad(self.ecm_stack, pad_width=pad_width, mode='constant')
        # Find padded height and width 
        self.padded_ecm_stack_fs = padded_ecm_stack
        self.padded_ecm_stack = padded_ecm_stack.transpose(1,2,0)
        self.padded_ecm_stack_shape = np.shape(self.padded_ecm_stack[:,:,self.ecm_mask])

    def cluster_ecm_patches(self):
        # Select the feature based on the user's choice
        if self.feature_type == 'mean':
            self.ecm_patch_flat = self.ecm_patches_rs.mean(axis=(1, 2))
        elif self.feature_type == 'median':
            self.ecm_patch_flat = np.median(self.ecm_patches_rs, axis=(1, 2))
        elif self.feature_type == 'std':
            self.ecm_patch_flat = self.ecm_patches_rs.std(axis=(1, 2))
        elif self.feature_type == 'variance':
            self.ecm_patch_flat = np.var(self.ecm_patches_rs, axis=(1, 2))
        elif self.feature_type == 'sum':
            self.ecm_patch_flat = self.ecm_patches_rs.sum(axis=(1, 2))
        elif self.feature_type == 'reshape':
            # Flatten each patch into a single vector
            self.ecm_patch_flat = self.ecm_patches_rs.reshape(self.ecm_patches_rs.shape[0], -1)
        elif self.feature_type == 'std+mean':
            # Combine std and mean (concatenate the mean and std for each patch)
            mean_vals = self.ecm_patches_rs.mean(axis=(1, 2))
            std_vals = self.ecm_patches_rs.std(axis=(1, 2))
            self.ecm_patch_flat = np.concatenate([mean_vals, std_vals], axis=1)  # Concatenate along the feature axis
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        

        self.embedding = self.ecm_patch_flat
        emb_dim = self.embedding.shape

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=self.ecm_KNN, n_init=50, algorithm='elkan', random_state=42)
        kmeans.fit(self.ecm_patch_flat)
        
        # Save the cluster labels
        self.cluster_labels = kmeans.labels_

    def reconstruct_cluster_patches_to_image(self):

        # Reshape patches to collapse the patch dimensions into one

        self.clustered_image = np.zeros_like(self.ecm_patches_rs)

        for i, label in enumerate(self.cluster_labels):
            self.clustered_image[i] = label    
            
        patch_size = self.ecm_patches_rs.shape[-3:]
        self.reconstructed_image = np.zeros(self.padded_ecm_stack_shape, dtype=self.ecm_patches_rs.dtype)
        
        # Counter for iterating through flattened_patches
        k = 0
        # Iterate over each block position in the original image
        for i in range(0, self.padded_ecm_stack_shape[0], patch_size[0]):
            for j in range(0, self.padded_ecm_stack_shape[1], patch_size[1]):
                for l in range(0, self.padded_ecm_stack_shape[2], patch_size[2]):
                    # Place the current patch into the appropriate position in the reconstructed image
                    self.reconstructed_image[i:i+patch_size[0], j:j+patch_size[1], l:l+patch_size[2]] = self.clustered_image[k]
                    k += 1
        

        min_max_img = self.full_stack_imgs_min_max[self.ecm_mask].transpose(1,2,0)
        ecm_patch_img = self.reconstructed_image[:,:,0]

        cluster_means = {}

        for i in np.unique(self.cluster_labels): 
            cluster_means[i] = min_max_img[ecm_patch_img == i].mean(0)

        cluster_df = pd.DataFrame(cluster_means, index=[self.panel[self.ecm_mask].name.values])
        self.cluster_df = cluster_df.T


        # Rename background and recalculate protein expression 
        self.background_label = self.cluster_df.sum(axis=1).argmin()
        self.reconstructed_image = self.rename_cluster_labels(ecm_patch_img, self.background_label)

        min_max_img = self.full_stack_imgs_min_max[self.ecm_mask].transpose(1,2,0)
        ecm_patch_img = self.reconstructed_image

        cluster_means = {}

        for i in np.unique(self.cluster_labels): 
            cluster_means[i] = min_max_img[ecm_patch_img == i].mean(0)

        cluster_df = pd.DataFrame(cluster_means, index=[self.panel[self.ecm_mask].name.values])
        self.cluster_df = cluster_df.T

        fig, ax = plt.subplots()
        image = ax.imshow(self.reconstructed_image, cmap='jet')
        plt.close(fig)


        # Create a legend with cluster labels
        unique_labels = np.unique(self.cluster_labels)
        self.cluster_colors = np.array([image.cmap(image.norm(label)) for label in unique_labels])
        self.cluster_colors_map = {}
        for c, k in zip(self.cluster_colors, unique_labels):
            self.cluster_colors_map['ECM ' + str(k)] = c 

        if self.save: 
            fig, ax = plt.subplots()
            image = ax.imshow(self.reconstructed_image, cmap='jet')

            ax.grid(False)
            ax.axis('off')

            handles = [Patch(facecolor=color, edgecolor='k', label=f'ECM {label}') for label, color in zip(unique_labels, self.cluster_colors)]
            
            # Place the legend outside the image bbox
            legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=18)
            
            for legend_handle in legend.get_children():
                if isinstance(legend_handle, plt.Line2D):
                    legend_handle.set_edgecolor('black')        # Set axis labels
                    legend_handle.set_linewidth(1)

            # Adjust legend and label font sizes
            #for text in legend.get_texts():
            #    text.set_fontsize(18) 
            plt.show() 
            fig.savefig(self.save_folder+'/ECM_patch_clusters.tiff', bbox_inches='tight', dpi=600)

    def rename_cluster_labels(self, image, background_label, ceg_dict=None):
        # Create a copy of the image to avoid modifying the original
        new_image = np.copy(image)
        
        # Initialize a dictionary to map old labels to new labels
        label_mapping = {}
        
        # Get unique labels in the image
        unique_labels = np.unique(image)
        
        # Initialize the new label counter
        new_label = 1
        
        # Iterate over each unique label
        for label in unique_labels:
            if label == background_label:
                # Map the background label to 0
                label_mapping[label] = 0
            else:
                # Map other labels to new_label and increment
                label_mapping[label] = new_label
                new_label += 1
        
        self.label_mapping = label_mapping
        
        # Apply the label mapping to the image
        for old_label, new_label in label_mapping.items():
            new_image[image == old_label] = new_label
        
        # Modify k-means labels
        if ceg_dict == None: 
            self.cluster_labels = np.vectorize(label_mapping.get)(self.cluster_labels)
        else:
            self.cluster_labels = np.vectorize(label_mapping.get)(ceg_dict.cluster_labels)
    
        return new_image

    def visualize_ecm_cluster_proteins(self):
        background_label = 0
        df = self.cluster_df.copy()

        df = df.drop(background_label)
        
        # Apply Min-Max normalisation to the remaining data to visualize it better
        df = pd.DataFrame(df, index=df.index, columns=df.columns)

        self.cluster_colors = np.delete(self.cluster_colors, background_label, axis=0)
        del self.cluster_colors_map['ECM ' + str(background_label)]

        df_percentage = df.div(df.sum(axis=1), axis=0) * 100

        df_percentage.columns = [col[0] for col in df_percentage.columns]
        # Generate HSL-based color palette with 'num_clusters' distinct colors
        colors = sns.color_palette('hls', 10)

        if self.single:
            # Set general font size
            plt.rcParams.update({'font.size': 18})  # Adjust global font size

            # Plot the bar chart with the specified colors
            ax = df_percentage.plot(kind='bar', stacked=True, color=colors, edgecolor='black', linewidth=0.5)
            ax.grid(False)

            # Move legend outside the bounding box and increase font size
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', edgecolor='black', fontsize=18)
            for legend_handle in legend.get_children():
                if isinstance(legend_handle, plt.Line2D):
                    legend_handle.set_edgecolor('black')
                    legend_handle.set_linewidth(0.5)

            # Modify x-tick labels by adding 'ECM '
            xtick_labels = ["ECM " + str(label) for label in df_percentage.index]
            ax.set_xticklabels(xtick_labels, rotation=0, fontsize=18)

            ax.set_xlabel('Extracellular matrix clusters', fontsize=18)
            ax.set_ylabel('Protein percentage', fontsize=18)


            # Show the plot
            plt.show()

            # Save the figure
            fig = ax.get_figure()
            fig.savefig(self.save_folder+'/ECM_protein_proportions.tiff', bbox_inches='tight', dpi=600)
            
    def get_ecm_patches(self):
        ''' Generates ECM patches '''
        self.load_ecm_data()
        self.pad_ecm_stack()
        self.create_patches()
        self.cluster_ecm_patches()
        self.reconstruct_cluster_patches_to_image()
        self.visualize_ecm_cluster_proteins()

        #background_idx = self.cluster_df.mean(1).argmin()
        self.background_mask = self.cluster_labels != 0
        self.background_removed_ecm_patches = self.ecm_patches_rs[self.background_mask]
        self.background_removed_labels = self.cluster_labels[self.background_mask]
        
    def load_patch_data(self):


        label_encoder = LabelEncoder()
        self.cell_y = label_encoder.fit_transform(self.cell_y_str)
        

        self.ecm_x = self.background_removed_ecm_patches 
        self.ecm_y_str = np.array(['ECM ' + str(i) for i in self.background_removed_labels])
        label_encoder = LabelEncoder()
        self.ecm_y = label_encoder.fit_transform(self.ecm_y_str) + len(np.unique(self.cell_y))

        self.cell_ecm_y_str = np.hstack((self.cell_y_str, self.ecm_y_str))
        cell_binary_label = ['Cells'] * len(self.cell_y_str)
        ecm_binary_label = ['ECM'] * len(self.ecm_y_str)
        self.cell_ecm_binary_label = np.hstack((cell_binary_label, ecm_binary_label))

    def build_cell_cell_graph(self):
        self.G = nx.Graph()

        node_count = 0 
        self.cell_node_pos = []
        self.cell_node_label = []
        # Add nodes with attributes (centroid and cell type)
        for ct, centroid_0, centroid_1 in zip(self.cell_data.celltype.values,
                                                    self.cell_data['centroid-0'].values,
                                                    self.cell_data['centroid-1'].values):
            self.G.add_node('cell_'+str(node_count),cell_type=ct, centroid=(centroid_1, -centroid_0)) 
            node_count+=1
            self.cell_node_pos.append((centroid_1,-centroid_0))
            self.cell_node_label.append(ct)
        
        self.cell_node_pos= np.array(self.cell_node_pos)
        self.cell_node_label = np.array(self.cell_node_label)

        def distance(centroid1, centroid2):
            return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
        
        # Add edges between nearest neighbors
        if self.edge_metric == 'knn':
            # Extract cell nodes and their centroids
            cell_nodes = [node for node in self.G.nodes() if 'cell' in node]
            centroids = np.array([self.G.nodes[node]['centroid'] for node in cell_nodes])
            
            # Use KNN to find neighbors
            nbrs = NearestNeighbors(n_neighbors=self.interaction_k, algorithm='ball_tree').fit(centroids)
            distances, indices = nbrs.kneighbors(centroids)
            
            for i, node1 in enumerate(cell_nodes):
                for j in range(self.interaction_k):
                        node2 = cell_nodes[indices[i][j]]
                        if node1 != node2: 
                            dist = distances[i][j]
                            if dist < self.Dmax_CC:
                                self.G.add_edge(node1, node2, distance=dist, interaction='cell-cell', feature =np.array([1.,0.,0.]))   
            
            self.cell_G = self.G.copy()
        # expansion 
        if self.edge_metric == 'expansion':
            for node1 in self.G.nodes():
                if 'cell' in node1:  # Check if node ID contains 'cell'
                    centroid1 = self.G.nodes[node1]['centroid']
                    distances = [(node2, distance(centroid1, self.G.nodes[node2]['centroid'])) for node2 in self.G.nodes() if node1 != node2 and 'cell' in node2]
                    distances.sort(key=lambda x: x[1])
                    for neighbor, dist in distances[:5]:
                        if dist < self.Dmax_CC: 
                            self.G.add_edge(node1, neighbor, distance=dist, interaction='cell-cell')
            
            self.cell_G = self.G.copy()        
    
    def visualize_cell_cell_interactions(self):
        # Extracting node attributes for nodes with 'cell' in ID
        cell_nodes = [node for node in self.G.nodes() if 'cell' in node]
        node_positions = {node: attributes['centroid'] for node, attributes in self.G.nodes(data=True) if node in cell_nodes}
        node_cell_types = {node: attributes['cell_type'] for node, attributes in self.G.nodes(data=True) if node in cell_nodes}
        unique_cell_types = np.unique(list(node_cell_types.values()))

        # Extract edges where 'cell-cell' interaction exists
        edges_to_plot = [(u, v) for u, v, attr in self.G.edges(data=True) if 'cell-cell' in attr['interaction']]

        # Generate node colors based on cell type
        node_colors = [self.color_map[node_cell_types[node]] for node in node_positions]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.axis('off')

        # Draw nodes and edges
        nx.draw_networkx_nodes(self.G, pos=node_positions, nodelist=cell_nodes, 
                            node_color=node_colors, node_size=self.node_size, ax=ax, 
                            edgecolors='black', linewidths=self.node_linewidth)
        nx.draw_networkx_edges(self.G, pos=node_positions, edgelist=edges_to_plot, 
                            width=self.edge_width, alpha=0.6, ax=ax, edge_color='darkblue')
        
        # Create legend handles
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=cell_type, 
                                    markersize=18, markerfacecolor=color) 
                        for cell_type, color in sorted(self.color_map.items()) if cell_type in unique_cell_types]

        # Add legend for cell-cell interaction
        legend_handles.append(plt.Line2D([0], [0], color='darkblue', lw=1, alpha=1, label='Cell-Cell Interaction'))


        plt.legend(
            handles=legend_handles, 
            loc='upper center',  # Change to 'lower center' or 'center left' for side placement
            bbox_to_anchor=(0.5, 1.2),  # Adjust height if keeping it on top
            ncol=4,  
            fontsize=18,  
            handletextpad=0.5,
            frameon=False  
        )

        # Save figure
        fig.savefig(self.save_folder + '/cell_cell_interactions.tiff', bbox_inches='tight', dpi=300)


        if not self.single:

            # Remove legend properly
            legend = ax.get_legend()
            if legend:
                legend.remove()

            # Save the figure without the legend
            fig.savefig(self.save_folder + '/cell_cell_interactions_no_legend.tiff', bbox_inches='tight', dpi=300)
            plt.close(fig)

    def set_up_colors(self):
        # Access celltype and node labels 
        self.node_labels_all = []
        self.cell_or_ecm_node = []
        for node,attr in self.G.nodes(data=True):
            if 'ecm_coords' in attr: 
                self.node_labels_all.append('ECM ' + str(attr['ecm_labels']))
                self.cell_or_ecm_node.append('ecm')
            if 'centroid' in attr: 
                self.node_labels_all.append(attr['cell_type'])
                self.cell_or_ecm_node.append('cell')
                
        
        # Add consistent node colors 
        unique_nodes = list(set(self.node_labels_all))
        self.color_map = defaultdict(lambda: 'blue')  
        self.color_map.update({n: plt.cm.tab20(i) for i, n in enumerate(unique_nodes)})        
        self.node_colors = [self.color_map[node] for node in self.node_labels_all]

    def build_ecm_ecm_graph(self, k_neighbors=6):
        self.ecm_G = nx.Graph()

        # Find patch centers, ECM labels, and pixel coordinates
        centers = np.array(self.centers)[self.background_mask]
        ecm_labels = self.cluster_labels[self.background_mask]
        ecm_coords = np.column_stack((centers[:, 1], -centers[:, 0]))  # Convert to correct coordinate system
        self.ecm_node_pos = ecm_coords
        self.ecm_node_labels = ecm_labels
        self.color_map.update(self.cluster_colors_map)

        # Add nodes to graph
        self.ecm_nodeid_to_patch_centre = {}
        for i, (x, y) in enumerate(ecm_coords):
            node_id = f"ecm_node{i+1}"
            attributes = {'ecm_labels': ecm_labels[i], 'ecm_coords': (x, y)}
            self.G.add_node(node_id, **attributes)
            self.ecm_G.add_node(node_id, **attributes)
            self.ecm_nodeid_to_patch_centre[(x, y)] = node_id

        # Build KDTree and use KNN to find edges
        coords_list = list(self.ecm_nodeid_to_patch_centre.keys())
        kd_tree = KDTree(coords_list)
        distances, indices = kd_tree.query(coords_list, k=self.interaction_k)  

        # Add edges from KNN
        for i, neighbors in enumerate(indices):
            node1_coord = coords_list[i]
            node1 = self.ecm_nodeid_to_patch_centre[node1_coord]
            for j in neighbors[1:]:  # skip the first one (itself)
                node2_coord = coords_list[j]
                node2 = self.ecm_nodeid_to_patch_centre[node2_coord]

                if not self.ecm_G.has_edge(node1, node2):  # avoid duplicates
                    self.ecm_G.add_edge(node1, node2, interaction='ecm-ecm', feature=[0., 1., 0.], ppi=[])
                    self.G.add_edge(node1, node2, interaction='ecm-ecm', feature=[0., 1., 0.], ppi=[])
    
    def visualize_ecm_ecm_interactions(self, edge_color='red'): 

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.axis('off') 

        # Extract ECM nodes and their attributes
        ecm_nodes = [node for node in self.G.nodes() if 'ecm' in node]
        ecm_nodes_label = ['ECM ' + str(attri['ecm_labels']) for node, attri in self.G.nodes(data=True) if 'ecm' in node]
        node_positions = {node: attr['ecm_coords'] for node, attr in self.G.nodes(data=True) if node in ecm_nodes}
        node_ecm_types = {node: 'ECM ' + str(attributes['ecm_labels']) for node, attributes in self.G.nodes(data=True) if node in ecm_nodes}

        # Extract edges for ECM-ECM interactions
        edges_to_plot = []
        for u, v, attr in self.G.edges(data=True):
            if 'ecm-ecm' in attr['interaction']:
                edges_to_plot.append((u, v))

        # Generate node colors
        node_colors = [self.cluster_colors_map[label] for label in ecm_nodes_label]

        # Plot ECM nodes as squares
        nx.draw_networkx_nodes(self.G, pos=node_positions, nodelist=ecm_nodes, node_color=node_colors, 
                            node_size=self.node_size, ax=ax, edgecolors='gray', 
                            linewidths=self.node_linewidth, node_shape='^')

        # Plot ECM-ECM edges
        nx.draw_networkx_edges(self.G, pos=node_positions, edgelist=edges_to_plot, width=self.edge_width, 
                            alpha=0.6, ax=ax, edge_color='darkgreen')  

        # Create legend
        legend_handles = []

        # Add ECM type legend (square markers)
        unique_labels = set(ecm_nodes_label)
        for label in unique_labels:
            color = self.cluster_colors_map[str(label)]
            legend_handles.append(plt.Line2D([0], [0], marker='^', color='w', label=str(label), 
                                            markersize=18, markerfacecolor=color))

        # Add edge legend
        edge_legend_handle = plt.Line2D([0], [0], color='darkgreen', lw=1, alpha=1, label='ECM-ECM Interaction')
        legend_handles.append(edge_legend_handle)

        # Adding legend at the top with multiple rows
        plt.legend(
            handles=legend_handles, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.15),  # Adjusted position above the plot
            ncol=4,  # Arranges legend items in rows of 4
            fontsize=18,
            frameon=False,
            handletextpad=0.5
        )        

        # Save the figure
        fig.savefig(self.save_folder + '/ecm_ecm_interactions.tiff', bbox_inches='tight', dpi=300)

        if not self.single:
            # Remove legend properly
            legend = ax.get_legend()
            if legend:
                legend.remove()
            
            # Save the figure without the legend
            fig.savefig(self.save_folder + '/ecm_ecm_interactions_no_legend.tiff', bbox_inches='tight', dpi=300)
            plt.close(fig)

    def visualize_cell_ecm_interactions(self):
        pos = {}
        cell_node_y = []
        ecm_node_y = []

        for node, attr in self.G.nodes(data=True):
            if 'ecm_coords' in attr: 
                pos[node] = attr['ecm_coords']
                ecm_node_y.append('ECM ' + str(attr['ecm_labels']))
            if 'centroid' in attr: 
                pos[node] = attr['centroid']
                cell_node_y.append(attr['cell_type'])

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.axis('off') 
        
        cell_cell_edges = []
        ecm_ecm_edges = []
        cell_ecm_edges = []
        
        for u, v, attr in self.G.edges(data=True):
            if 'cell-ecm' in attr['interaction']:
                cell_ecm_edges.append((u, v))

        # Generate colors for cell and ECM nodes separately
        cell_node_colors = [self.color_map[str(n)] for n in cell_node_y]
        ecm_node_colors = [self.color_map[str(n)] for n in ecm_node_y]

        # Plot cell nodes (circle shape)
        cell_nodes = [node for node, attr in self.G.nodes(data=True) if 'centroid' in attr]
        nx.draw_networkx_nodes(self.G, pos=pos, nodelist=cell_nodes, node_color=cell_node_colors, 
                               node_size=self.node_size, ax=ax, edgecolors='black', 
                               linewidths=self.node_linewidth, node_shape='o')

        # Plot ECM nodes (triangle shape)
        ecm_nodes = [node for node, attr in self.G.nodes(data=True) if 'ecm_coords' in attr]
        nx.draw_networkx_nodes(self.G, pos=pos, nodelist=ecm_nodes, node_color=ecm_node_colors, 
                               node_size=self.node_size, ax=ax, edgecolors='gray', 
                               linewidths=self.node_linewidth, node_shape='^')

        # Plot edges
        nx.draw_networkx_edges(self.G, pos=pos, edgelist=cell_ecm_edges, width=self.edge_width, 
                               alpha=0.6, ax=ax, edge_color='darkred') 

        # Create legend
        legend_handles = []

        # Add cell type legend (circle markers)
        unique_cell_types = set(cell_node_y)
        for cell_type in unique_cell_types:
            color = self.color_map[cell_type]
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=cell_type, 
                                             markersize=18, markerfacecolor=color))

        # Add ECM type legend (triangle markers)
        unique_ecm_types = set(ecm_node_y)
        for ecm_type in unique_ecm_types:
            color = self.color_map[ecm_type]
            legend_handles.append(plt.Line2D([0], [0], marker='^', color='w', label=ecm_type, 
                                             markersize=18, markerfacecolor=color))

        # Add edge legend
        cell_ecm_handle = plt.Line2D([0], [0], color='darkred', lw=1, alpha=1, label='Cell-ECM Interaction')
        legend_handles.append(cell_ecm_handle)
                                        
        # Adding legend
        plt.legend(
            handles=legend_handles, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.25),  # Adjusted height for more spacing
            ncol=4,  # Set to 4 columns
            fontsize=12,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0,  # Adjust spacing between columns
            labelspacing=1.2  # Adjust spacing between rows
        )        
        fig.savefig(self.save_folder + '/cell_ecm_interactions.tiff', bbox_inches='tight')

        if not self.single:
            # Remove legend 
            legend = ax.get_legend()
            if legend:
                legend.remove()
            
            # Save the figure without the legend
            fig.savefig(self.save_folder + '/cell_ecm_interactions_no_legend.tiff', bbox_inches='tight', dpi=300)
            plt.close(fig)

    def visualize_cell_ecm_graph(self):
        pos = {}
        cell_node_y = []
        ecm_node_y = []

        for node, attr in self.G.nodes(data=True):
            if 'ecm_coords' in attr: 
                pos[node] = attr['ecm_coords']
                ecm_node_y.append('ECM ' + str(attr['ecm_labels']))
            if 'centroid' in attr: 
                pos[node] = attr['centroid']
                cell_node_y.append(attr['cell_type'])

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.axis('off') 
        
        cell_cell_edges = []
        ecm_ecm_edges = []
        cell_ecm_edges = []
        
        for u, v, attr in self.G.edges(data=True):
            if 'cell-cell' in attr['interaction']:
                cell_cell_edges.append((u, v))
            elif 'ecm-ecm' in attr['interaction']:
                ecm_ecm_edges.append((u, v))
            elif 'cell-ecm' in attr['interaction']:
                cell_ecm_edges.append((u, v))

        # Generate colors for cell and ECM nodes separately
        cell_node_colors = [self.color_map[str(n)] for n in cell_node_y]
        ecm_node_colors = [self.color_map[str(n)] for n in ecm_node_y]

        # Plot edges
        nx.draw_networkx_edges(self.G, pos=pos, edgelist=cell_cell_edges, width=self.edge_width, 
                               alpha=0.3, ax=ax, edge_color='darkblue') 
        nx.draw_networkx_edges(self.G, pos=pos, edgelist=ecm_ecm_edges, width=self.edge_width, 
                               alpha=0.3, ax=ax, edge_color='darkgreen') 
        nx.draw_networkx_edges(self.G, pos=pos, edgelist=cell_ecm_edges, width=self.edge_width, 
                               alpha=0.6, ax=ax, edge_color='darkred') 

        
        # Plot cell nodes (circle shape)
        cell_nodes = [node for node, attr in self.G.nodes(data=True) if 'centroid' in attr]
        nx.draw_networkx_nodes(self.G, pos=pos, nodelist=cell_nodes, node_color=cell_node_colors, 
                               node_size=self.node_size, ax=ax, edgecolors='black', 
                               linewidths=self.node_linewidth, node_shape='o', alpha=0.8)

        # Plot ECM nodes (triangle shape with similar alpha)
        ecm_nodes = [node for node, attr in self.G.nodes(data=True) if 'ecm_coords' in attr]
        nx.draw_networkx_nodes(self.G, pos=pos, nodelist=ecm_nodes, node_color=ecm_node_colors, 
                               node_size=self.node_size, ax=ax, edgecolors='gray', 
                               linewidths=self.node_linewidth, node_shape='^', alpha=0.8)

        # Create legend
        legend_handles = []

        # Add cell type legend (circle markers)
        unique_cell_types = set(cell_node_y)
        for cell_type in unique_cell_types:
            color = self.color_map[cell_type]
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=cell_type, 
                                             markersize=18, markerfacecolor=color))

        # Add ECM type legend (triangle markers)
        unique_ecm_types = set(ecm_node_y)
        for ecm_type in unique_ecm_types:
            color = self.color_map[ecm_type]
            legend_handles.append(plt.Line2D([0], [0], marker='^', color='w', label=ecm_type, 
                                             markersize=18, markerfacecolor=color))

        # Add edge legends
        legend_handles.append(plt.Line2D([0], [0], color='darkblue', lw=1, alpha=1, label='Cell-Cell Interaction'))
        legend_handles.append(plt.Line2D([0], [0], color='darkgreen', lw=1, alpha=1, label='ECM-ECM Interaction'))
        legend_handles.append(plt.Line2D([0], [0], color='darkred', lw=1, alpha=1, label='Cell-ECM Interaction'))
                                        
        # Adding legend
        plt.legend(
            handles=legend_handles, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.25),  
            ncol=6,  # Set to 5 columns
            fontsize=12,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0,  
            labelspacing=1.2  
        )        
        fig.savefig(self.save_folder + '/cell_ecm_graph.tiff', bbox_inches='tight')

        if not self.single:
            # Remove legend 
            legend = ax.get_legend()
            if legend:
                legend.remove()
            
            # Save the figure without the legend
            fig.savefig(self.save_folder + '/cell_ecm_graph_no_legend.tiff', bbox_inches='tight', dpi=300)
            plt.close(fig)

    def get_pygeo_data(self, y, g_type):

        if g_type == 'ecm_graph':
            temp_G = self.ecm_G
            pos = self.ecm_node_pos
            node_label = self.ecm_node_labels

        if g_type == 'cell_graph':
            temp_G = self.cell_G
            pos = self.cell_node_pos
            node_label = self.cell_node_label

        if g_type == 'cell_ecm_graph':
            temp_G = self.G
            pos = []
            node_label = [ ]
            # Get all the nodes 
            for n,attri in self.G.nodes(data=True):
                if 'cell' in n:
                    pos.append(attri['centroid'])
                    node_label.append(attri['cell_type'])

                else:
                    pos.append(attri['ecm_coords'])
                    node_label.append(attri['ecm_labels'])


        y = torch.tensor([y], dtype=torch.long)

        x = []
        for n,attri in temp_G.nodes(data=True):
            x.append(attri['feature'])

        edge_attr = []
        for u,v,attri in temp_G.edges(data=True):
            edge_attr.append(attri['feature'])
        
        x = torch.tensor(x, dtype=torch.float32).squeeze(1)
        edge_attr = torch.tensor(edge_attr,  dtype=torch.float32).squeeze(1).float()

        node_mapping = {node: i for i, node in enumerate(temp_G.nodes())}
        self.node_mapping = node_mapping

        # Get all nodes but only cell-ecm interactions 
        if g_type == 'cell_ecm_graph':
            edges_int = [(node_mapping[u], node_mapping[v]) for u, v in temp_G.edges() if temp_G[u][v].get('interaction') == 'cell-ecm']

        else: 
            edges_int = [(node_mapping[u], node_mapping[v]) for u, v in temp_G.edges()]

        edge_index = torch.tensor(edges_int, dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, node_labels=node_label)
        return data 

    def add_node_ohe_features(self, unique_labels, g_type):
            
        enc = OneHotEncoder(handle_unknown='ignore')
        ohe_encoder = enc.fit(unique_labels.reshape(-1,1))    
        if g_type == 'cellgraph':
            for n,attri in self.cell_G.nodes(data=True):
                if 'cell' in n:
                    attri['feature'] = ohe_encoder.transform(np.array(attri['cell_type']).reshape(-1,1)).toarray()
                if 'ecm' in n:
                    attri['feature'] = ohe_encoder.transform(np.array(str(attri['ecm_labels'])).reshape(-1,1)).toarray()
        
        if g_type == 'ecmgraph':
            for n,attri in self.ecm_G.nodes(data=True):
                if 'cell' in n:
                    attri['feature'] = ohe_encoder.transform(np.array(attri['cell_type']).reshape(-1,1)).toarray()
                if 'ecm' in n:
                    attri['feature'] = ohe_encoder.transform(np.array(str(attri['ecm_labels'])).reshape(-1,1)).toarray()
        
                
        if g_type == 'cellecmgraph':
            for n,attri in self.G.nodes(data=True):
                if 'cell' in n:
                    attri['feature'] = ohe_encoder.transform(np.array(attri['cell_type']).reshape(-1,1)).toarray()
                if 'ecm' in n:
                    attri['feature'] = ohe_encoder.transform(np.array(str(attri['ecm_labels'])).reshape(-1,1)).toarray()

    def add_edge_ohe_features(self): # Remove not needed, done manually 
        all_edge_labels = []

        for u,v,attri in self.G.edges(data=True):
            all_edge_labels.append(attri['interaction'])

            enc = OneHotEncoder(handle_unknown='ignore')
            ohe_encoder = enc.fit(np.array(all_edge_labels).reshape(-1,1))

        for u,v,attri in self.G.edges(data=True):
            attri['feature'] = ohe_encoder.transform(np.array(attri['interaction']).reshape(-1,1)).toarray()

    def count_cells_on_patches(self,cell_mask_path):

        scaler = MinMaxScaler()
        cells_per_patch = {}
        background_idx = self.cluster_df.sum(axis=1).argmin()
        for i in np.unique(self.reconstructed_image):
            if i != background_idx: 
                seg_mask = imread(cell_mask_path)
                patch_mask = self.reconstructed_image.copy()
                patch_mask = patch_mask == i
                #print(i)
                #plt.imshow(patch_mask)
                #plt.show()
                seg_mask[~patch_mask] = 0 
                cell_counts = self.cell_data[self.cell_data.ObjectNumber.isin(np.unique(seg_mask))].celltype.value_counts()
                
                # Convert to DataFrame for proper scaling
                cell_counts_df = cell_counts.reset_index()
                cell_counts_df.columns = ['celltype', 'count']
                
                # Scale counts and store back in dictionary
                cell_counts_df['scaled_count'] = scaler.fit_transform(cell_counts_df['count'].values.reshape(-1, 1)).flatten()
                cells_per_patch[i] = cell_counts_df.set_index('celltype')['scaled_count'].to_dict()
        
            # Visualize 
        # Convert the data into a DataFrame
        df = pd.DataFrame(cells_per_patch).fillna(0).round(2)
        df.columns = ['ECM_cluster_'+ str(int(i)) for i in list(cells_per_patch.keys())]

        # Plotting
        plt.figure(figsize=(14, 12), dpi=300)
        sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5, fmt='g')

        # Adding labels and title
        plt.title("Cell Type on ECM Clusters Patches")
        plt.xlabel("Sample Index")
        plt.ylabel("Cell Type")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

