

### Imports ### 

import pandas as pd 
from tifffile import imread, imwrite 
import numpy as np 
import os 
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
#from permutation_test import *
from Graph_builder import GraphBuilder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import argparse
from matplotlib.patches import Patch
import random

random.seed(42)  
np.random.seed(42)  

### Cell Matrix Graph Main Module  ###

class Cell_ECM_Graphs(GraphBuilder):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Pass manually provided arguments
        self.single=False
        
    def prechecks(self):

        # Check panel has ecm column
        self.panel = pd.read_csv(self.panel_path)
        if 'ecm' in self.panel.columns:
            print('Check list: 0/2')
            print(' - ecm column found in panel. 1/2')
        else:
            raise ValueError('Failed check, please add ecm column to panel, with 1 indicating ECM markers')
        
        # Check if celltype in cell data column
        celltype_bool = []
        for i in self.cell_data_path:
            df = pd.read_csv(i)
            if 'celltype' in df:
                celltype_bool.append(True)
            else:
                celltype_bool.append(False)    
        if np.all(celltype_bool):
            print(' - celltype found in cell data files. 2/2')
        else:
            raise ValueError('Failed checks, please add celltype column to celldata file' )
    
    def build_multiple_graphs(self, Dmax_CC, Dmax_CE, interaction_k, norm='znorm', feature_type='std+mean'):
        ''' To process multiple ROIs together '''
        self.ceg_dict = {}
        print('Building Cell-ECM-Graphs...')
        for i in range(len(self.full_stack_img_path)):

            save_folder_sub = self.save_folder + '/Results' + '_' + str(i)
            os.makedirs(save_folder_sub, exist_ok=True)
            ceg =  GraphBuilder(full_stack_img_path=self.full_stack_img_path[i],
                                panel_path=self.panel_path,
                                cell_data_path=self.cell_data_path[i],
                                save_folder = save_folder_sub,  
                                single=self.single,
                                Dmax_CC= Dmax_CC,
                                Dmax_CE = Dmax_CE,
                                feature_type=feature_type,
                                norm=norm,
                                patch_size= 5,
                                interaction_k=interaction_k
                                )
            ceg.load_imgs()
            ceg.build_cell_ecm_graph()
            self.ceg_dict[i] = ceg
            print('ROI '+str(i)+ ' complete.')   

    def joint_ecm_clustering(self):
        print('Clustering all ECM patches together ... ')
        # get ECM patches 
        means = []
        patches_per_image = self.ceg_dict[0].ecm_patches_rs.shape[0] 
        for i in range(len(self.ceg_dict)):
            means.append(self.ceg_dict[i].ecm_patches_rs.mean(axis=(1,2)))
        ecm_means = np.concatenate(means)

        # Recluster 
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=100, max_iter=600, algorithm='elkan', tol=1e-5)
        kmeans.fit(ecm_means)
        cluster_labels = kmeans.labels_
        
        # Find background label 
        background_means = []
        for i in np.unique(cluster_labels):
            mask = cluster_labels == i 
            background_means.append(ecm_means[mask].mean())
        background_idx = np.argmin(background_means)
        
        # Update cluster labels and background 
        cluster_labels_per_G = {}
        background_per_G = {}
        counter = 0 
        
        for i in self.ceg_dict:
            cluster_labels_per_G[i] = cluster_labels[(patches_per_image*counter):(patches_per_image*(counter+1))]
            background_per_G[i] = cluster_labels_per_G[i] != np.unique(cluster_labels)[background_idx]
            counter +=1 
            self.ceg_dict[i].cluster_labels = cluster_labels_per_G[i]
            self.ceg_dict[i].background_mask = background_per_G[i]
            ecm_nodes_to_remove = [n for n,attri in self.ceg_dict[i].G.nodes(data=True) if 'ecm' in n ]
            self.ceg_dict[i].G.remove_nodes_from(ecm_nodes_to_remove)

            edges_to_remove = [(u, v) for u, v, attri in self.ceg_dict[i].G.edges(data=True) if attri['interaction'] == 'cell-ecm']
            self.ceg_dict[i].G.remove_edges_from(edges_to_remove)
        
            edges_to_remove = [(u, v) for u, v, attri in self.ceg_dict[i].G.edges(data=True) if attri['interaction'] == 'ecm-ecm']
            self.ceg_dict[i].G.remove_edges_from(edges_to_remove)
  
            self.rename_cluster_labels(self.ceg_dict[i].reconstructed_image, np.unique(cluster_labels)[background_idx], self.ceg_dict[i])
            self.ceg_dict[i].background_mask = self.cluster_labels != 0
            self.ceg_dict[i].background_removed_ecm_patches = self.ceg_dict[i].ecm_patches_rs[self.ceg_dict[i].background_mask]
            self.ceg_dict[i].background_removed_labels = self.cluster_labels[self.ceg_dict[i].background_mask]
            self.ceg_dict[i].build_ecm_ecm_graph()
            self.ceg_dict[i].reconstruct_cluster_patches_to_image()
            self.ceg_dict[i].build_cell_ecm_interactions()
            
            # Update labels 1 to 3
            for n, attri in self.ceg_dict[i].G.nodes(data=True):
                if 'ecm' in n:
                    attri['ecm_labels'] = self.label_mapping[attri['ecm_labels']]


        self.ceg_dict[0].set_up_colors()
        for i in range(len(self.ceg_dict)):
            self.ceg_dict[i].color_map = self.ceg_dict[0].color_map       
            
    def visualize_joint_ecm_clustered_patches(self):
        print('Visualizing joint ECM clusters ... ')
        for i in range(len(self.ceg_dict)):
            fig, ax = plt.subplots(dpi=300)
            if self.single == False:
                plt.close(fig)
            image = ax.imshow(self.ceg_dict[i].reconstructed_image, cmap='jet')

            ax.grid(self.ceg_dict[i])
            ax.axis('off')

            # Create a legend with cluster labels
            unique_labels = np.unique(self.ceg_dict[i].cluster_labels)
            self.ceg_dict[i].cluster_colors = np.array([image.cmap(image.norm(label)) for label in unique_labels])
            self.ceg_dict[i].cluster_colors_map = {}
            for c, k in zip(self.ceg_dict[i].cluster_colors, unique_labels):
                self.ceg_dict[i].cluster_colors_map['ECM ' + str(k)] = c 

            handles = [Patch(facecolor=color, edgecolor='k', label=f'ECM {label}') for label, color in zip(unique_labels, self.ceg_dict[i].cluster_colors)]

            # Place the legend outside the image bbox
            legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)

            for legend_handle in legend.get_children():
                if isinstance(legend_handle, plt.Line2D):
                    legend_handle.set_edgecolor('black')        # Set axis labels
                    legend_handle.set_linewidth(1)
            # Adjust legend and label font sizes
            for text in legend.get_texts():
                text.set_fontsize(7) 

            fig.savefig(self.ceg_dict[i].save_folder+'/ecm_patch_clusters.png', bbox_inches='tight')
            plt.show() 

    def visualize_multiple_graphs(self):
        print('Visualizing cell-cell, ecm-ecm, cell-ecm interactions')
        # Improve method for color alignments 
        #self.ceg_dict[0].color_map['ECM 1'] = self.ceg_dict[0].color_map.pop('ecm_cluster_1')
        #self.ceg_dict[0].color_map['ECM 2'] = self.ceg_dict[0].color_map.pop('ecm_cluster_2')
        #self.ceg_dict[0].color_map['ECM 3'] = self.ceg_dict[0].color_map.pop('ecm_cluster_3')

        for i in range(len(self.ceg_dict)):
                self.ceg_dict[i].visualize_cell_cell_interactions()
                self.ceg_dict[i].visualize_ecm_ecm_interactions()
                self.ceg_dict[i].visualize_cell_ecm_interactions()
                self.ceg_dict[i].visualize_cell_ecm_graph()    

    def visualize_cluster_protein_percentages(self):
        'Visualizing ECM proteins per ECM cluster'
        cluster_dfs = []
        for i in range(len(self.ceg_dict)):
            min_max_img = self.ceg_dict[i].full_stack_imgs_min_max[self.ceg_dict[i].ecm_mask].transpose(1,2,0)
            ecm_patch_img = self.ceg_dict[i].reconstructed_image

            cluster_means = {}

            for j in np.unique(self.ceg_dict[i].cluster_labels): 

                cluster_means[j] = min_max_img[ecm_patch_img == j].mean(0)

            cluster_df = pd.DataFrame(cluster_means, index=[self.panel[self.ceg_dict[i].ecm_mask].name.values])
            background_label = cluster_df.mean(axis=0).argmin()
            df = cluster_df.copy()
            df = df.drop(background_label, axis=1)
            df_percentage = df.div(df.sum(axis=0), axis=1) * 100



            cluster_dfs.append(df_percentage)
        
        concat_dfs = pd.concat(cluster_dfs)
        df_to_plot = concat_dfs.groupby(concat_dfs.index).mean()
        colors = sns.color_palette('hls', 10)
        df_to_plot.index = [i[0] for i in df_to_plot.index]
        df_copy = df_to_plot.copy()

        # Plot the bar chart with the specified colors
        ax = df_to_plot.T.plot(kind='bar', stacked=True, color=colors, edgecolor='black',
                                linewidth=1)
        ax.grid(False)

        # Move legend outside the bounding box and increase font size
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', edgecolor='black', fontsize=18)
        for legend_handle in legend.get_children():
            if isinstance(legend_handle, plt.Line2D):
                legend_handle.set_edgecolor('black')
                legend_handle.set_linewidth(0.5)

        # Modify x-tick labels by adding 'ECM '
        xtick_labels = ["ECM " + str(label) for label in df_percentage.columns]
        ax.set_xticklabels(xtick_labels, rotation=0, fontsize=18)

        ax.set_xlabel('Extracellular matrix clusters', fontsize=18)
        ax.set_ylabel('Protein percentage', fontsize=18)

        # Set DPI for the figure
        plt.gcf().set_dpi(600)

        fig = ax.get_figure()
        fig.set_dpi(300)

        # Show the plot
        fig.savefig(self.save_folder+'/joint_cluster_protein_exp.png', bbox_inches='tight')
        plt.show()
        plt.close(fig)
