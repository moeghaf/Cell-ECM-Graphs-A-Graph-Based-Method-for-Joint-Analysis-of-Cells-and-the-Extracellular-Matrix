
import numpy as np
import matplotlib.pyplot as plt
import cv2  
import random
import seaborn as sns 
import pandas as pd 
import os 

from scipy.spatial import Delaunay
from tifffile import imread, imwrite
from matplotlib.colors import LinearSegmentedColormap

class imc_ecm_data_generator:
    def __init__(self, img_size, n_points, save_location= False):
        self.img_size = img_size 
        self.n_points = n_points   
        if save_location: 
            self.save_location = save_location
            os.makedirs(self.save_location+'/cell_data', exist_ok=True)
            os.makedirs(self.save_location+'/imc_data', exist_ok=True)


    def simulate_ECM_structure(self, thickness=2):

        # Initialize binary map
        binary_map = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Generate ECM points
        ecm_points = np.random.randint(0, self.img_size, size=(self.n_points, 2))
        
        # Compute Delaunay triangulation
        tri = Delaunay(ecm_points)
        for simplex in tri.simplices:
            pts = ecm_points[simplex]
            for i in range(3):  # Each simplex has 3 edges
                pt1, pt2 = tuple(pts[i]), tuple(pts[(i + 1) % 3])
                cv2.line(binary_map, pt1, pt2, 1, thickness)

        
        self.binary_map = binary_map
        
    def add_ECM_substructure(self, second_condition):
        centre_points = (random.randint(self.img_size // 4, 3 * self.img_size // 4), 
                        random.randint(self.img_size // 4, 3 * self.img_size // 4))
        
        circle_binary_map = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        filled_circle = cv2.circle(circle_binary_map.copy(), centre_points, 50, 1, -1)  # Filled circle
        unfilled_circle = cv2.circle(circle_binary_map.copy(), centre_points, 50, 1, 2)  # Outline only
        small_unfilled_circle = cv2.circle(circle_binary_map.copy(), centre_points, 25, 1, 2)  # Outline only
        small_filled_circle = cv2.circle(circle_binary_map.copy(), centre_points, 25, 1, -1)  # Outline only

        if not second_condition:
            self.binary_map[filled_circle.astype(bool)] = 0 
            
            structure_bm = self.binary_map + unfilled_circle + unfilled_circle
            structure_bm[structure_bm == 3] = 2 
            structure_bm = structure_bm + small_unfilled_circle + small_unfilled_circle + small_unfilled_circle  
        else: 
            structure_bm = self.binary_map + unfilled_circle + unfilled_circle
            structure_bm[structure_bm == 3] = 2 
            structure_bm[small_filled_circle.astype(bool)] = 0 
            structure_bm = structure_bm + small_unfilled_circle + small_unfilled_circle + small_unfilled_circle  
            structure_bm[structure_bm == 4] = 3 

        self.final_ecm_structure = structure_bm
        return self.final_ecm_structure

    def find_boundary_pixels(self,label):
        """Finds the boundary pixels of a given label in sub_network."""
        mask = (self.final_ecm_structure == label).astype(np.uint8)
        edges = cv2.Canny(mask * 255, 100, 200)  # Detect edges
        y, x = np.where(edges > 0)  # Get pixel coordinates
        return list(zip(x, y))

    def place_cells(self, boundary_pixels, num_cells):
        """Randomly places circles near boundary pixels."""
        num_cells = min(num_cells, len(boundary_pixels))  # Avoid excessive cell placement
        selected_points = np.random.choice(len(boundary_pixels), num_cells, replace=False)
        return [boundary_pixels[i] for i in selected_points]
    
    def place_cells_on_ecm(self, immune_cells=False):
        
        # Get boundary pixels for each ECM label
        boundary_1 = self.find_boundary_pixels(1)
        boundary_2 = self.find_boundary_pixels(2)
        boundary_3 = self.find_boundary_pixels(3)

        # Place cells near ECM regions
        self.cells_A_centroids = self.place_cells(boundary_1, 4000)
        if immune_cells:
                self.cells_E_centroids = self.place_cells(boundary_1, 50)
        self.cells_B_centroids = self.place_cells(boundary_2, 200)
        cells_C_D = self.place_cells(boundary_3, 100)  # Ensure enough for C and D

        # Split C and D equally
        self.cells_C_centroids = cells_C_D[:50]
        self.cells_D_centroids = cells_C_D[50:]
        return 
    
    def visualize_cells_on_ecm(self, radius=2):
        # Generate a color palette using seaborn
        palette = sns.color_palette("hls", 4)
        colors = [palette[0], palette[1], palette[2], palette[3]]

        # Plot ECM + Cells
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(self.final_ecm_structure, cmap="gray")

        # Draw circles for each cell type
        for x, y in self.cells_A_centroids:
            ax.add_patch(plt.Circle((x, y), radius, color=colors[0], fill=True, alpha=0.7))

        for x, y in self.cells_B_centroids:
            ax.add_patch(plt.Circle((x, y), radius, color=colors[1], fill=True, alpha=0.7))

        for x, y in self.cells_C_centroids:
            ax.add_patch(plt.Circle((x, y), radius, color=colors[2], fill=True, alpha=0.7))

        for x, y in self.cells_D_centroids:
            ax.add_patch(plt.Circle((x, y), radius, color=colors[3], fill=True, alpha=0.7))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Simulated ECM with Cell Types")
        plt.savefig('cells_on_ecm.pdf')

    def save_cell_data(self, savename, immune=False):
        '''Save cell data - cell type and centroids for cell-ECM graph'''

        # Create cell type labels to match centroids  
        cells_A_label = ['Cell A'] * len(self.cells_A_centroids)
        cells_B_label = ['Cell B'] * len(self.cells_B_centroids)
        cells_C_label = ['Cell C'] * len(self.cells_C_centroids)
        cells_D_label = ['Cell D'] * len(self.cells_D_centroids)
        if immune: 
            cells_E_label = ['Cell E'] * len(self.cells_E_centroids)
        # Combine cell and centroids 
        if immune: 
            cell_types = np.concatenate((cells_A_label, cells_B_label, cells_C_label, cells_D_label,cells_E_label)).reshape(-1,1)

            centroids = np.concatenate((self.cells_A_centroids,
                                        self.cells_B_centroids,
                                        self.cells_C_centroids,
                                        self.cells_D_centroids,
                                        self.cells_E_centroids,
                                        ))
        else: 
            cell_types = np.concatenate((cells_A_label, cells_B_label, cells_C_label, cells_D_label)).reshape(-1,1)

            centroids = np.concatenate((self.cells_A_centroids,
                                        self.cells_B_centroids,
                                        self.cells_C_centroids,
                                        self.cells_D_centroids))

        # load as DF 
        cell_data = np.concatenate((cell_types, centroids),axis=1)
        cell_data = pd.DataFrame(cell_data, columns=['celltype', 'centroid-1', 'centroid-0'])

        cell_data.to_csv(self.save_location+'/cell_data'+'/simulated_cell_data_'+ str(savename) +'.csv')

    def get_imc_ecm_markers(self):
        # Define mapping for labels 0, 1, 2
        label_map = {
            1: [1, 0, 0],
            2: [0, 1, 0],
            3: [0, 0, 1],
        }

        # Get shape of sub_network
        H, W = self.final_ecm_structure.shape

        # Initialize array for 3 channels (RGB) per pixel
        ecm_markers = np.zeros((H, W, 3))

        # Assign the first 4 markers based on sub_network labels
        for label, values in label_map.items():
            mask = self.final_ecm_structure == label  # Find pixels with this label
            ecm_markers[mask, :3] = values

        # Generate different random values for each channel
        random_values_r = np.random.uniform(0.3, 1000, size=(H, W))  # Channel 1 (Red)
        random_values_g = np.random.uniform(0.5, 500, size=(H, W))  # Channel 2 (Green)
        random_values_b = np.random.uniform(1, 1500, size=(H, W))  # Channel 3 (Blue)

        # Apply the random values to the markers where they are non-zero
        ecm_markers[ecm_markers[:, :, 0] > 0, 0] = random_values_r[ecm_markers[:, :, 0] > 0]  # Red channel
        ecm_markers[ecm_markers[:, :, 1] > 0, 1] = random_values_g[ecm_markers[:, :, 1] > 0]  # Green channel
        ecm_markers[ecm_markers[:, :, 2] > 0, 2] = random_values_b[ecm_markers[:, :, 2] > 0]  # Blue channel

        self.imc_ecm_channels = ecm_markers


    def visualize_ecm_imc_markers(self):
        # Create custom colormaps
        black_yellow = LinearSegmentedColormap.from_list("black_yellow", [(0, 0, 0), (1, 1, 0)])
        black_red = LinearSegmentedColormap.from_list("black_red", [(0, 0, 0), (1, 0, 0)])
        black_green = LinearSegmentedColormap.from_list("black_green", [(0, 0, 0), (0, 1, 0)])
        black_blue = LinearSegmentedColormap.from_list("black_blue", [(0, 0, 0), (0, 0, 1)])

        fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=600)
        
        # Show the images and remove axes
        im0 = ax[0].imshow(self.imc_ecm_channels[:, :, 0], cmap=black_red)
        ax[0].axis('off')
        im1 = ax[1].imshow(self.imc_ecm_channels[:, :, 1], cmap=black_green)
        ax[1].axis('off')
        im2 = ax[2].imshow(self.imc_ecm_channels[:, :, 2], cmap=black_blue)
        ax[2].axis('off')
        
        # Add color bars
        fig.colorbar(im0, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()

    def save_imc_data(self, savename):
        ''' Save IMC ECM channels for Cell-ECM graphs '''
                
        imc_ecm_channels_rs = self.imc_ecm_channels.transpose(2,0,1)
        imwrite(self.save_location+'/imc_data/sim_imc_ecm_'+str(savename)+'.tiff',imc_ecm_channels_rs)
     