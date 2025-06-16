# 🧬 CEG: Cell–ECM Graph Analysis

This repository provides code and notebooks for constructing, analyzing, and visualizing **Cell–ECM Graphs (CEGs)** — focusing on cell–extracellular matrix (ECM) interactions in biological datasets.

---

## 📚 Overview

This project includes tools for:

- 📊 **Graph construction** from spatial and molecular data  
- 🔁 **Permutation testing** for statistical analysis  
- 🧠 **Explainable Graph Neural Network (GNN) models**  
- 🖼️ **Publication-ready figure generation**  

---

## 📁 Repository Structure

| Folder/File                | Description                                                         |
|---------------------------|---------------------------------------------------------------------|
| `main_figures/`           | Reproduce primary figures and results for the publication.          |
| `Notebooks/supplementary/`| Additional analyses, exploratory plots, and extended context.       |
| `data/`                   | Place raw data and metadata here.                                   |
| `scripts/` (if applicable)| Scripts used across multiple notebooks.                             |

---

## 🚀 How to Use

bash
Copy
Edit
pip install -r requirements.txt
Or follow the instructions in the relevant notebook if environment details differ.

2. Prepare Data
Place your raw data and metadata in the data/ directory.

Alternatively, update the data paths in the notebooks as needed.

3. Run Notebooks
Open any notebook in the main_figures/ or Notebooks/supplementary/ directories.

Run all cells in order to reproduce results.

✅ Main notebooks will reproduce key results and publication-quality plots.
🧪 Supplementary notebooks offer additional context and analyses.

🧩 Key Modules
Module	Description
Graph_builder.py	Constructs cell–ECM graphs from spatial and molecular data.
Permutation_test.py	Performs permutation-based statistical testing of interaction frequencies.
CellECMGraphs_multiple.py	Classes/utilities for handling and analyzing multiple CEGs.
Helper_functions.py	Miscellaneous utilities for data processing and visualization.
SimData_Generator.py	Tools for generating simulated cell–ECM interaction data.

📄 Citation
If you use this code or data, please cite:

[Your Paper Title]
Journal/Preprint, Year

📬 Contact
For questions, issues, or contributions:

Open an Issue

Or contact: [your email]
