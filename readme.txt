
# 🧬 CEG: Cell–ECM Graph Analysis

This repository provides code and notebooks for constructing, analyzing, and visualizing Cell–ECM Graphs (CEGs) — focusing on cell–extracellular matrix (ECM) interactions in biological datasets.

--------------------------------------------------------------------------------

📚 Overview

This project includes tools for:

- Graph construction from spatial and molecular data  
- Permutation testing for statistical analysis  
- Explainable Graph Neural Network (GNN) models  
- Publication-ready figure generation  

--------------------------------------------------------------------------------

📁 Repository Structure

Folder/File                  | Description
--------------------------- | ------------------------------------------------------------
main_figures/               | Reproduce primary figures and results for the publication.
Notebooks/supplementary/    | Additional analyses, exploratory plots, and extended context.
data/                       | Place raw data and metadata here.
scripts/ (if applicable)    | Scripts used across multiple notebooks.

--------------------------------------------------------------------------------

🚀 How to Use

1. Install Dependencies

Make sure you have Python installed. Then install all required packages:

    pip install -r requirements.txt

Alternatively, refer to environment instructions provided in individual notebooks if needed.

2. Prepare Data

Place your raw data and metadata inside the data/ directory.  
Update the data paths in the notebooks if your folder structure is different.

3. Run Notebooks

Open any notebook in:

- main_figures/ to reproduce key results and publication-quality plots  
- Notebooks/supplementary/ for additional context and extended analyses

📌 Tip: Run all cells sequentially for accurate outputs.

--------------------------------------------------------------------------------

🧩 Key Modules

Module                       | Description
--------------------------- | ------------------------------------------------------------
Graph_builder.py            | Constructs cell–ECM graphs from spatial and molecular data.
Permutation_test.py         | Performs permutation-based statistical testing of interaction frequencies.
CellECMGraphs_multiple.py   | Classes/utilities for managing multiple CEGs.
Helper_functions.py         | Utilities for data processing and visualization.
SimData_Generator.py        | Tools for generating synthetic cell–ECM interaction datasets.

--------------------------------------------------------------------------------

📄 Citation

If you use this repository in your work, please cite:

[Your Paper Title]  
Journal/Preprint, Year

--------------------------------------------------------------------------------

📬 Contact

For questions, issues, or contributions:

- Open an Issue: https://github.com/your-repo/issues
- Contact: [your-email@example.com]

© [Year] Your Name / Lab. All rights reserved.
