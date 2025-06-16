**CEG: Cell-ECM Graph Analysis**
This repository provides code and notebooks for building, analyzing, and visualizing Cell-ECM Graphs (CEGs), with a focus on cell-extracellular matrix interactions in biological datasets. The project includes graph construction, permutation testing, explainable GNN models, and publication-ready figures.

Repository Structure
Main Notebooks

Supplementary Notebooks


How to Use
Install dependencies:

Prepare data:
Place your raw data and metadata in the data/ directory or update paths in the notebooks.

Run notebooks:
Open any notebook in main_figures or Notebooks/supplementary/ and run cells in order.

Main figures will reproduce key results and plots for publication.
Supplementary notebooks provide additional context and analyses.
Key Modules
Graph_builder.py: Functions for constructing cell-ECM graphs from spatial and molecular data.
Permutation_test.py: Permutation-based statistical testing of interaction frequencies.
CellECMGraphs_multiple.py: Classes and utilities for handling multiple CEGs.
Helper_functions.py: Miscellaneous utilities for data processing and visualization.
SimData_Generator.py: Tools for generating simulated cell-ECM data.
Citation
If you use this code or data, please cite our paper:

[Your Paper Title]
Journal/Preprint, Year

Contact
For questions or contributions, please open an issue or contact [your email].

Note:

All notebooks are designed for reproducibility and publication-quality figures.
See comments in each notebook for figure-specific instructions and data requirements.
