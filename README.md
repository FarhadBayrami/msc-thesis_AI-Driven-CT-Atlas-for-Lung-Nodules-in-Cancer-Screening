# msc-thesis_AI-Driven-CT-Atlas-for-Lung-Nodules-in-Cancer-Screening
Master's thesis on creating a computed tomography atlas of pulmonary nodules for lung cancer screening, including processing scripts and notebooks.

# A Computed Tomography Atlas of Pulmonary Nodules for Lung Cancer Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains the Master's thesis and associated code for "A Computed Tomography Atlas of Pulmonary Nodules for Lung Cancer Screening" (Academic Year 2024/2025, University of Bologna, Department of Computer Science and Engineering, Artificial Intelligence for Medicine M). 

The thesis develops a CT atlas by registering pulmonary nodule segmentations to a template (e.g., BHI template), generating frequency maps, heatmaps, and multi-view visualizations. It addresses challenges in lung cancer screening through image processing, registration, and analysis using tools like ANTs, SimpleITK, and Python libraries.

**Candidate**: Farhad Bayrami  
**Supervisor**: Prof. Stefano Diciotti  
**Co-Supervisor**: Dr. Giulia Raffaella De Luca  

Key contributions:
- Pipeline for CT image registration and warping of nodule masks.
- Frequency mapping of nodule locations across subjects.
- Visualizations (heatmaps, projections, multi-view overlays) for anatomical insights.
- Quality control and processing scripts for NIfTI files.

## Files and Structure
- `Dissertation.pdf`: Full 43-page thesis document (introduction, methodology, results, discussion).
- **Scripts** (in `scripts/`):
  - `create_visualizations.py`: Generates multi-view (axial, coronal, sagittal) PNG visualizations of warped nodule masks overlaid on a template.
  - `fig5script (1).py`: Processes annotations, segments lungs, and creates location distribution heatmaps and composite views.
  - `heatmap_warped (1).py`: Builds frequency maps from warped masks, visualizes mid-slices, and creates 3-view projections.
  - `fig05_modified_fixed_f (2).py`: Full pipeline for masking with lung atlas, applying Jet colormap, overlaying on BHI template, flipping to RAS orientation, and extracting best anatomical views.
  - `frequencymap (2).py`: Generates absolute frequency maps from warped NIfTI segmentations, with filename sanitization.
  - `antRegistrationApply_Farhad (4).rtf` (or `.sh`): Bash script for ANTs registration and transformation application (rigid, affine, SyN).
- **Notebooks** (in `notebooks/`):
  - `heatmap-warped (2).ipynb`: Interactive notebook for loading frequency maps, normalizing, and creating 3-view heatmaps with colorbars.
  - `phase1__final (5).ipynb`: Phase 1 processing notebook (installs dependencies, handles NIfTI files, generates QC outputs like JSON logs and PNG visualizations).
  - `fig5 (2).ipynb`: Notebook for best slice selection, overlay creation with Jet colormap, and combined 3-view visualizations.
- `requirements.txt`: Python dependencies (see below).
- (Optional) `outputs/`: Generated files (e.g., `frequency_map.nii.gz`, `frequency_heatmap_3view_clean.png`).

## Requirements
Install dependencies via:
pip install -r requirements.txt
Contents of `requirements.txt` (copy this into the file):
numpy
pandas
scipy
SimpleITK
matplotlib
opencv-python-headless
scikit-image


For ANTs registration: Install ANTs (e.g., via Homebrew on Mac: `brew install ants` or from [GitHub](https://github.com/ANTsX/ANTs)). Set `ANTSPATH` in scripts as needed.

For notebooks: Use Jupyter/Colab with GPU if available.

## Usage
1. Clone the repo:

git clone https://github.com/FarhadBayrami/CT-Atlas-Pulmonary-Nodules-Lung-Cancer-Screening.git
cd CT-Atlas-Pulmonary-Nodules-Lung-Cancer-Screening

2. Install requirements (above).
3. Prepare data: Place CT images, masks, and templates (e.g., `BHI_template.nii.gz`) in a folder. Update paths in scripts/notebooks.
4. Run a script (example: visualizations):

python scripts/create_visualizations.py --input_folder path/to/warped_masks --template_path path/to/BHI_template.nii.gz --output_folder outputs

- Use `--help` for arguments.
5. Run notebooks: `jupyter notebook notebooks/heatmap-warped.ipynb` (or open in Colab).
6. For registration: Run the Bash script with fixed/moving images and mask:
- Use `--help` for arguments.
5. Run notebooks: `jupyter notebook notebooks/heatmap-warped.ipynb` (or open in Colab).
6. For registration: Run the Bash script with fixed/moving images and mask:

bash scripts/antRegistrationApply_Farhad.sh path/to/fixed.nii.gz path/to/moving.nii.gz path/to/mask.nii.gz


## Results and Visualizations
- Frequency maps show nodule distribution (e.g., max frequency in central slices).
- Heatmaps: 3-view projections (axial, sagittal, coronal) with 'hot' colormap.
- Overlays: Nodule masks on BHI template, flipped to RAS orientation.
- Best views: Automatically selected slices with highest variance for detailed analysis.

Example outputs (from scripts/notebooks):
- `frequency_absolute_map.nii.gz`: NIfTI frequency map.
- `frequency_heatmap_3view_clean.png`: 3-view heatmap.
- `overlay_best3views_combined.png`: Combined axial/sagittal/coronal overlays.

## License
MIT License - see [LICENSE](LICENSE) for details.

## Contact
Farhad Bayrami - farhad.bayrami@studio.unibo.it

For citations: Refer to the thesis PDF.
