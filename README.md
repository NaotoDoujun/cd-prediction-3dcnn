# 3D CNN Aerodynamic Drag (Cd) Prediction Pipeline

This project provides a complete end-to-end deep learning pipeline to predict the Aerodynamic Drag Coefficient ($C_d$ value) of vehicle geometries directly from 3D CAD data (STL files) using a 3D Convolutional Neural Network (3D CNN) in PyTorch.

### Installation

```bash
pip install -r requirements.txt
```

### Directory Structure

```text
cd-prediction-3dcnn/
├── data/
│   ├── processed_voxels/   # Cached voxel matrices (.npy)
│   └── raw_stl/            # Input CAD geometry files (.stl)
├── models/
│   └── cd_predictor_3dcnn.pth # Saved model weights
├── outputs/                # Generated visualization plots
│   ├── loss_history.png    # Training loss curve profile
│   └── prediction_accuracy.png # Predicted vs. True Cd value scatter plot
├── src/
│   ├── model.py            # 3D CNN architecture
│   ├── predict.py          # Inference script for aerodynamic drag
│   ├── preprocess.py       # STL alignment & voxelization logic
│   └── train.py            # Training loop with Apple Silicon (MPS) & CUDA 
├── dataset_meta.csv        # Metadata mapping filenames to true Cd values
├── make_dummy_stl.py       # Helper script to generate a primitive 3D box for pipeline preparation
├── test_run.py             # 64³ pipeline verification script with auto-downsampled terminal preview
├── requirements.txt
└── README.md
```

### 1. Metadata Configuration (dataset_meta.csv)
Before running the training pipeline, prepare a dataset_meta.csv file in the root directory. This file maps your STL filenames to their respective ground-truth $C_d$ values (obtained from wind tunnel tests or CFD simulations).

#### Format Example

```csv
filename,cd_value
test_sphere.stl,0.47
car_model_v1.stl,0.32
```

### 2. Pipeline Components

#### A. 3D Geometry Preprocessor
A data processing program that loads 3D CAD models and transforms them into standardized inputs optimized for deep learning.
- Description: Loads STL files using trimesh, performs rigid alignments (anchoring the lowest vehicle point to Z=0 for ground-clearance effects and centering the Y-axis for lateral symmetry), scales the geometry, and embeds it into a fixed-size cubic voxel grid.
- Key Features: Automates complex spatial transformations to ensure consistent orientation and positioning across various CAD exports.

#### Run
```bash
PYTHONPATH=. python src/preprocess.py
```

#### B. 3D Convolutional Neural Network
A neural network architecture engineered specifically for capturing spatial and volumetric characteristics of 3D objects.
- Description: Implements a deep 3D CNN (CdValuePredictor3D) featuring 4 sequential layers of 3D convolutions, batch normalization, and max pooling, culminating in a linear regression head.
- Key Features: Features robust spatial feature extraction, native handling of raw 3D voxels, and built-in dropout layers to effectively mitigate overfitting.

#### Run
```bash
PYTHONPATH=. python src/model.py
```

#### C. Production Training Pipeline
The core execution script that coordinates data loading, hardware acceleration, and optimization loops.
- Description: Orchestrates custom PyTorch DataLoaders to map file entries in `dataset_meta.csv` with raw STL files. Computes Mean Squared Error (MSE) to minimize variance against true $C_d$ values.
- Key Features: Features on-the-fly caching of multi-resolution `.npy` voxel matrices to eliminate redundant processing and natively leverages Apple Silicon GPU power via Metal Performance Shaders (MPS).

#### Pipeline Resolution Control
The architecture follows a single-source-of-truth configuration. You can globally scale the entire execution pipeline (including preprocessing, model architecture channels, and tensor shapes) by modifying the master constant at the top of `src/train.py`:

```python
# =========================================================
# MASTER CONTROL CONSTANT
# Change this single variable to scale between 64 and 128 pipeline
# =========================================================
GLOBAL_RESOLUTION = 64  # Toggle between 64 and 128 depending on your compute targets
```

#### Run
Once your target resolution is set, initiate the training sequence:
```bash
# Execute the training loop using the selected global resolution
PYTHONPATH=. python src/train.py
```

#### D. Production Inference Pipeline (Prediction)
An evaluation script to predict the aerodynamic drag coefficient of a brand-new, unseen CAD geometry using the trained model weights.
- Description: Takes an unlabelled STL file, passes it through the exact same alignment and voxelization pipeline, loads the trained `.pth` weight dictionary, and outputs the final predicted continuous $C_d$ value.
- Key Features: Pure inference setup (utilizing `torch.no_grad()`) optimized for fast evaluation during iterative vehicle design phases. It supports dynamic resolution scaling to instantly switch between benchmark environments.

#### Run
You can execute inference by targeting your new STL file. By default, the script processes at `64³` resolution using the standard weights:
```bash
# Predict using the default 64³ resolution setup
PYTHONPATH=. python ./src/predict.py --stl data/raw_stl/new_design_test.stl
```

To run inference using a high-fidelity 128³ trained pipeline, simply append the --res 128 argument. The script will automatically load the corresponding cd_predictor_3dcnn_128.pth weights and upscale the preprocessing container:
```bash
# Predict using the high-fidelity 128³ resolution setup
PYTHONPATH=. python ./src/predict.py --stl data/raw_stl/new_design_test.stl --res 128
```

## Dataset Scaling & Training Roadmap
Below is the recommended configuration pipeline based on the number of available 3D CAD variants in your dataset:

| Dataset Size (STL Count) | Recommended Epochs | Target Batch Size | Expected Convergence & Behavior | Core Strategy & Focus |
| :--- | :--- | :--- | :--- | :--- |
| **1 ~ 10** <br>*(Sandbox / Test)* | 5 ~ 10 | 1 ~ 4 | High risk of negative predictions or complete overfitting to a single geometry profile. | Pipeline validation only. Verify that preprocessing alignments and GPU (MPS) tensors run without errors. |
| **50 ~ 100** <br>*(Early Prototype)* | 50 | 4 ~ 8 | Loss begins to decrease steadily. The model captures coarse feature differences (e.g., Sedan vs. SUV). | Establish a baseline validation score. Focus on stabilizing the variance using early dropout constraints. |
| **300 ~ 500** <br>*(Production Ready)* | 100 ~ 200 | 8_/_16 | Smooth loss convergence. The network accurately maps fine geometric changes to localized $C_d$ changes. | Hyperparameter tuning. Optimize learning rates and test deep spatial features for precise aerodynamic tracking. |

### Pipeline Verification (Quick Test)
A standalone test script (`test_run.py`) that generates a dummy geometric shape to validate the end-to-end preprocessing, grid alignment, and voxelization logic.
#### Run
To verify the pipeline health and render a visual 2D cross-section directly inside your terminal window.
```bash
PYTHONPATH=. python test_run.py
```
