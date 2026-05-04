# aerodynamic-coefficient-prediction
CSCI 6366 Neural Networks &amp; Deep Learning Project: NN Surrogate Models for Aerodynamic Coefficient Prediction
## Overview

Airfoil design optimization requires evaluating lift (C_L) and drag (C_D) coefficients across thousands of geometry and operating condition combinations. XFOIL, the standard panel-method solver, takes approximately 100ms per evaluation, making large-scale design sweeps slow. This project trains neural network surrogate models that predict C_L and C_D from airfoil geometry, angle of attack (AoA), and Reynolds number (Re) faster than using simulation software.

Three model architectures are trained and compared:
- **MLP Baseline**: flat input of raw coordinates + AoA + Re
- **Improved 1D CNN**: physics-based input (thickness and camber distributions) with FiLM conditioning
- **1D CNN + Self-Attention**: same as CNN with multi-head self-attention over the chord dimension

## How to Run

All notebooks are designed for Google Colab with Google Drive for storage.

1. Run `data_preprocessing_nb1.ipynb` to build dataset
2. Run `get_geometry_files.ipynb` to extract geometric features 
3. Train models using `model_training_nb2.ipynb`
4. Run `evaluation_analysis_nb3.ipynb` to evaluate models and generate figures

Each notebook mounts Google Drive and reads/writes from `MyDrive/airfoil_surrogate/`. XFOIL must be installed in the Colab environment (`apt-get install xfoil xvfb`).

## Dependencies

- Python 3.10+
- PyTorch
- NumPy, Pandas, SciPy
- scikit-learn
- Matplotlib, Seaborn
- BeautifulSoup4, Requests
- XFOIL (system package via `apt-get install xfoil xvfb`)

## Notebook Descriptions

### `get_geometry_files.ipynb`
Computes geometric features from raw airfoil coordinate arrays and saves them as `X_geom_{train/val/test}.npy`. For each airfoil, the notebook splits the contour into upper and lower surfaces, interpolates both onto a common 100-point chord-fraction grid, and extracts:
- **Thickness distribution** t(x) — upper surface minus lower surface at each chord station
- **Camber distribution** c(x) — mean of upper and lower surfaces at each chord station
- **Five scalar descriptors:** maximum thickness-to-chord ratio, maximum camber, camber location, cross-sectional area, and trailing edge angle

These features are used as input to the CNN models instead of raw coordinates. Run this notebook once after `data_preprocessing_nb1.ipynb` and before `model_training_nb2.ipynb`.

### `data_preprocessing_nb1.ipynb`
Builds the labeled dataset from scratch through the following steps:
1. Scrapes airfoil coordinate files from the UIUC Airfoil Database
2. Parses and resamples each geometry to 200 arc-length-uniform points
3. Runs XFOIL over a grid of operating conditions to generate C_L and C_D labels
4. Filters to the attached-flow design envelope (AoA values in [−2°, 10°]) 
5. Splits data at the **airfoil level** 70/15/15 train/val/test
6. Saves arrays to Google Drive: `X_{train/val/test}.npy`, `y_{train/val/test}.npy`, `airfoil_dataset_final.pkl`, `dataset_stats.json`

### `model_training_nb2.ipynb`
Trains all three models and saves checkpoints. Key design choices:
- **MLP Baseline:** 402-dim flat input (raw coordinates + AoA + Re), 4 hidden layers with LayerNorm and GELU activation, 636K parameters
- **Improved 1D CNN:** thickness and camber distributions (2×100) + 5 shape scalars + AoA + Re; residual blocks with dilations [1, 2, 4] for multi-scale feature extraction; FiLM conditioning injects AoA and Re by scaling and shifting the geometry embedding; 1,076K parameters
- **CNN + Self-Attention:** same as CNN with multi-head self-attention over the chord dimension to capture long-range geometric dependencies; 1,340K parameters
- **Ensemble:** simple average of CNN and CNN+Attention predictions

Training uses AdamW with a warmup LR scheduler, early stopping with patience 30, and batch size 256. All models are evaluated on the same held-out test set.

**Outputs saved to `models/`:** `mlp_baseline_best.pt`, `cnn1d_best.pt`, `cnn1d_attention_best.pt`, `scalers.pkl`, `envelope.json`

---

### `evaluation_analysis_nb3.ipynb`
Loads trained models and generates all evaluation figures and metrics. Includes:
- **In-distribution test metrics:** RMSE, MAE, MAPE (filtered to |C_L| > 0.1), R^2 for all models
- **Parity plots:** predicted vs. true C_L and C_D across all test observations (macroscopic view)
- **Aerodynamic polar curves:** predicted vs. XFOIL C_L and C_D across AoA for three held-out airfoils (microscopic view)
- **OOD degradation analysis:** error comparison inside vs. outside the training envelope, showing C_L error degrades ~6× outside the envelope for the CNN
- **Statistical significance testing:** cluster bootstrap resampling (10,000 iterations) at the airfoil level — the correct approach given within-airfoil correlation
- **Inference time benchmark:** wall-clock comparison of CNN vs. XFOIL
- **Physical constraint check:** fraction of predictions with C_D < 0 (non-physical)

## Folder Descriptions

### `dataset/`
Contains dataset files produced by `data_preprocessing_nb1.ipynb` and `get_geometry_files.ipynb`:
- `X_{train/val/test}.npy` — raw input arrays (402-dim: coordinates + AoA + log Re)
- `y_{train/val/test}.npy` — label arrays (C_L, C_D)
- `X_geom_{train/val/test}.npy` — physics-based geometry feature arrays (207-dim: thickness + camber distributions + scalars + AoA + log Re)
- `names_{train/val/test}.npy` — airfoil name strings per sample
- `airfoil_dataset_final.pkl` — full labeled DataFrame with split assignments
- `dataset_stats.json` — dataset summary statistics

### `models/`
Contains trained model checkpoints and artifacts produced by `model_training_nb2.ipynb`:
- `mlp_baseline_best.pt` — MLP baseline weights
- `cnn1d_best.pt` — Improved 1D CNN weights
- `cnn1d_attention_best.pt` — CNN + Self-Attention weights
- `scalers.pkl` — fitted StandardScaler objects for X and y normalization
- `envelope.json` — design envelope bounds used for OOD detection

### `evaluation_outputs/`
Contains all figures and results tables produced by `evaluation_analysis_nb3.ipynb`:
- `parity_plots.png` — predicted vs. true C_L and C_D for all models
- `ood_degradation_by_aoa.png` — error comparison inside vs. outside training envelope
- `polar_curves.png` — aerodynamic polars for three held-out airfoils
- `error_breakdown.png` — MAE broken down by AoA and Reynolds number
- `inference_time.png` — speedup comparison vs. XFOIL
- `training_curves.png` — training and validation loss curves
- `paper_table1.csv` — test set metrics table


## Data Source

Airfoil coordinate files are sourced from the **UIUC Airfoil Coordinates Database**:

> Selig, M. S., *UIUC Airfoil Coordinates Database*, University of Illinois at Urbana-Champaign, Applied Aerodynamics Group.  
> https://m-selig.ae.illinois.edu/ads/coord_database.html




