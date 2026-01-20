# PINN Laser ablations

A PyTorch-based implementation of Physics-Informed Neural Networks for learning and predicting hydrodynamic flows. This project combines data-driven neural networks with physical constraints to accurately model complex fluid dynamics.

## Project Overview

This project implements a PINN that learns to approximate solutions to hydrodynamic equations from simulation data. The network is trained using both data constraints (loss from simulated data) and physics constraints (residuals of the governing differential equations), allowing it to make physically-plausible predictions.

### Key Applications
- Velocity profile prediction (vx, vy components)
- Stress tensor prediction (sxx, sxy, syy)
- Density/height (rho/h) field recovery
- Multi-parameter flow modeling

## Project Structure

```
.
├── README.md                          # This file
├── model.py                           # PINN architecture and trainer
├── train.py                           # Main training script
├── dataload.py                        # Data loading and preprocessing utilities
├── boundary.py                        # Boundary condition setup
├── plot.py                            # Visualization and plotting functions
├── simulations_failure.py             # Analysis of failure cases
├── simulations_plot.py                # Simulation plotting utilities
├── simulation_files/                  # Processed simulation data
│   └── simulation_data.feather        # Pre-processed data (fast loading format)
├── simulation_files_raw/              # Raw simulation output files
│   ├── output_hydro_test_param*.txt   # Velocity data for different parameters
│   └── output_rho_test_param*.txt     # Density data for different parameters
├── plots/                             # Output directory for generated plots
├── .git/                              # Git version control
├── .gitignore                         # Git ignore file
└── __pycache__/                       # Python cache directory

```

## File Descriptions

### Core Training Files

**[model.py](model.py)**
- `PINN`: Neural network architecture with physics loss computation
  - Inputs: Parameters (param1-4), spatial coordinate (x), and time (t) - 7 features total
  - Outputs: Velocity component (vx), vertical velocity (vy), stress tensor (sxx, sxy, syy), and density/height (h) - 6 outputs
  - Activation: Tanh (smooth activation for PINNs)
  - Physics constraints: Residuals of hydrodynamic equations computed via automatic differentiation
- `PINNTrainer`: Training loop manager
  - Adam optimizer with configurable learning rate and weight decay
  - Learning rate scheduling with ReduceLROnPlateau
  - Automatic differentiation for physics loss computation

**[train.py](train.py)**
- Main training script for the PINN model
- Configurable hyperparameters:
  - Network depth and hidden dimensions
  - Data/physics loss weighting (lambda_data, lambda_physics)
  - Boundary and physics sampling regions and point counts
  - Learning rate and weight decay
- Two-phase training: initial phase + optional refinement phase
- Periodic checkpoint visualization during training
- Loads pre-processed simulation data from `simulation_data.feather`

### Data Management

**[dataload.py](dataload.py)**
- `load_simulation_data()`: Reads raw simulation files from `simulation_files/`
  - Extracts parameter values from filenames
  - Combines velocity (vx) and density (rho) data
  - Returns structured pandas DataFrame
- `load_data_feather()`: Fast loading of pre-processed data from `.feather` format
- `get_closer_parameters()`: Filters dataset by parameter values
- `get_range_parameters()`: Selects data within parameter ranges
- `split_data()`: Train/test splitting utilities

**[boundary.py](boundary.py)**
- `compute_h0()`: Computes initial density profile based on spatial coordinates
  - Implements smooth tanh-based transition functions
  - Supports parameterized initial conditions
- `boudary()`: Generates boundary condition data
  - Creates random sampling points in inner and outer regions
  - Returns both spatial coordinates and boundary values

### Visualization

**[plot.py](plot.py)**
- `plot_profiles()`: Visualizes simulation data
  - Density vs time
  - Velocity (vx) vs time at multiple spatial locations
  - Velocity (vx) vs space at multiple times
- `plot_pinn_profiles()`: Overlays PINN predictions on data plots
  - Compares model predictions with ground truth
  - Shows model convergence and accuracy
- `plot_learning_curve()`: Plots training loss history
  - Data loss, physics loss, and total loss
  - Logarithmic scale for better visualization
  - Auto-creates output directories

### Analysis & Utilities

**[simulations_failure.py](simulations_failure.py)**
- Analysis tools for examining parameter sensitivity
- Generates heatmaps of physical quantities (rho) across parameter combinations
- Identifies regions of different physical behavior
- Helps visualize how parameters affect simulation outcomes

**[simulations_plot.py](simulations_plot.py)**
- Comprehensive plotting and analysis of simulation results
- Parameter sweep visualization
- Finds closest parameter combinations to query points
- Spatial interpolation and visualization of results

## Usage

### Quick Start: Train a PINN Model

```python
# Run the training script
python train.py
```

The script will:
1. Load simulation data from `simulation_data.feather`
2. Initialize a PINN model with configurable architecture
3. Generate boundary conditions based on parameter values
4. Train for a specified number of epochs
5. Save plots to `plots/` directory at regular intervals

### Configuration

Edit parameters in [train.py](train.py):

```python
# Dataset parameters - which parameter set to train on
param1, param2, param3, param4 = 1.0, 1.0, 1.0, 0.0
x_max, t_max = 1.0, 1.0

# Training hyperparameters
lambda_data = 1.0           # Weight for data loss
lambda_physics = 0.01       # Weight for physics loss (lower = more data-driven)
learning_rate = 1e-4        # Adam learning rate
weight_decay = 1e-6         # L2 regularization
n_epochs = 30000            # Number of training epochs

# Boundary condition sampling (initial conditions)
boundary_x_min = 0.4        # Inner region x extent
boundary_y_min = 1.2        # Inner region y extent
boundary_n_min = 1000       # Number of points in inner region
boundary_x_max = 1.2        # Outer region x extent
boundary_y_max = 1.2        # Outer region y extent
boundary_n_max = 1000       # Number of points in outer region

# Physics collocation points (for residual computation)
phys_x_min, phys_y_min = 0.4, 1.2
phys_n_min = 1000           # Collocation points in inner region
phys_x_max, phys_y_max = 1.2, 1.2
phys_n_max = 1000           # Collocation points in outer region
```

## Data Format

### Simulation Data
- **Format**: Feather (binary columnar format for fast I/O)
- **Location**: `simulation_data.feather`
- **Columns**:
  - `param1`, `param2`, `param3`, `param4`: Physical parameters
  - `x`, `y`: Spatial coordinates
  - `t`: Time
  - `vx`, `vy`: Velocity components
  - `rho`: Density field

### Raw Simulation Files
- Located in `simulation_files/`
- Naming convention: `output_hydro_test_param1_X_param2_X_param3_X_param4_X_deltaTX.txt`
- Each file contains velocity data in CSV format
- Corresponding density file: `output_rho_test_param1_X_...txt`

## Model Architecture

The PINN consists of:
1. **Input layer**: 7 features (param1, param2, param3, param4, x, y, t)
2. **Hidden layers**: Configurable depth (default 4 layers, 50-256 units each)
3. **Activation**: Tanh between layers (smooth, periodic-friendly activation)
4. **Output layer**: 6 predictions (vx, vy, sxx, sxy, syy, h)

### Loss Components
- **Data Loss**: MSE between predictions and simulation data at known points
- **Physics Loss**: Residuals of hydrodynamic equations at collocation points (computed via automatic differentiation)
- **Total Loss**: `lambda_data * data_loss + lambda_physics * physics_loss`

The physics loss enforces conservation laws and governing equations on the entire domain, not just data points.

## Training Details

### Training Strategy
- **Single-phase or multi-phase**: Standard training with combined data + physics constraints
- **Loss weighting**: Balance between fitting data and satisfying physical equations
- **Adaptive learning**: Learning rate adjustment based on data loss convergence

### Optimization
- **Optimizer**: AdamW with configurable learning rate (default 1e-4)
- **Learning Rate Scheduler**: ReduceLROnPlateau (reduces LR when loss plateaus)
- **Weight Decay**: L2 regularization for network weights (default 1e-6)
- **Physics computation**: Automatic differentiation for computing derivatives in physics loss

## Output Files

Training generates plots in the `plots/` directory:
- `velocity_profiles.png`: Ground truth simulation data
- `profiles.png`: PINN predictions vs data
- `learning_curve.png`: Training loss history