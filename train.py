import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import random

from model import PINN, PINNTrainer
from dataload import load_data_feather, get_closer_parameters, get_range_parameters, split_data
from boundary import boudary
from plot import plot_pinn_profiles, plot_learning_curve

# Dataset
param1 = 1.0
param2 = 1.0
param3 = 1.0
param4 = 0.0
x_max = 1.0
t_max = 1.0

# Data settings
test_fraction = 0.2

# Boundary settings
boundary_x_min = 0.4
boundary_y_min = 1.2
boundary_n_min = int(1e4)
boundary_x_max = 1.2
boundary_y_max = 1.2
boundary_n_max = int(1e4)

# Physics data settings
phys_x_min = boundary_x_min
phys_y_min = boundary_y_min
phys_n_min = int(1e4)

phys_x_max = boundary_x_max
phys_y_max = boundary_y_max
phys_n_max = int(1e4)
phys_t_max = 1.0

# Training settings
lambda_data = 1.0
lambda_physics = 0.1

learning_rate = 1e-4
weight_decay = 1e-6

batch_size_boundary = 1024
batch_size_physics = 2048

n_epochs = 70000
n_epochs_refine = 2000

# Progress settings
plot_interval = 100

if __name__ == "__main__":

    # Load and prepare data
    print("Loading and preparing data...")
    dataset = load_data_feather()
    dataset, parameters = get_closer_parameters(dataset, param1, param2, param3, param4, t_max=t_max, x_max=x_max)
    print(f"Parameters used for training: param1={parameters.iloc[0]}, param2={parameters.iloc[1]}, param3={parameters.iloc[2]}, param4={parameters.iloc[3]}")
    param1 = parameters.iloc[0]
    param2 = parameters.iloc[1]
    param3 = parameters.iloc[2]
    param4 = parameters.iloc[3]

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = PINN(n_layers=4, hidden_dim=256).to(device)
    print(f"Model created with {model.n_layers} hidden layers")
    
    # Create trainer
    trainer = PINNTrainer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    print("Trainer initialized")

    # Data
    # TODO

    # Boundary data
    X_boundary, y_boundary = boudary(param1, param2, param3, param4, x_max=boundary_x_max, y_max=boundary_y_max, npoints_max=boundary_n_max, x_min=boundary_x_min, y_min=boundary_y_min, npoints_min=boundary_n_min)
    
    X_boundary_tensor = torch.tensor(X_boundary, dtype=torch.float32)
    y_boundary_tensor = torch.tensor(y_boundary, dtype=torch.float32)
    
    # Create DataLoader for boundary data
    boundary_dataset = TensorDataset(X_boundary_tensor, y_boundary_tensor)
    boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size_boundary, shuffle=True)
    print(f"Boundary data prepared: {len(boundary_dataset)} samples, batch size {batch_size_boundary}")

    # Sampling data
    X_phys_min = np.random.rand(phys_n_min, 3)
    X_phys_min[:,0] = X_phys_min[:,0]*2*phys_x_min - phys_x_min  # x
    X_phys_min[:,1] = X_phys_min[:,1]*2*phys_y_min - phys_y_min  # y
    X_phys_min[:,2] = X_phys_min[:,2]*phys_t_max                 # t
    X_phys_max = np.random.rand(phys_n_max, 3)
    X_phys_max[:,0] = X_phys_max[:,0]*2*phys_x_max - phys_x_max  # x
    X_phys_max[:,1] = X_phys_max[:,1]*2*phys_y_max - phys_y_max  # y
    X_phys_max[:,2] = X_phys_max[:,2]*phys_t_max                 # t
    X_phys = np.vstack([X_phys_min, X_phys_max])
    X_param1 = np.full((X_phys.shape[0], 1), param1)
    X_param2 = np.full((X_phys.shape[0], 1), param2)
    X_param3 = np.full((X_phys.shape[0], 1), param3)
    X_param4 = np.full((X_phys.shape[0], 1), param4)
    X_phys = np.hstack([X_param1, X_param2, X_param3, X_param4, X_phys])

    X_phys_tensor = torch.tensor(X_phys, dtype=torch.float32)
    
    # Create DataLoader for physics data
    physics_dataset = TensorDataset(X_phys_tensor)
    physics_loader = DataLoader(physics_dataset, batch_size=batch_size_physics, shuffle=True)
    print(f"Physics data prepared: {len(physics_dataset)} samples, batch size {batch_size_physics}")

    # Training loop    
    print("Starting training...")
    l_data = []
    l_phys = []
    l_total = []
    for epoch in range(n_epochs+n_epochs_refine):        
        # Training phase
        if epoch < n_epochs:
            refine = False
        else:
            refine = True
        
        # Batch training
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        # Iterate through batches
        for (X_boundary_batch, y_boundary_batch), (X_phys_batch,) in zip(boundary_loader, physics_loader):
            # Move batches to device
            X_boundary_device = X_boundary_batch.to(device)
            y_boundary_device = y_boundary_batch.to(device)
            X_phys_device = X_phys_batch.to(device)
            
            # Training step on batch
            losses = trainer.train_step(
                X_boundary_device, y_boundary_device,
                X_phys_device,
                lambda_data=lambda_data,
                lambda_physics=lambda_physics,
                refine=refine
            )
            
            # Accumulate losses
            epoch_data_loss += losses['data_loss']
            epoch_phys_loss += losses['physics_loss']
            epoch_total_loss += losses['total_loss']
            num_batches += 1
        
        # Average losses over batches
        avg_data_loss = epoch_data_loss / num_batches
        avg_phys_loss = epoch_phys_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        
        l_data.append(avg_data_loss)
        l_phys.append(avg_phys_loss)
        l_total.append(avg_total_loss)
    
        if epoch < n_epochs:
            trainer.scheduler.step(avg_data_loss)

        if (epoch + 1) % plot_interval == 0:
            if refine:
                print(
                    f"Epoch {epoch+1}/{n_epochs+n_epochs_refine} (Refine) - "
                    f"Data Loss: {avg_data_loss:.6f}, "
                    f"Physics Loss: {avg_phys_loss:.6f}, "
                    f"Total Loss: {avg_total_loss:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{n_epochs+n_epochs_refine} - "
                    f"Data Loss: {avg_data_loss:.6f}, "
                    f"Physics Loss: {avg_phys_loss:.6f}, "
                    f"Total Loss: {avg_total_loss:.6f}"
                )

            plot_pinn_profiles(dataset, model, param1, param2, param3, param4, save_path="plots/profiles.png")
            plt.close()

            plot_learning_curve(
                l_data, l_phys, l_total,
                save_path="plots/learning_curve.png"
            )
                
    print("\nTraining complete!")
