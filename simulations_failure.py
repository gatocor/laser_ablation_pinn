import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import dataload

dataset = dataload.load_data_feather()

# Get unique param3 and param4 values
param3_values = sorted(dataset['param3'].unique())
param4_values = sorted(dataset['param4'].unique())

for p4 in param4_values:
    # Filter data for this param4 value
    data_p4 = dataset[dataset['param4'] == p4]
    
    # Create subplots grid for param3 values
    n_plots = len(param3_values)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, p3 in enumerate(param3_values):
        subset = data_p4[data_p4['param3'] == p3]
        
        # Create pivot table for heatmap
        pivot_data = subset.pivot_table(values='rho', index='param2', columns='param1', aggfunc='max')
        
        sns.heatmap(pivot_data, ax=axes[idx], cmap='viridis', cbar_kws={'label': 'rho'}, vmin=0, vmax=1)
        axes[idx].set_title(f'param3={p3:.5f}')
        axes[idx].set_xlabel('param1')
        axes[idx].set_ylabel('param2')
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Heatmaps for param4={p4:.5f}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/2D_heatmaps_param4_{p4:.5f}.png', dpi=300)
    plt.close()
