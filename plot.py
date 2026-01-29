import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from dataload import get_closer_parameters
import os

def plot_profiles(dataset, param1=1.0, param2=1.0, param3=1.0, param4=0.0, 
                           figsize=(18, 6), save_path=None):
    """
    Plot velocity profiles (vx vs t and vx vs x) and rho recovery for a given parameter set.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        The dataset containing simulation data
    param1, param2, param3, param4 : float, optional
        Parameter values to filter. If None, will use the unique set if only one exists.
    figsize : tuple, optional
        Figure size (width, height). Default is (18, 6)
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    fig, (ax1, ax2, ax3) : matplotlib figure and axes
    """
    
    filtered_data, _ = get_closer_parameters(dataset, param1, param2, param3, param4)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Plot rho vs t
    sns.lineplot(data=filtered_data, x='t', y='rho', ax=ax1)
    ax1.set_xlabel('t')
    ax1.set_ylabel('rho')
    ax1.set_title('rho vs t')
    ax1.grid(True, alpha=0.3)
    
    # 2. Line plot vx vs t
    filtered_sample_x = filtered_data[filtered_data['x'].isin(sorted(filtered_data['x'].unique())[::5])]
    sns.lineplot(data=filtered_sample_x, x='t', y='vx', hue='x', ax=ax2, 
                alpha=0.7, linewidth=1, 
                legend='auto' if len(sorted(filtered_data['x'].unique())) < 20 else False)
    ax2.set_xlabel('t')
    ax2.set_ylabel('vx')
    ax2.set_title('vx vs t')
    ax2.grid(True, alpha=0.3)
    if len(sorted(filtered_data['x'].unique())) < 20 and ax2.get_legend():
        ax2.legend(fontsize=8, ncol=2, title='x')
    
    # 3. Line plot vx vs x
    filtered_sample_t = filtered_data[filtered_data['t'].isin(sorted(filtered_data['t'].unique())[::5])]
    sns.lineplot(data=filtered_sample_t, x='x', y='vx', hue='t', ax=ax3, 
                alpha=0.7, linewidth=1, 
                legend='auto' if len(sorted(filtered_data['t'].unique())) < 20 else False)
    ax3.set_xlabel('x')
    ax3.set_ylabel('vx')
    ax3.set_title('vx vs x')
    ax3.grid(True, alpha=0.3)
    if len(sorted(filtered_data['t'].unique())) < 20 and ax3.get_legend():
        ax3.legend(fontsize=8, ncol=2, title='t')
    
    plt.suptitle(f'Analysis for param1={param1:.5f}, param2={param2:.5f}, '
                f'param3={param3:.5f}, param4={param4:.5f}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig, (ax1, ax2, ax3)

def plot_pinn_profiles(dataset, model, param1=1.0, param2=1.0, param3=1.0, param4=0.0, 
                       fig=None, figsize=(18, 6), save_path=None, label='PINN'):
    """
    Plot PINN predictions for velocity profiles and rho recovery.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        The dataset containing simulation data
    model : torch.nn.Module
        The trained PINN model
    param1, param2, param3, param4 : float
        Parameter values to filter
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, creates new figure.
    figsize : tuple, optional
        Figure size (width, height). Default is (18, 6)
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    label : str, optional
        Label for the PINN predictions in the legend. Default is 'PINN'
    
    Returns:
    --------
    fig, (ax1, ax2, ax3) : matplotlib figure and axes
    """
    
    filtered_data, _ = get_closer_parameters(dataset, param1, param2, param3, param4)
    
    # Prepare inputs for model
    device = next(model.parameters()).device
    
    # Add y coordinate if not present (set to 0.0 for 1D simulation)
    if 'y' not in filtered_data.columns:
        filtered_data['y'] = 0.0
    
    # Prepare input tensor in correct order: param1, param2, param3, param4, x, y, t
    X = torch.tensor(filtered_data[['param1', 'param2', 'param3', 'param4', 'x', 'y', 't']].values, 
                     dtype=torch.float32).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X).cpu().numpy()
    
    # Add predictions to dataframe
    pred_data = filtered_data.copy()
    pred_data['vx_pred'] = predictions[:, 0]
    pred_data['vy_pred'] = predictions[:, 1]
    pred_data['sxx_pred'] = predictions[:, 2]
    pred_data['sxy_pred'] = predictions[:, 3]
    pred_data['syy_pred'] = predictions[:, 4]
    pred_data['h_pred'] = predictions[:, 5]
    
    # Create or use existing figure
    if fig is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        create_labels = True
    else:
        axes = fig.get_axes()
        if len(axes) < 3:
            raise ValueError("Provided figure must have at least 3 axes")
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
        create_labels = False
    
    # 1. Plot rho vs t at x=0, y=0 - Data and Predictions
    if create_labels:
        # Filter data at x=0, y=0
        data_at_origin = filtered_data[(filtered_data['x'] == 0.0) & (filtered_data['y'] == 0.0)]
        sns.lineplot(data=data_at_origin, x='t', y='rho', ax=ax1, label='Data', linewidth=2)
    # Filter predictions at x=0, y=0
    pred_at_origin = pred_data[(pred_data['x'] == 0.0) & (pred_data['y'] == 0.0)]
    sns.lineplot(data=pred_at_origin, x='t', y='h_pred', ax=ax1, 
                label=label, linestyle='--', linewidth=2)
    if create_labels:
        ax1.set_xlabel('t')
        ax1.set_ylabel('rho')
        ax1.set_title('rho vs t (at x=0, y=0)')
        ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Line plot vx vs t - Data and Predictions
    filtered_sample_x = pred_data[pred_data['x'].isin(sorted(pred_data['x'].unique())[::5])]
    if create_labels:
        # Plot data first
        filtered_data_sample_x = filtered_data[filtered_data['x'].isin(sorted(filtered_data['x'].unique())[::5])]
        sns.lineplot(data=filtered_data_sample_x, x='t', y='vx', hue='x', ax=ax2, 
                    alpha=0.5, linewidth=1, legend=False)
    # Plot predictions
    for x_val in sorted(filtered_sample_x['x'].unique()):
        x_data = filtered_sample_x[filtered_sample_x['x'] == x_val]
        ax2.plot(x_data['t'], x_data['vx_pred'], linestyle='--', 
                linewidth=1.5, alpha=0.7)
    
    if create_labels:
        ax2.set_xlabel('t')
        ax2.set_ylabel('vx')
        ax2.set_title('vx vs t (solid: data, dashed: PINN)')
        ax2.grid(True, alpha=0.3)
    
    # 3. Line plot vx vs x - Data and Predictions
    filtered_sample_t = pred_data[pred_data['t'].isin(sorted(pred_data['t'].unique())[::5])]
    if create_labels:
        # Plot data first
        filtered_data_sample_t = filtered_data[filtered_data['t'].isin(sorted(filtered_data['t'].unique())[::5])]
        sns.lineplot(data=filtered_data_sample_t, x='x', y='vx', hue='t', ax=ax3, 
                    alpha=0.5, linewidth=1, legend=False)
    # Plot predictions
    for t_val in sorted(filtered_sample_t['t'].unique()):
        t_data = filtered_sample_t[filtered_sample_t['t'] == t_val]
        ax3.plot(t_data['x'], t_data['vx_pred'], linestyle='--', 
                linewidth=1.5, alpha=0.7)
    
    if create_labels:
        ax3.set_xlabel('x')
        ax3.set_ylabel('vx')
        ax3.set_title('vx vs x (solid: data, dashed: PINN)')
        ax3.grid(True, alpha=0.3)
        plt.suptitle(f'PINN Analysis for param1={param1:.5f}, param2={param2:.5f}, '
                    f'param3={param3:.5f}, param4={param4:.5f}', fontsize=14)
        plt.tight_layout()
    
    if save_path:
        # Create parent directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:  # only if a directory part is present
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    
    return fig, (ax1, ax2, ax3)

def plot_learning_curve(data_losses, physics_losses, total_losses, fig=None, figsize=(10, 5), save_path=None, log_scale=True):
    """
    Plot learning curves for data, physics, and total losses.

    Parameters
    ----------
    data_losses : Sequence[float]
        Losses from the data term.
    physics_losses : Sequence[float]
        Losses from the physics term.
    total_losses : Sequence[float]
        Combined losses used for optimization.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. If None, a new figure is created.
    figsize : tuple, optional
        Size of the figure when creating a new one.
    save_path : str, optional
        Path to save the plot. Creates directories if they do not exist.
    log_scale : bool, optional
        Whether to use a logarithmic scale on the y-axis.

    Returns
    -------
    fig, ax : matplotlib figure and axis with the plot.
    """

    if not (len(data_losses) == len(physics_losses) == len(total_losses)):
        raise ValueError("All loss sequences must have the same length")
    if len(total_losses) == 0:
        raise ValueError("Loss sequences must not be empty")

    epochs = np.arange(1, len(total_losses) + 1)
    created_fig = False

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        axes = fig.get_axes()
        ax = axes[0] if axes else fig.add_subplot(111)
        ax.clear()

    ax.plot(epochs, data_losses, label='Data loss', linewidth=1.5)
    ax.plot(epochs, physics_losses, label='Physics loss', linewidth=1.5)
    ax.plot(epochs, total_losses, label='Total loss', linewidth=1.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig, ax

if False:

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import dataload

    dataset = dataload.load_data_feather()

    param1 = 1.0
    param2 = 1.0
    param3 = 1.0
    param4 = 0.0

    plot_profiles(dataset, param1, param2, param3, param4, save_path='plots/test_velocity_profiles.png')
