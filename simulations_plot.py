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

x_max = 10.0
t_max = 1.0

# Find unique parameter combinations
unique_params = dataset[['param1', 'param2', 'param3', 'param4']].drop_duplicates()

# Calculate distance to target params
distances = np.sqrt(
    (unique_params['param1'] - param1)**2 + 
    (unique_params['param2'] - param2)**2 + 
    (unique_params['param3'] - param3)**2 + 
    (unique_params['param4'] - param4)**2
)

# Find closest parameter set
closest_idx = distances.idxmin()
closest_params = unique_params.loc[closest_idx]

print(f"Target params: param1={param1}, param2={param2}, param3={param3}, param4={param4}")
print(f"Closest params: param1={closest_params['param1']:.5f}, param2={closest_params['param2']:.5f}, param3={closest_params['param3']:.5f}, param4={closest_params['param4']:.5f}")

# Filter dataset by closest params
filtered_data = dataset[
    (dataset['param1'] == closest_params['param1']) &
    (dataset['param2'] == closest_params['param2']) &
    (dataset['param3'] == closest_params['param3']) &
    (dataset['param4'] == closest_params['param4'])
].sort_values('t')

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. Plot rho vs t
ax1 = plt.subplot(2, 2, 1)
data_main1 = filtered_data[filtered_data['t'] <= t_max]
sns.lineplot(data=data_main1, x='t', y='rho', ax=ax1)
ax1.set_xlim(0, t_max)
ax1.set_xlabel('t')
ax1.set_ylabel('rho')
ax1.set_title(f'rho vs t (zoomed to t_max={t_max})')
ax1.grid(True, alpha=0.3)

# Add inset for full range context
axins1 = inset_axes(ax1, width="50%", height="50%", loc='lower right')
sns.lineplot(data=filtered_data, x='t', y='rho', ax=axins1, color='C0')
axins1.set_xlabel('t', fontsize=8)
axins1.set_ylabel('rho', fontsize=8)
axins1.tick_params(labelsize=7)
axins1.grid(True, alpha=0.3)
axins1.set_title('Full range', fontsize=8)

# 2. Heatmap of vx vs x and t
ax2 = plt.subplot(2, 2, 2)

# Create uniform grid for zoomed region
data_main2 = filtered_data[(filtered_data['t'] <= t_max) & (filtered_data['x'] <= x_max)]
t_main = np.linspace(0, t_max, 500)
x_main = np.linspace(0, x_max, 50)
T_main, X_main = np.meshgrid(t_main, x_main)
points_main = data_main2[['t', 'x']].values
values_main = data_main2['vx'].values
vx_main = griddata(points_main, values_main, (T_main, X_main), method='linear')

# Plot zoomed region in main plot
im = ax2.imshow(vx_main, aspect='auto', cmap='viridis', origin='lower',
                extent=[0, t_max, 0, x_max])
ax2.set_xlim(0, t_max)
ax2.set_ylim(0, x_max)
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title(f'vx heatmap (zoomed to x_max={x_max}, t_max={t_max})')
plt.colorbar(im, ax=ax2, label='vx')

# Add inset for full range context
axins2 = inset_axes(ax2, width="60%", height="60%", loc='upper right')
axins2.patch.set_facecolor('white')
axins2.patch.set_alpha(0.9)
t_unique = np.sort(filtered_data['t'].unique())
x_unique = np.sort(filtered_data['x'].unique())
t_uniform = np.linspace(t_unique.min(), t_unique.max(), 200)
x_uniform = np.linspace(x_unique.min(), x_unique.max(), len(x_unique))
T_uniform, X_uniform = np.meshgrid(t_uniform, x_uniform)
points = filtered_data[['t', 'x']].values
values = filtered_data['vx'].values
vx_uniform = griddata(points, values, (T_uniform, X_uniform), method='linear')
im_zoom = axins2.imshow(vx_uniform, aspect='auto', cmap='viridis', origin='lower',
                        extent=[t_uniform.min(), t_uniform.max(), x_uniform.min(), x_uniform.max()])
axins2.tick_params(labelsize=7, colors='white')
for spine in axins2.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(2)
axins2.set_title('Full range', fontsize=8, color='white')

# 3. Line plot vx vs t
ax3 = plt.subplot(2, 2, 3)
filtered_sample_x = filtered_data[filtered_data['x'].isin(sorted(filtered_data['x'].unique())[::5])]
data_main3 = filtered_sample_x[(filtered_sample_x['t'] <= t_max) & (filtered_sample_x['x'] <= x_max)]
sns.lineplot(data=data_main3, x='t', y='vx', hue='x', ax=ax3, alpha=0.7, linewidth=1, legend='auto' if len(sorted(filtered_data['x'].unique())) < 20 else False)
ax3.set_xlim(0, t_max)
ax3.set_xlabel('t')
ax3.set_ylabel('vx')
ax3.set_title(f'vx vs t (zoomed to x_max={x_max}, t_max={t_max})')
ax3.grid(True, alpha=0.3)
if len(sorted(filtered_data['x'].unique())) < 20 and ax3.get_legend():
    ax3.legend(fontsize=8, ncol=2, title='x')

# Add inset for full range context
axins3 = inset_axes(ax3, width="60%", height="60%", loc='upper right')
sns.lineplot(data=filtered_sample_x, x='t', y='vx', hue='x', ax=axins3, alpha=0.7, linewidth=1, legend=False)
axins3.set_xlabel('t', fontsize=8)
axins3.set_ylabel('vx', fontsize=8)
axins3.tick_params(labelsize=7)
axins3.grid(True, alpha=0.3)
axins3.set_title('Full range', fontsize=8)

# 4. Line plot vx vs x
ax4 = plt.subplot(2, 2, 4)
filtered_sample_t = filtered_data[filtered_data['t'].isin(sorted(filtered_data['t'].unique())[::5])]
data_main4 = filtered_sample_t[(filtered_sample_t['t'] <= t_max) & (filtered_sample_t['x'] <= x_max)]
sns.lineplot(data=data_main4, x='x', y='vx', hue='t', ax=ax4, alpha=0.7, linewidth=1, legend='auto' if len(sorted(filtered_data['t'].unique())) < 20 else False)
ax4.set_xlim(0, x_max)
ax4.set_xlabel('x')
ax4.set_ylabel('vx')
ax4.set_title(f'vx vs x (zoomed to x_max={x_max}, t_max={t_max})')
ax4.grid(True, alpha=0.3)
if len(sorted(filtered_data['t'].unique())) < 20 and ax4.get_legend():
    ax4.legend(fontsize=8, ncol=2, title='t')

# Add inset for full range context
axins4 = inset_axes(ax4, width="60%", height="60%", loc='upper right')
sns.lineplot(data=filtered_sample_t, x='x', y='vx', hue='t', ax=axins4, alpha=0.7, linewidth=1, legend=False)
axins4.set_xlabel('x', fontsize=8)
axins4.set_ylabel('vx', fontsize=8)
axins4.tick_params(labelsize=7)
axins4.grid(True, alpha=0.3)
axins4.set_title('Full range', fontsize=8)

plt.suptitle(f'Analysis for param1={closest_params["param1"]:.5f}, param2={closest_params["param2"]:.5f}, param3={closest_params["param3"]:.5f}, param4={closest_params["param4"]:.5f}', 
             fontsize=14)
plt.tight_layout()
plt.savefig(f'plots/simulation_analysis_param1_{closest_params["param1"]:.5f}_param2_{closest_params["param2"]:.5f}_param3_{closest_params["param3"]:.5f}_param4_{closest_params["param4"]:.5f}.png', dpi=300)
plt.show()

