import os
import re
import pandas as pd
import numpy as np
# from sklearn.model_validation import train_test_split

def load_simulation_data(path='simulation_files/'):

    files = [i for i in os.listdir(path) if i.endswith('.txt') and i.startswith('output_hydro')]

    #extract parameters from filenames
    df = pd.DataFrame(columns=['param1', 'param2', 'param3', 'param4', 'x', 't', 'vx', 'rho'])
    param_list = []
    df_final = []
    for file in files:
        # Pattern: output_hydro_test_param1_X.XX_param2_X.XX_param3_X.XXXXX_param4_X.XXXXX_deltatX.txt
        match = re.search(r'param1_([\d.]+)_param2_([\d.]+)_param3_([\d.]+)_param4_([\d.]+)_deltat(\d+)', file)
        if match:
            param_list.append({
                'param1': match.group(1),
                'param2': match.group(2),
                'param3': match.group(3),
                'param4': match.group(4),
            })

        if not os.path.isfile(os.path.join('simulation_files/', file)):
            continue
        elif not os.path.isfile(os.path.join(f'simulation_files/output_rho_test_param1_{match.group(1)}_param2_{match.group(2)}_param3_{match.group(3)}_param4_{match.group(4)}_deltat{match.group(5)}.txt')):
            continue
        df_vx = pd.read_csv(os.path.join('simulation_files/', file), sep=",", header=None)
        df_rho = pd.read_csv(os.path.join(f'simulation_files/output_rho_test_param1_{match.group(1)}_param2_{match.group(2)}_param3_{match.group(3)}_param4_{match.group(4)}_deltat{match.group(5)}.txt'), sep=",", header=None, names=['t', 'rho'])
        l = []
        for i in range(1, df_vx.shape[1]):
            df_ = df_vx.iloc[:, [0, i]].copy()
            df_.columns = ['t', 'vx']
            df_.loc[:,'x'] = i - 1
            df_.loc[:,'rho'] = df_rho['rho']
            df_.loc[:,'param1'] = float(match.group(1))
            df_.loc[:,'param2'] = float(match.group(2))
            df_.loc[:,'param3'] = float(match.group(3))
            df_.loc[:,'param4'] = float(match.group(4))
            l.append(df_)
        df = pd.concat(l, axis=0, ignore_index=True)
        df = df[['param1', 'param2', 'param3', 'param4', 'x', 't', 'vx', 'rho']]
        df_final.append(df)

    return pd.concat(df_final, axis=0, ignore_index=True)

def load_data_feather(file_path='simulation_files/simulation_data.feather'):
    df = pd.read_feather(file_path).astype('float32')
    return df

def downsample_data(df, fraction=0.1, random_state=42):
    df_sampled = df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
    return df_sampled

def split_data(df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def get_closer_parameters(df, param1, param2, param3, param4, t_max=None, x_max=None):
    """
    Find the closest parameter set to the target parameters and return filtered data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing simulation data
    param1, param2, param3, param4 : float
        Target parameter values
    t_max : float, optional
        Maximum time value to filter data. If None, includes all time values.
    x_max : float, optional
        Maximum x value to filter data. If None, includes all x values.
    
    Returns:
    --------
    tuple: (filtered_data, closest_params)
        - filtered_data: pd.DataFrame with data from closest parameter set
        - closest_params: pd.Series with the closest parameter values found
    """
    # Find unique parameter combinations
    unique_params = df[['param1', 'param2', 'param3', 'param4']].drop_duplicates()
    
    # Calculate Euclidean distance to target params
    distances = np.sqrt(
        (unique_params['param1'] - param1)**2 + 
        (unique_params['param2'] - param2)**2 + 
        (unique_params['param3'] - param3)**2 + 
        (unique_params['param4'] - param4)**2
    )
    
    # Find closest parameter set
    closest_idx = distances.idxmin()
    closest_params = unique_params.loc[closest_idx]
    
    # Filter dataset by closest params
    filtered_data = df[
        (df['param1'] == closest_params['param1']) &
        (df['param2'] == closest_params['param2']) &
        (df['param3'] == closest_params['param3']) &
        (df['param4'] == closest_params['param4'])
    ].copy()
    
    # Apply t_max filter if provided
    if t_max is not None:
        filtered_data = filtered_data[filtered_data['t'] <= t_max]
    
    # Apply x_max filter if provided
    if x_max is not None:
        filtered_data = filtered_data[filtered_data['x'] <= x_max]
    
    return filtered_data.reset_index(drop=True), closest_params

def get_range_parameters(df, param1_range=None, param2_range=None, param3_range=None, param4_range=None, 
                         t_max=None, x_max=None):
    """
    Return data within specified parameter ranges.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing simulation data
    param1_range, param2_range, param3_range, param4_range : tuple, optional
        (min, max) tuples for each parameter. If None, includes all values for that parameter.
    t_max : float, optional
        Maximum time value to filter data. If None, includes all time values.
    x_max : float, optional
        Maximum x value to filter data. If None, includes all x values.
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataset within the specified parameter ranges
    """
    filtered_data = df.copy()
    
    # Filter by parameter ranges
    if param1_range is not None:
        filtered_data = filtered_data[
            (filtered_data['param1'] >= param1_range[0]) & 
            (filtered_data['param1'] <= param1_range[1])
        ]
    
    if param2_range is not None:
        filtered_data = filtered_data[
            (filtered_data['param2'] >= param2_range[0]) & 
            (filtered_data['param2'] <= param2_range[1])
        ]
    
    if param3_range is not None:
        filtered_data = filtered_data[
            (filtered_data['param3'] >= param3_range[0]) & 
            (filtered_data['param3'] <= param3_range[1])
        ]
    
    if param4_range is not None:
        filtered_data = filtered_data[
            (filtered_data['param4'] >= param4_range[0]) & 
            (filtered_data['param4'] <= param4_range[1])
        ]
    
    # Filter by t_max if provided
    if t_max is not None:
        filtered_data = filtered_data[filtered_data['t'] <= t_max]
    
    # Filter by x_max if provided
    if x_max is not None:
        filtered_data = filtered_data[filtered_data['x'] <= x_max]
    
    return filtered_data.reset_index(drop=True)