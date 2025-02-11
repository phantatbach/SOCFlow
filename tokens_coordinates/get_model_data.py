import pandas as pd
import os
import hdbscan

def get_coordinates(token, model, input_folder):
    """
    Extract the coordinates of the given token from the t-SNE output file generated by SemasioFlow.
    
    Parameters
    ----------
    token : str
        The token of interest
    model : str
        The model name
    input_folder : str
        The folder where the input files are located
    
    Returns
    -------
    None
    """
    
    input_file = os.path.join(input_folder, f'{token}.tsne.30.tsv')
    df = pd.read_csv(input_file, sep='\t')

    # Get the coordinates for each token
    df['x'] = pd.to_numeric(df[model + '.x'], errors='coerce')  # Convert to numeric and round to 4 decimals
    df['y'] = pd.to_numeric(df[model + '.y'], errors='coerce')
    
    df = df[['_id', 'x', 'y']]
    output_file = os.path.join(input_folder, 'visualisation', f'{token}-{model}.tsne.30.tsv')
    df.to_csv(output_file, sep='\t', index=False)
    print(f'The coordinates of {token}-{model} were extracted')

def apply_HDBSCAN(token, model, input_folder, min_cluster_size, min_samples):
    """
    Applies HDBSCAN clustering to the distance matrix of a given model and adds cluster information to the t-SNE output file.

    Parameters
    ----------
    token : str
        The token of interest
    model : str
        The model name
    input_folder : str
        The folder where the input files are located
    min_cluster_size : int
        The minimum number of samples required to form a dense region
    min_samples : int
        The number of samples in a neighbourhood for a point to be considered as a core point

    Returns
    -------
    None
    """
    distance_file = os.path.join(input_folder, f'{model}.ttmx.dist.csv')
    distance_df = pd.read_csv(distance_file, sep=',', index_col=0, header=0)
    distance_matrix = distance_df.to_numpy()

    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed', cluster_selection_method='leaf')
    clusterer.fit(distance_matrix)

    # Get cluster information
    cluster_labels = pd.Series(clusterer.labels_, index=distance_df.index, name='cluster')
    membership_prob = pd.Series(clusterer.probabilities_, index=distance_df.index, name='membprob')

    # Load t-SNE file
    tsne_file_path = os.path.join(input_folder, 'visualisation', f'{token}-{model}.tsne.30.tsv')
    tsne_df = pd.read_csv(tsne_file_path, sep='\t')

    # Filter for tokens present in both t-SNE and distance matrix
    available_tokens = tsne_df['_id'].isin(distance_df.index)
    filtered_tsne_df = tsne_df[available_tokens]

    # Merge cluster information with tSNE file for visualisation
    filtered_tsne_df = filtered_tsne_df.merge(cluster_labels, how='left', left_on='_id', right_index=True)
    filtered_tsne_df = filtered_tsne_df.merge(membership_prob, how='left', left_on='_id', right_index=True)

    # Save the updated t-SNE file
    filtered_tsne_df.to_csv(tsne_file_path, sep='\t', index=False)
    print(f'Updated t-SNE file saved to {tsne_file_path}')

    # Save a version without noise points (cluster != -1)
    tsne_file_path_no_noise = os.path.join(input_folder, 'visualisation', f'no_noise-{token}-{model}.tsne.30.tsv')
    tsne_df_no_noise = filtered_tsne_df[filtered_tsne_df['cluster'] != -1]  # Filter out noise points
    tsne_df_no_noise.to_csv(tsne_file_path_no_noise, sep='\t', index=False)
    print(f'No-noise t-SNE file saved to {tsne_file_path_no_noise}')

def get_HDBSCAN_semcloud(token, model, input_folder):
    # Load t-SNE file
    """
    Integrates HDBSCAN clustering results from a semcloud file into a t-SNE output file.

    Parameters
    ----------
    token : str
        The token of interest.
    model : str
        The model name.
    input_folder : str
        The folder where the input files are located.

    Returns
    -------
    None

    Notes
    -----
    - The function reads a t-SNE file and an HDBSCAN semcloud file, filters tokens present in both,
      and merges cluster and membership probability data.
    - The updated t-SNE file is saved, along with a version excluding noise points (cluster != 0).
    """

    tsne_file_path = os.path.join(input_folder, 'visualisation', f'{token}-{model}.tsne.30.tsv')
    tsne_df = pd.read_csv(tsne_file_path, sep='\t')

    # Load HDBSCAN file
    HDBSCAN_semcloud_file_path = os.path.join(input_folder, f'{model}_HDBSCAN_semcloud.tsv')
    HDBSCAN_semcloud_df = pd.read_csv(HDBSCAN_semcloud_file_path, sep='\t')

    # Filter for tokens present in both t-SNE and HDBSCAN matrices
    available_tokens = tsne_df['_id'].isin(HDBSCAN_semcloud_df['_id'])
    filtered_tsne_df = tsne_df[available_tokens]

    # Merge t-SNE data with cluster and membprob columns from HDBSCAN data
    filtered_tsne_df = filtered_tsne_df.merge(
        HDBSCAN_semcloud_df[['_id', 'cluster', 'membprob']],
        on='_id',
        how='inner'
    )

    # Save the updated t-SNE file
    filtered_tsne_df.to_csv(tsne_file_path, sep='\t', index=False)
    print(f'Updated t-SNE file saved to {tsne_file_path}')

    # Save a version without noise points (cluster != 0)
    tsne_file_path_no_noise = os.path.join(input_folder, 'visualisation', f'no_noise-{token}-{model}.tsne.30.tsv')
    tsne_df_no_noise = filtered_tsne_df[filtered_tsne_df['cluster'] != 0]  # Filter out noise points
    tsne_df_no_noise.to_csv(tsne_file_path_no_noise, sep='\t', index=False)
    print(f'No-noise t-SNE file saved to {tsne_file_path_no_noise}')


def get_other_info(token, model, input_folder):
    """
    Add other information of the tokens to the t-SNE output file generated by SemasioFlow.

    Parameters
    ----------
    token : str
        The token of interest
    model : str
        The model name
    input_folder : str
        The folder where the input files are located

    Returns
    -------
    None
    """

    # Construct file paths
    tsne_file_path = os.path.join(input_folder, 'visualisation', f'{token}-{model}.tsne.30.tsv')
    variables_file_path = os.path.join(input_folder, f'{token}.variables.tsv')
    tsne_file_path_no_noise = os.path.join(input_folder, 'visualisation', f'no_noise-{token}-{model}.tsne.30.tsv')

    # Ensure input files exist
    if not os.path.exists(tsne_file_path):
        raise FileNotFoundError(f"t-SNE file not found: {tsne_file_path}")
    if not os.path.exists(variables_file_path):
        raise FileNotFoundError(f"Variables file not found: {variables_file_path}")
    if not os.path.exists(tsne_file_path_no_noise):
        raise FileNotFoundError(f"t-SNE file not found: {tsne_file_path_no_noise}")
    
    # Read data
    tsne_df = pd.read_csv(tsne_file_path, sep='\t')
    variables_df = pd.read_csv(variables_file_path, sep='\t')
    tsne_df_no_noise = pd.read_csv(tsne_file_path_no_noise, sep='\t')

    # Filter out columns that start with '_'
    info_columns = [col for col in variables_df.columns if not col.startswith('_')]

    # Ensure there are columns to map
    if not info_columns:
        print(f"No additional columns found in {variables_file_path} to map.")
        return

    # Set index and merge additional information
    tsne_df = tsne_df.merge(variables_df[['_id'] + info_columns], on='_id', how='left')
    tsne_df_no_noise = tsne_df_no_noise.merge(variables_df[['_id'] + info_columns], on='_id', how='left')

    # Save the updated t-SNE file
    tsne_df.to_csv(tsne_file_path, sep='\t', index=False)
    tsne_df_no_noise.to_csv(tsne_file_path_no_noise, sep='\t', index=False)
    print(f'The additional information for {token}-{model} was added successfully.')

def get_model_data(token, model, input_folder, min_cluster_size=None, min_samples=None, source=None):
    """
    Extracts model data by obtaining coordinates, clustering information, and additional data.

    Depending on the source parameter, the function either uses available HDBSCAN results from 

    semcloud or computes HDBSCAN clustering with specified parameters.

    Parameters
    ----------
    token : str
        The token of interest.
    model : str
        The model name.
    input_folder : str
        The folder where the input files are located.
    min_cluster_size : int, optional
        The minimum number of samples required to form a dense region (used if source is not 'semcloud').
    min_samples : int, optional
        The number of samples in a neighbourhood for a point to be considered as a core point 
        (used if source is not 'semcloud').
    source : str, optional
        Determines whether to use existing HDBSCAN results from semcloud or to compute new clustering.

    Returns
    -------
    None
    """
    if source == 'semcloud':
        get_coordinates(token, model, input_folder)
        get_HDBSCAN_semcloud(token, model, input_folder)
        get_other_info(token, model, input_folder)
    else:
        get_coordinates(token, model, input_folder)
        apply_HDBSCAN(token, model, input_folder, min_cluster_size, min_samples)
        get_other_info(token, model, input_folder)
    print('The model data was extracted')