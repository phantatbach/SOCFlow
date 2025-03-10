import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import plotly.graph_objects as go
import os

class SOCAnalyser:
    def __init__(self):
        self.assoc_total = None
        self.region = None

    def get_context(self, token, region, token_list, input_folder, output_folder):
        """
        Extracts the context for the given token and region from the variables.tsv file.

        Parameters:
            token (str): The token for which to extract the context.
            region (str): The region for which to extract the context.
            token_list (list): List of token IDs to filter the context by.
            input_folder (str): Input folder containing the variables.tsv file.
            output_folder (str): Output folder to write the extracted context to.

        Returns:
            None
        """
        input_file = os.path.join(input_folder, f'{token}.variables.tsv')

        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            return

        # Ensure token_list is a list
        if not isinstance(token_list, list):
            raise TypeError("token_list must be a list of token IDs.")

        # Read the input file
        context_df = pd.read_csv(input_file, header=0, sep='\t')

        # Filter the DataFrame to include only rows where _id is in token_list
        context_df = context_df[context_df['_id'].isin(token_list)]

        # Handle empty DataFrame
        if context_df.empty:
            print(f"No matching records found for token '{token}' and region '{region}'.")
            return

        # Select only the _id and _ctxt.raw columns
        context_df = context_df[['_id', '_ctxt.raw']]

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Write the resulting DataFrame to a CSV file
        output_file = os.path.join(output_folder, f'{token}-{region}_context.csv')
        context_df.to_csv(output_file, sep=',', index=False)
        print(f"Context for {token}-{region} extracted and saved to {output_file}.")

    def elbow_finder(self, region, sub_regSOC_file, mode):
        """
        Finds the elbow point in the association scores for a given region.

        This method calculates the sum and average of association scores from a specified submatrix
        and uses the Kneedle algorithm to determine the elbow point, which indicates a natural cutoff
        point in the data. It visualizes the curve with the detected elbow point.

        Parameters:
            region (str): The region for which to find the elbow point.
            sub_regSOC_file (str): The file containing the submatrix of association scores.
            mode (str): The mode to use for analysis. Options are 'sum' for total scores and 'avg'
                        for average scores.

        Returns:
            None
        """
        SOC_df = pd.read_csv(sub_regSOC_file, header=0, index_col=0)

        # Calculate the sum and average of all the association scores by rows in the dataframe
        self.assoc_total = SOC_df.sum(axis=0).sort_values(ascending=False)
        self.assoc_avg = SOC_df.mean(axis=0).sort_values(ascending=False)
        self.region = region

        if mode == 'sum':
            # Extract x and y values for the curve
            x = np.arange(len(self.assoc_total))
            y = self.assoc_total.values

        elif mode == 'avg':
            # Divide the sum by the number of SOCs
            x = np.arange(len(self.assoc_total))
            y = self.assoc_avg.values         

        # Use the Kneedle algorithm to find the elbow
        knee = KneeLocator(x, y, curve="convex", direction="decreasing")

        # Print the elbow point
        if knee.knee is not None:
            print(f"Elbow index: {knee.knee + 1}")
            if mode == 'sum':
                print(f"Elbow SOC: {self.assoc_total.index[knee.knee]} (Association Score: {self.assoc_total.values[knee.knee]})")
            elif mode == 'avg':
                print(f"Elbow SOC: {self.assoc_avg.index[knee.knee]} (Association Score: {self.assoc_avg.values[knee.knee]})")
        else:
            print("No elbow point detected.")

        # Visualize the curve with the elbow point
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label='Association Scores', color='blue')
        if knee.knee is not None:
            plt.axvline(x=knee.knee, color='red', linestyle='--', label=f'Elbow (Index: {knee.knee + 1})')
            plt.scatter(knee.knee, y[knee.knee], color='red', s=100, label='Elbow Point')
        plt.title('Elbow Point', fontsize=16)
        plt.xlabel('Sorted SOCs', fontsize=14)
        plt.ylabel('Association Score', fontsize=14)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def SOC_dist_vis(self, n, sub_regSOC, output_folder, mode, POS_name=None):
        """
        Visualizes the association scores of the top N SOCs in a radial graph.

        Parameters:
            n (int): The number of top SOCs to visualize.
            sub_regSOC (str): The submatrix of the region for which to visualize.
            output_folder (str): The folder where the visualization is saved.
            mode (str): The mode to use for analysis. Options are 'sum' for total scores and 'avg'
                        for average scores.

        Returns:
            None
        """

        if mode == 'sum':
            if self.assoc_total is None or self.region is None:
                raise ValueError("You must run elbow_finder first to set assoc_total and region.")
            # Get a DataFrame of top N SOCs
            top_n = self.assoc_total.head(n).to_frame()
        
        elif mode == 'avg':
            if self.assoc_avg is None or self.region is None:
                raise ValueError("You must run elbow_finder first to set assoc_avg and region.")
            # Get a DataFrame of top N SOCs
            top_n = self.assoc_avg.head(n).to_frame()

        top_n_df = top_n.reset_index()
        top_n_df.columns = ['SOC', 'Association Score']

        # Save to CSV
        os.makedirs(output_folder, exist_ok=True)
        if POS_name is not None:
            output_file = os.path.join(output_folder, f'{sub_regSOC}-{mode}-top_{n}-{POS_name}_SOCs.csv')
        else:
            output_file = os.path.join(output_folder, f'{sub_regSOC}-{mode}-top_{n}_SOCs.csv')
        top_n_df.to_csv(output_file, sep=',', index=False)
        print(f'Top {n} SOCs extracted and saved to: {output_file}')

        # Calculate angles for even distribution of SOCs
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        # Calculate inverse distances for plotting (higher scores = closer to the center)
        max_score = top_n_df['Association Score'].max()
        top_n_df['distance'] = max_score / top_n_df['Association Score']  # Inverse proportional distance

        # Calculate x and y coordinates based on inverse distance
        top_n_df['x'] = top_n_df['distance'] * np.cos(angles)
        top_n_df['y'] = top_n_df['distance'] * np.sin(angles)

        # Create a Plotly scatter plot
        fig = go.Figure()

        # Add the central token
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='markers+text',
            marker=dict(size=20, color='red'),
            text=[self.region],
            textposition='top center',
            name='region',
            hoverinfo='text',  # Disable x and y in hover
        ))

        # Add the SOCs with raw association scores displayed in hover information
        fig.add_trace(go.Scatter(
        x=top_n_df['x'],
        y=top_n_df['y'],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=top_n_df['SOC'],  # SOC names
        customdata=np.stack((top_n_df['Association Score'], top_n_df['distance']), axis=-1),  # Raw score and inverse distance
        hovertemplate=(
            '<b>SOC:</b> %{text}<br>'
            '<b>Raw Association Score:</b> %{customdata[0]:.4f}<br>'
            '<b>Distance (Inverse):</b> %{customdata[1]:.4f}'
        ),
        name='SOCs'
        ))

        # Add concentric reference circles
        max_distance = max_score / top_n_df['Association Score'].min()  # Furthest point (lowest score)
        circle_steps = 5  # Number of circles
        for i in range(1, circle_steps + 1):
            radius = max_distance * (i / circle_steps)
            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=x_circle,
                y=y_circle,
                mode='lines',
                line=dict(color='gray', dash='dot'),
                hoverinfo='skip',
                showlegend=False
            ))

        # Customize layout
        fig.update_layout(
            title='Radial Graph of SOCs and Association Scores',
            showlegend=False,
            template='plotly_white',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                scaleanchor="x"
            ),
            width=1000,
            height=1000,
            dragmode='pan',
            hovermode='closest',
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Show the plot
        fig.show()

def get_POS_submtx(RegSOC_file, sub_regSOC, POS_list, output_folder, POS_name=None, sep=','):
    """
    Filters the input SOC matrix to only include columns that match the POS tags provided.

    Parameters:
        RegSOC_file (str): The input file path to the SOC matrix.
        sub_regSOC (str): The name of the submatrix to generate.
        POS_list (list): A list of POS tags to filter the columns by.
        output_folder (str): The folder to save the filtered submatrix to.
        POS_name (str): Optional. The name of the POS tag to append to the output filename.
        sep (str): Optional. The separator used in the input file. Defaults to ','

    Returns:
        None
    """
    # Ensure the path exists
    if not os.path.exists(RegSOC_file):
        raise FileNotFoundError(f"Input file not found: {RegSOC_file}")
    
    # Load the SOC matrix
    RegSOC_matrix = pd.read_csv(RegSOC_file, sep=sep, header=0)

    # If POS_list is empty, log a warning and return the original matrix
    if not POS_list:
        print("Warning: POS_list is empty. No filtering applied.")
        return

    # Filter columns based on POS tags
    column_names = [
        column for POS in POS_list for column in RegSOC_matrix.columns if POS in column
    ]

    # Ensure at least one column matches
    if not column_names:
        raise ValueError(f"No columns found matching POS tags: {POS_list}")
    
    if POS_name is not None:
        output_file = os.path.join(output_folder, f'{sub_regSOC}_{POS_name}_SOCs.csv')
    else:
        output_file = os.path.join(output_folder, f'{sub_regSOC}_SOCs.csv')

    # Save the filtered submatrix
    RegSOC_matrix[column_names].to_csv(output_file, sep=',')
    print(f"Filtered SOC matrix saved to: {output_file}")
