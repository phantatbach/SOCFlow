import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import plotly.graph_objects as go
import os

class SOCAnalyser:
    def __init__(self):
        self.assoc_total = None
        self.sense = None

    def get_context(self, token, sense, token_list, input_folder, output_folder):
        """
        Extracts the context for the given token and sense from the variables.tsv file.

        Parameters:
            token (str): The token for which to extract the context.
            sense (str): The sense for which to extract the context.
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
            print(f"No matching records found for token '{token}' and sense '{sense}'.")
            return

        # Select only the _id and _ctxt.raw columns
        context_df = context_df[['_id', '_ctxt.raw']]

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Write the resulting DataFrame to a CSV file
        output_file = os.path.join(output_folder, f'{token}-{sense}_context.csv')
        context_df.to_csv(output_file, sep=',', index=False)
        print(f"Context for {token}-{sense} extracted and saved to {output_file}.")

    def elbow_finder(self, sense, sub_senSOC_file, mode):
        """
        Finds the elbow point in the association scores for a given sense.

        This method calculates the sum and average of association scores from a specified submatrix
        and uses the Kneedle algorithm to determine the elbow point, which indicates a natural cutoff
        point in the data. It visualizes the curve with the detected elbow point.

        Parameters:
            sense (str): The sense for which to find the elbow point.
            sub_senSOC_file (str): The file containing the submatrix of association scores.
            mode (str): The mode to use for analysis. Options are 'sum' for total scores and 'avg'
                        for average scores.

        Returns:
            None
        """
        SOC_df = pd.read_csv(sub_senSOC_file, header=0, index_col=0)

        # Calculate the sum and average of all the association scores by rows in the dataframe
        self.assoc_total = SOC_df.sum(axis=0).sort_values(ascending=False)
        self.assoc_avg = SOC_df.mean(axis=0).sort_values(ascending=False)
        self.sense = sense

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

    def SOC_dist_vis(self, n, sub_senSOC, output_folder, mode):
        """
        Visualizes the association scores of the top N SOCs in a radial graph.

        Parameters:
            n (int): The number of top SOCs to visualize.
            sub_senSOC (str): The submatrix of the sense for which to visualize.
            output_folder (str): The folder where the visualization is saved.
            mode (str): The mode to use for analysis. Options are 'sum' for total scores and 'avg'
                        for average scores.

        Returns:
            None
        """

        if mode == 'sum':
            if self.assoc_total is None or self.sense is None:
                raise ValueError("You must run elbow_finder first to set assoc_total and sense.")
            # Get a DataFrame of top N SOCs
            top_n = self.assoc_total.head(n).to_frame()
        
        elif mode == 'avg':
            if self.assoc_avg is None or self.sense is None:
                raise ValueError("You must run elbow_finder first to set assoc_avg and sense.")
            # Get a DataFrame of top N SOCs
            top_n = self.assoc_avg.head(n).to_frame()

        top_n_df = top_n.reset_index()
        top_n_df.columns = ['SOC', 'Association Score']

        # Save to CSV
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f'top_{n}-{sub_senSOC}-{mode}_SOCs.csv')
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
            text=[self.sense],
            textposition='top center',
            name='Sense',
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

def get_POS_submtx(SenSOC_file, sub_SenSOC, POS_list, input_folder, sep=','):
    """
    Extracts a submatrix from a SOC matrix containing only the columns that match the specified POS tags.

    Args:
        SenSOC_file (str): Path to the SOC matrix file to be filtered.
        sub_SenSOC (str): Name for the output submatrix file.
        POS_list (list of str): List of Part-Of-Speech tags to filter the columns by.
        input_folder (str): Directory where the output file will be saved.
        sep (str): Separator used in the CSV file. Defaults to ','.
    
    Raises:
        FileNotFoundError: If the specified SOC matrix file does not exist.
        ValueError: If no columns match the specified POS tags.

    Returns:
        None: The function saves the filtered submatrix to a CSV file in the specified directory.
    """
    # Ensure input file exists
    if not os.path.exists(SenSOC_file):
        raise FileNotFoundError(f"Input file not found: {SenSOC_file}")
    
    # Load the SOC matrix
    SenSOC_matrix = pd.read_csv(SenSOC_file, sep=sep, header=0)

    # If POS_list is empty, log a warning and return the original matrix
    if not POS_list:
        print("Warning: POS_list is empty. No filtering applied.")
        return

    # Filter columns based on POS tags
    column_names = [
        column for POS in POS_list for column in SenSOC_matrix.columns if POS in column
    ]

    # Ensure at least one column matches
    if not column_names:
        raise ValueError(f"No columns found matching POS tags: {POS_list}")
    
    output_file = os.path.join(input_folder, f'{sub_SenSOC}_SOCs.csv')

    # Save the filtered submatrix
    SenSOC_matrix[column_names].to_csv(output_file, sep=',')
    print(f"Filtered SOC matrix saved to: {output_file}")
