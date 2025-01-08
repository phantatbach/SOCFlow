import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import plotly.graph_objects as go

class SOCAnalyser:
    def __init__(self):
        self.assoc_total = None
        self.sense = None

    def get_context(self, token, sense, token_list):
        context_df = pd.read_csv(f'./input/{token}.variables.tsv', header=0, sep='\t')
        # Filter the DataFrame to include only rows where _id is in token_list
        context_df = context_df[context_df['_id'].isin(token_list)]

        # Select only the _id and _ctxt.raw columns
        context_df = context_df[['_id', '_ctxt.raw']]

        # Assign the column names
        context_df.columns = ['_id', '_ctxt.raw']

        # Display or process the resulting DataFrame
        context_df.to_csv(f'./output/{token}-{sense}_context.csv', sep=',', index=False)
        print(f'Context for {token}-{sense} extracted.')

    def elbow_finder(self, sense):
        """
        Identifies the elbow point in the distribution of association scores for SOCs
        and saves the association totals and sense in the class.

        Args:
            sense (str): Name of the sense, used to locate the corresponding CSV file.
        """
        # Read the data
        soc_df = pd.read_csv(f'./input/{sense}_SOCs.csv', header=0, index_col=0)

        # Sum all the association scores by rows in the dataframe
        self.assoc_total = soc_df.sum(axis=0).sort_values(ascending=False)
        self.sense = sense

        # Extract x and y values for the curve
        x = np.arange(len(self.assoc_total))
        y = self.assoc_total.values

        # Use the Kneedle algorithm to find the elbow
        knee = KneeLocator(x, y, curve="convex", direction="decreasing")

        # Print the elbow point
        if knee.knee is not None:
            print(f"Elbow index: {knee.knee + 1}")
            print(f"Elbow SOC: {self.assoc_total.index[knee.knee]} (Association Score: {self.assoc_total.values[knee.knee]})")
        else:
            print("No elbow point detected.")

        # Visualize the curve with the elbow point
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label='Association Scores', color='blue')
        if knee.knee is not None:
            plt.axvline(x=knee.knee, color='red', linestyle='--', label=f'Elbow (Index: {knee.knee})')
            plt.scatter(knee.knee, y[knee.knee], color='red', s=100, label='Elbow Point')
        plt.title('Elbow Point', fontsize=16)
        plt.xlabel('Sorted SOCs', fontsize=14)
        plt.ylabel('Association Score', fontsize=14)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def soc_dist_vis(self, n):
        """
        Visualizes the top N SOCs in a radial plot with association scores determining the radial distance.

        Args:
            n (int): Number of top SOCs to visualize.
        """
        if self.assoc_total is None or self.sense is None:
            raise ValueError("You must run `elbow_finder` first to set `assoc_total` and `sense`.")

        # Get a DataFrame of top N SOCs
        top_n = self.assoc_total.head(n).to_frame()
        top_n_df = top_n.reset_index()
        top_n_df.columns = ['SOC', 'Association Score']

        # Save to CSV
        top_n_df.to_csv(f'./output/top_{n}-{self.sense}_SOCs.csv', sep=',', index=False)
        print(f'Top {n} SOCs extracted.')

        # Calculate angles for even distribution of SOCs
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        # Calculate x and y coordinates for each SOC
        top_n_df['x'] = top_n_df['Association Score'] * np.cos(angles)
        top_n_df['y'] = top_n_df['Association Score'] * np.sin(angles)

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
            name='Sense'
        ))

        # Add the SOCs
        fig.add_trace(go.Scatter(
            x=top_n_df['x'],
            y=top_n_df['y'],
            mode='markers+text',
            marker=dict(size=10, color='blue'),
            text=top_n_df['SOC'],
            textposition='top center',
            name='SOCs'
        ))

        # Add concentric reference circles
        max_radius = top_n_df['Association Score'].max()
        circle_steps = 5  # Number of circles
        for i in range(1, circle_steps + 1):
            radius = max_radius * (i / circle_steps)
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

