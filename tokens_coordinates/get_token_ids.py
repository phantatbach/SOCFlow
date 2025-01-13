# Running the App:
# Save the code to a Python file (get_tokens_ids.py).
# Run the file: python get_tokens_ids.py.
# Open the app in your browser (http://127.0.0.1:8050).

import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Output, Input, State
import os

def load_data(token, model, input_folder):
    """
    Load the data for a given token and model from the specified input folder.
    
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
    df : pandas.DataFrame
        A data frame with the token coordinates and senses
    """
    # Load the data
    input_file = os.path.join(input_folder, f'{token}-{model}.tsne.30.tsv')
    df = pd.read_csv(input_file, sep='\t', header=0, names=['_id', 'x', 'y', 'senses'])
    
    # Ensure x and y are numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['x', 'y'])  # Remove rows with invalid x or y
    return df

def create_app(token, model, input_folder):
    """
    Create a Dash app that displays an interactive scatter plot of the tokens and their senses in a given model.
    
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
    app : dash.Dash
        The Dash app
    """
    df = load_data(token, model, input_folder)
    def create_figure():
        """
        Create a scatter plot of the tokens and their senses in a given model.

        Parameters
        ----------
        df : pandas.DataFrame
            A data frame with the token coordinates and senses

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The scatter plot
        """
        # Create scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='senses',
            title=f'Interactive Token Visualization of {token}-{model}',
            labels={'x': 'X-Axis', 'y': 'Y-Axis'},
            hover_name='_id'
        )

        # Layout of the plot
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x"),
            width=1000,
            height=1000
        )
        return fig
    
    fig = create_figure()

    # Initialize the interactive Dash app
    app = dash.Dash(__name__)

    # Layout of the app
    app.layout = html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=fig,
            config={'scrollZoom': True},  # Allow zooming with the mouse
            style={'display': 'inline-block', 'width': '1000px', 'height': '1000px'}  # Container style
        ),
        html.Div(
            id='selected-tokens', 
            style={'margin-top': '20px', 'font-size': '16px'}
        ),
        html.Button('Clear Selection', id='clear-button', n_clicks=0, style={'margin-top': '20px'})  # Add button here
    ])

    # Define callback to handle box selection
    @app.callback(
        Output('selected-tokens', 'children'),
        Input('scatter-plot', 'selectedData'),
        prevent_initial_call=True
    )

    # Display the IDs of the selected tokens
    def display_selected_tokens(selectedData):
        if selectedData is None:
            return "No tokens selected."
        # Extract the token IDs from the selected points and add single quotes
        selected_tokens = [f"'{point['hovertext']}'" for point in selectedData['points']]
        return f"Selected Tokens: {', '.join(selected_tokens)}"

    # Define callback to clear the selection box completely
    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('clear-button', 'n_clicks'),
        State('scatter-plot', 'figure'),
        prevent_initial_call=True
    )
    # Reset the graph to completely clear the selection box
    def clear_selection(n_clicks, current_figure):
        return create_figure()
    
    return app

# Run the app
def get_token_ids(token, model, input_folder):
    """
    Creates a Dash app that displays the coordinates of tokens of a model in an interactive scatter plot
    and prints the IDs of the selected tokens.

    Parameters
    ----------
    token : str
        The token to investigate
    model : str
        The name of the model of interest
    input_folder : str
        The folder where the coordinate files are stored

    Returns
    -------
    None
    """

    app = create_app(token, model, input_folder)
    app.run_server(debug=True)
    print('Open the app in your browser (http://127.0.0.1:8050).')