# Running the App:
# Save the code to a Python file (get_tokens_ids.py).
# Run the file: python get_tokens_ids.py.
# Open the app in your browser (http://127.0.0.1:8050).

import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Output, Input, State

def load_data(token, model):
    # Load the data
    data = f'{token}-{model}.tsne.30.tsv'
    df = pd.read_csv(data, sep='\t', header=0, names=['_id', 'x', 'y', 'senses'])
    
    # Ensure x and y are numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['x', 'y'])  # Remove rows with invalid x or y
    return df

def create_app(token, model):
    df = load_data(token, model)
    def create_figure():
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
        # Extract the token IDs from the selected points
        selected_tokens = [point['hovertext'] for point in selectedData['points']]
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
def get_token_ids(token, model):
    app = create_app(token, model)
    app.run_server(debug=True)