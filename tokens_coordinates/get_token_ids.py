import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Output, Input
import os

def load_data(token, model, input_folder):
    input_file = os.path.join(input_folder, f'{token}-{model}.tsne.30.tsv')
    df = pd.read_csv(input_file, sep='\t')

    # Ensure x and y are numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['x', 'y'])  # Remove rows with invalid x or y
    return df

def create_app(token, model, input_folder):
    df = load_data(token, model, input_folder)

    # Identify all possible metadata columns (excluding '_id', 'x', and 'y')
    metadata_columns = [col for col in df.columns if col not in ['_id', 'x', 'y']]

    def create_figure(color_by=None, symbol_by=None):
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color=color_by if color_by else None,
            symbol=symbol_by if symbol_by else None,
            title=f'Interactive Token Visualization of {token}-{model}',
            labels={'x': 'X-Axis', 'y': 'Y-Axis'},
            hover_name='_id',
            color_discrete_sequence=['blue'] if not color_by else None
        )

        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x"),
            width=1000,
            height=1000
        )
        return fig

    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Layout of the app
    app.layout = html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=create_figure(),
            config={'scrollZoom': True},
            style={'display': 'inline-block', 'width': '1000px', 'height': '1000px'}
        ),
        html.Div(
            id='selected-tokens',
            style={'margin-top': '20px', 'font-size': '16px'}
        ),
        dcc.Textarea(
            id='token-list',
            style={'width': '100%', 'height': '200px', 'margin-top': '10px'},
            placeholder='List of selected token IDs will appear here...'
        ),
        html.Div([
            dcc.Dropdown(
                id='color-dropdown',
                options=[{'label': col, 'value': col} for col in metadata_columns],
                placeholder="Select a column for colour coding",
                style={'width': '300px'}
            ),
            dcc.Dropdown(
                id='shape-dropdown',
                options=[{'label': col, 'value': col} for col in metadata_columns],
                placeholder="Select a column for shape coding",
                style={'width': '300px', 'margin-top': '10px'}
            )
        ], style={'margin-top': '20px'}),
        html.Button('Clear Selection', id='clear-button', n_clicks=0, style={'margin-top': '20px'})
    ])

    # Combined callback to handle updates for the scatter plot, token selection, and clear selection
    @app.callback(
        [Output('scatter-plot', 'figure'),
         Output('selected-tokens', 'children'),
         Output('token-list', 'value')],
        [Input('color-dropdown', 'value'),
         Input('shape-dropdown', 'value'),
         Input('scatter-plot', 'selectedData'),
         Input('clear-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_app(color_by, shape_by, selectedData, clear_clicks):
        """
        Update the scatter plot, selected tokens, and handle graph reset when the clear button is clicked.

        Parameters
        ----------
        color_by : str
            The column selected for colour coding.
        shape_by : str
            The column selected for shape coding.
        selectedData : dict
            Data representing the points selected by the user on the scatter plot.
        clear_clicks : int
            The number of times the 'Clear Selection' button has been clicked.

        Returns
        -------
        dict
            The updated figure for the scatter plot.
        str
            A message indicating the selected tokens.
        str
            The full list of selected token IDs.
        """
        ctx = dash.callback_context

        # Check which input triggered the callback
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # If the clear button was clicked, reset everything
            if trigger_id == 'clear-button':
                return create_figure(), "No tokens selected.", ""

            # If metadata selection changed, update the figure without resetting selections
            if trigger_id in ['color-dropdown', 'shape-dropdown']:
                return create_figure(color_by=color_by, symbol_by=shape_by), dash.no_update, dash.no_update

        # Handle token selection update
        selected_tokens = set()
        if selectedData:
            selected_tokens.update([point['hovertext'] for point in selectedData['points']])

        if not selected_tokens:
            return dash.no_update, "No tokens selected.", ""
        
        return dash.no_update, f"Selected Tokens: {len(selected_tokens)} tokens selected", ", ".join([f"'{token}'" for token in sorted(selected_tokens)])

    return app

# Run the app
def get_token_ids(token, model, input_folder):
    app = create_app(token, model, input_folder)
    app.run_server(debug=True, port=1111)
    print('Open the app in your browser (http://127.0.0.1:1111).')
