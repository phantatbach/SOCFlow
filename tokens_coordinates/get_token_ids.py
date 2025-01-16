# Running the App:
# Save the code to a Python file (get_tokens_ids.py).
# Run the file: python get_tokens_ids.py.
# Open the app in your browser (http://127.0.0.1:8050).

import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Output, Input, State
import os

def create_app(token, model, input_folder):
    """
    Create a Dash app that displays an interactive scatter plot of the tokens and their senses in a given model.
    """
    df = load_data(token, model, input_folder)

    def create_figure():
        """
        Create a scatter plot of the tokens and their senses in a given model.
        """
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='senses',
            title=f'Interactive Token Visualization of {token}-{model}',
            labels={'x': 'X-Axis', 'y': 'Y-Axis'},
            hover_name='_id'
        )
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x"),
            width=1000,
            height=1000
        )
        return fig

    fig = create_figure()

    # Initialize the Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=fig,
            config={'scrollZoom': True},
            style={'display': 'inline-block', 'width': '1000px', 'height': '1000px'}
        ),
        html.Div(
            id='selected-tokens',
            style={'margin-top': '20px', 'font-size': '16px'}
        ),
        html.Div([
            dcc.Dropdown(
                id='sense-dropdown',
                options=[{'label': sense, 'value': sense} for sense in sorted(df['senses'].unique())],
                placeholder="Select a sense",
                style={'width': '300px'}
            )
        ], style={'margin-top': '20px'}),
        html.Button('Clear Selection', id='clear-button', n_clicks=0, style={'margin-top': '20px'})
    ])

    # Callback to handle all updates: selected tokens and clear button
    @app.callback(
        [Output('selected-tokens', 'children'),
         Output('scatter-plot', 'figure'),
         Output('sense-dropdown', 'value')],

        [Input('scatter-plot', 'selectedData'),
         Input('sense-dropdown', 'value'),
         Input('clear-button', 'n_clicks')],
        prevent_initial_call=True
    )

    def update_selected_tokens(selectedData, selected_sense, clear_clicks):
        """
        Update the selected tokens based on user interaction with the scatter plot and sense dropdown.

        Parameters
        ----------
        selectedData : dict
            Data representing the points selected by the user on the scatter plot.
        selected_sense : str
            The sense selected by the user from the dropdown menu.
        clear_clicks : int
            The number of times the 'Clear Selection' button has been clicked.

        Returns
        -------
        str
            A message indicating the selected tokens.
        dash.Figure
            The updated figure for the scatter plot.
        any
            The updated value for the sense dropdown.
        """

        # Handle clear button: Reset everything
        ctx = dash.callback_context
        if ctx.triggered and 'clear-button' in ctx.triggered[0]['prop_id']:
            return "No tokens selected.", create_figure(), None

        selected_tokens = set()

        # Add tokens from box selection
        if selectedData:
            selected_tokens.update([f"'{point['hovertext']}'" for point in selectedData['points']])

        # Add tokens from sense selection
        if selected_sense:
            tokens_with_sense = df.loc[df['senses'] == selected_sense, '_id'].astype(str)
            selected_tokens.update(f"'{token}'" for token in tokens_with_sense)

        if not selected_tokens:
            return "No tokens selected.", dash.no_update, dash.no_update

        return f"Selected Tokens: {', '.join(sorted(selected_tokens))}", dash.no_update, dash.no_update

    return app
