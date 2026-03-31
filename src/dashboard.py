import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from datetime import date

from src.data_processing import WeatherDataProcessor
from src.visualization import (
    get_visualisation_data,
    get_matrice_correlation,
    get_graph_comparaison,
)

app = dash.Dash(__name__)
app.title = "Météo Analytics"
processor = WeatherDataProcessor()

# liste des départements
try:
    liste_departements=processor.get_available_departments()
except Exception:
    liste_departements=[]

CARD_STYLE = {
    'backgroundColor': 'white',
    'borderRadius': '15px',
    'boxShadow': '0 4px 15px rgba(0,0,0,0.05)',
    'padding': '15px',
    'boxSizing': 'border-box',
    'height': '100%',
    'display': 'flex',
    'flexDirection': 'column' 
}

GRAPH_STYLE = {'flexGrow': '1', 'width': '100%', 'height': '100%'}
GRAPH_CONFIG = {'responsive': True, 'displayModeBar': False} 


# --- Layout ---
app.layout = html.Div([
    html.Div([
        
        # contrôleur : sélection département + période
        html.Div([
            html.H2("Paramètres", style={'color': '#2c3e50', 'marginTop': '0', 'fontSize': '1.5rem'}),
            
            html.Label("Département :", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='selection-departement',
                options=[{'label': str(d), 'value': d} for d in liste_departements],
                value=liste_departements[0] if liste_departements else None,
                clearable=False,
                style={'marginBottom': '20px'}
            ),

            html.Label("Période :", style={'fontWeight': 'bold'}),
            html.Div(
                dcc.DatePickerRange(
                    id='selection-date',
                    start_date=date(2010, 1, 1),
                    end_date=date(2023, 12, 31),
                    display_format='DD/MM/YYYY',
                    style={'width': '100%'}
                ), style={'marginBottom': '20px'}
            ),
            
            html.Hr(),

        ], style={**CARD_STYLE, 'width': '30%', 'marginRight': '20px'}),

        # matrice de corrélation
        html.Div([
            html.H4("Corrélation des variables", style={'margin': '0 0 10px 0', 'textAlign': 'center'}),
            dcc.Graph(id='matrice-correlation', style=GRAPH_STYLE, config=GRAPH_CONFIG)
        ], style={**CARD_STYLE, 'width': '70%', 'minWidth': '0'})

    ], style={'display': 'flex', 'height': '350px', 'marginBottom': '20px'}),

    html.Div([
        
        # Température VS Ensoleillement
        html.Div([
            html.H4("Température vs Ensoleillement", style={'textAlign': 'center', 'margin': '0 0 10px 0'}),
            dcc.Graph(id='comparaison-sun-temp', style=GRAPH_STYLE, config=GRAPH_CONFIG)
        ], style={**CARD_STYLE, 'width': '49%', 'marginRight': '2%', 'minWidth': '0'}),

        # Température VS Précipitations
        html.Div([
            html.H4("Température vs Précipitations", style={'textAlign': 'center', 'margin': '0 0 10px 0'}),
            dcc.Graph(id='comparaison-rain-temp', style=GRAPH_STYLE, config=GRAPH_CONFIG)
        ], style={**CARD_STYLE, 'width': '49%', 'minWidth': '0'})

    ], style={'display': 'flex', 'height': '350px', 'marginBottom': '20px'}),



    # visualisation Température, Soleil, Pluie, Vent
    html.Div([
        html.H3("Chronologie Météorologique Complète", style={'margin': '0 0 15px 0'}),
        dcc.Loading(
            type="circle", 
            # Application des styles et config réactifs
            children=dcc.Graph(id='graph-visualisation', style=GRAPH_STYLE, config=GRAPH_CONFIG),
            style={'height': '100%'} # Le Loading doit aussi prendre toute la hauteur
        )
    ], style={**CARD_STYLE, 'height': '600px'}) # Hauteur de la carte fixée

], style={'backgroundColor': '#f4f6f9', 'padding': '20px', 'fontFamily': 'Segoe UI, sans-serif', 'minHeight': '100vh'})


# callbaks
@app.callback(
    [Output('graph-visualisation', 'figure'),
     Output('matrice-correlation', 'figure'),
     Output('comparaison-sun-temp', 'figure'),
     Output('comparaison-rain-temp', 'figure')],
    [Input('selection-departement', 'value'),
     Input('selection-date', 'start_date'),
     Input('selection-date', 'end_date')]
)
def update_dashboard(dept_code, start_date, end_date):
    empty = px.scatter(title="Pas de données")

    if not dept_code:
        return empty, empty, empty, empty

    df = processor.load_dept_data(str(dept_code))
    
    if df is None:
        return empty, empty, empty, empty

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    if start_date and end_date:
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_filtered = df.loc[mask]
        if not df_filtered.empty:
            df = df_filtered

    fig_main=get_visualisation_data(df)
    fig_corr=get_matrice_correlation(df)
    fig_sun, fig_rain = get_graph_comparaison(df)

    return fig_main, fig_corr, fig_sun, fig_rain

if __name__ == '__main__':
    app.run(debug=True, port=8050)