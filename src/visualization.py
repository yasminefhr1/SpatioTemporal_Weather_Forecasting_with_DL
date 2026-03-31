import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def get_visualisation_data(df):
    fig=make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        subplot_titles=("Températures", "Ensoleillement", "Précipitations", "Vent (FFM)"),
        row_heights=[0.35, 0.2, 0.2, 0.25]
    )

    # températures : min/max et moyenne
    if 'Temp_Min' in df.columns and 'Temp_Max' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Temp_Min'],
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Temp_Max'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)',
            name='Amplitude',
            legendgroup='temp'
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Temperature'],
        mode='lines', line=dict(color='red', width=2),
        name='Temp Moyenne',
        legendgroup='temp'
    ), row=1, col=1)

    # ensoleillement
    if 'Ensoleillement' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Ensoleillement'],
            mode='lines', line=dict(color='orange'),
            fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.3)',
            name='Ensoleillement'
        ), row=2, col=1)

    # précipitations
    fig.add_trace(go.Bar(
        x=df.index, y=df['Precipitation'],
        marker_color='blue', opacity=0.6,
        name='Précipitations'
    ), row=3, col=1)

    # vent
    if 'Wind' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Wind'],
            mode='lines', line=dict(color='green'),
            fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.1)',
            name='Vent Moyen'
        ), row=4, col=1)


    fig.update_layout(
        autosize=True,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )
    
    # labels
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    fig.update_yaxes(title_text="m/s", row=3, col=1)
    fig.update_yaxes(title_text="Heures", row=4, col=1)

    return fig

def get_matrice_correlation(df):
   # matrice de corrélation
    target_cols = ['Temperature', 'Temp_Min', 'Temp_Max', 'Ensoleillement', 'Precipitation', 'Wind']
    cols_present = [c for c in target_cols if c in df.columns]
    
    if len(cols_present) < 2:
        return px.scatter(title="Pas assez de données pour les corrélations")

    corr_matrix = df[cols_present].corr()

    fig = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale='RdBu_r', 
        zmin=-1, zmax=1, 
        aspect="auto"
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), autosize=True,)
    return fig

def get_graph_comparaison(df):
   # Ensoleillement VS Température
    if 'Ensoleillement' in df.columns and 'Temperature' in df.columns:
        fig_sun = px.scatter(
            df, x='Ensoleillement', y='Temperature', 
            trendline="ols",
            color='Temperature', 
            title="Température VS ensoleillement",
            labels={'Ensoleillement': 'Soleil (h)', 'Temperature': 'Temp (°C)'},
            opacity=0.6
        )
    else:
        fig_sun = px.scatter(title="Données manquantes (Soleil)")

    # Pluie VS Température
    if 'Precipitation' in df.columns and 'Temperature' in df.columns:
        fig_rain = px.scatter(
            df, x='Precipitation', y='Temperature', 
            trendline="ols",
            size='Precipitation', 
            title="Température vs Précipitations",
            labels={'Temperature': 'Temp (°C)', 'Precipitation': 'Pluie (mm)'},
            opacity=0.5
        )
    else:
        fig_rain = px.scatter(title="Données manquantes (Pluie)")

    for fig in [fig_sun, fig_rain]:
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), autosize=True)
        
    return fig_sun, fig_rain