import os
import sys
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Path management
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(project_root)

# Import utils
from utils import *

# Flask Setup
server = Flask(__name__)

# Define model and predictions paths
model_path = os.path.join(project_root, 'results', 'best_model.pkl')
predictions_path = os.path.join(project_root, 'results', 'objects_final_predictions.pkl')

# Load model and data
best_model = joblib.load(model_path)
objects_final_predictions = joblib.load(predictions_path)

# Create Dash app
app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dashboard/')

# Layout for Dash app
app.layout = html.Div(
    style={'backgroundColor': '#2c2f33', 'color': '#ffffff', 'padding': '20px'},
    children=[
        html.H1("Dashboard - Claim Injury Type Prediction", style={'color': '#bfd630', 'font-family': 'Arial'}),
        html.Div(id='file-name-container', style={'color': '#bfd630', 'margin-bottom': '30px', 'font-family': 'Arial'}),
        
        # Dropdown for selecting feature
        dcc.Dropdown(
            id='feature-dropdown',
            options=[
                {'label': feature, 'value': feature}
                for feature in ['Claim Injury Type'] + sorted([
                    'Accident Date', 'Age at Injury', 'Alternative Dispute Resolution', 'Assembly Date',
                    'Attorney/Representative', 'Average Weekly Wage', 'C-2 Date', 'C-3 Date', 'Carrier Type',
                    'COVID-19 Indicator', 'District Name', 'First Hearing Date',
                    'Medical Fee Region', 'Male', 'C-2 Date Missing', 
                    'C-3 Date Missing', 'First Hearing Date Missing', 
                    'Part of Body Category', 'Nature of Injury Category', 'Cause of Injury Category', 
                ])
            ],
            value='Claim Injury Type',
            style={
                'backgroundColor': '#ffffff', 'color': '#2c2f33', 'fontSize': '16px', 
                'borderRadius': '5px', 'border': '1px solid #ffffff', 'width': '50%',
            },
            placeholder="Select a feature",
        ),
        
        # Graph for visualizing data
        dcc.Graph(id='feature-graph'),
    ]
)

@app.callback(
    [Output('feature-graph', 'figure'),
     Output('file-name-container', 'children')],
    [Input('feature-dropdown', 'value')]
)
def update_graph(selected_feature):
    color_map = {
        'Claim Injury Type': {
            'type1': '#2ca02c', 'type2': '#ff7f0e', 'type3': '#1f77b4', 'type4': '#d62728', 
            'type5': '#9467bd', 'type6': '#8c564b', 'type7': '#e377c2', 'type8': '#7f7f7f',
        },
        'binary': {0: '#7f7f7f', 1: '#bfd630'}
    }

    # Get the latest predictions file (based on date)
    predictions_folder = os.path.join(project_root, 'predictions')
    files = [f for f in os.listdir(predictions_folder) if f.endswith('.csv')]
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(predictions_folder, f)))
    predictions_filepath = os.path.join(predictions_folder, latest_file)

    # Get the name of the current predictions file to display in the dashboard
    current_file_name = latest_file if os.path.exists(predictions_filepath) else "No Data Available"

    # If the predictions file does not exist, show a placeholder graph
    if not os.path.exists(predictions_filepath):
        return px.box(title="No Data Available", template="plotly_dark").update_layout(
            paper_bgcolor='#2c2f33', plot_bgcolor='#2c2f33', font_color='#ffffff'
        ), f"Predictions File Used: {current_file_name}"

    df = pd.read_csv(predictions_filepath)

    # Sort 'Claim Injury Type' alphabetically before plotting
    df['Claim Injury Type'] = pd.Categorical(df['Claim Injury Type'], categories=sorted(df['Claim Injury Type'].unique()), ordered=True)

    if selected_feature == 'Claim Injury Type':
        df_claim_injury_counts = df['Claim Injury Type'].value_counts().reset_index()
        df_claim_injury_counts.columns = ['Claim Injury Type', 'Count']

        fig = px.bar(df_claim_injury_counts, x='Claim Injury Type', y='Count',
                     title='Claim Injury Type Frequencies',
                     labels={'Claim Injury Type': 'Claim Injury Type', 'Count': 'Frequency'},
                     color='Claim Injury Type', template='plotly_dark', 
                     color_discrete_map=color_map['Claim Injury Type'])
        fig.update_layout(
            xaxis_title="Claim Injury Type",
            yaxis_title="Count",
            showlegend=False
        )

    elif selected_feature in ['Age at Injury', 'Average Weekly Wage', 'Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date']:
        fig = px.box(df, x='Claim Injury Type', y=selected_feature, title=f'{selected_feature} Distribution by Claim Injury Type',
                     color='Claim Injury Type', template='plotly_dark',
                     category_orders={'Claim Injury Type': sorted(df['Claim Injury Type'].unique())},
                     color_discrete_map=color_map['Claim Injury Type'])
        fig.update_layout(
            xaxis_title="Claim Injury Type",
            yaxis_title=selected_feature,
            showlegend=False
        )

    elif selected_feature in ['Alternative Dispute Resolution', 'Attorney/Representative', 
                            'COVID-19 Indicator', 'Male', 'C-2 Date Missing', 'C-3 Date Missing', 
                            'First Hearing Date Missing']:
        binary_counts = df.groupby(['Claim Injury Type', selected_feature]).size().reset_index(name='Count')
        binary_counts[selected_feature] = binary_counts[selected_feature].astype('object')

        fig = px.bar(binary_counts, x='Claim Injury Type', y='Count', color=selected_feature,
                    title=f'{selected_feature} Distribution by Claim Injury Type',
                    barmode='stack', template='plotly_dark',
                    color_discrete_map=color_map['binary'])
        fig.update_traces(marker_line=dict(color='white', width=1))

    elif selected_feature in ['Carrier Type', 'District Name', 'Medical Fee Region', 'Part of Body Category', 'Nature of Injury Category', 'Cause of Injury Category']:
        categorical_counts = df.groupby([selected_feature, 'Claim Injury Type']).size().reset_index(name='Count')
        categorical_counts = categorical_counts.sort_values(by=[selected_feature, 'Count'], ascending=[True, False])

        fig = px.bar(categorical_counts, x=selected_feature, y='Count', color='Claim Injury Type',
                     title=f'{selected_feature} Distribution by Claim Injury Type',
                     barmode='group', template='plotly_dark',
                     category_orders={'Claim Injury Type': sorted(df['Claim Injury Type'].unique())},
                     color_discrete_map=color_map['Claim Injury Type'])
        fig.update_traces(marker_line=dict(color='white', width=1))

        fig.update_layout(
            xaxis_title=selected_feature,
            yaxis_title="Count"
        )

    else:
        fig = px.box(title="No Data Available", template="plotly_dark").update_layout(
            paper_bgcolor='#2c2f33', plot_bgcolor='#2c2f33', font_color='#ffffff'
        )

    # Customize hover behavior for all graph types
    fig.update_traces(
        hoverlabel=dict(
            bgcolor="white", font=dict(color="black"),
        ),
        hovertemplate="%{x}: %{y}"
    )

    # Customize layout
    fig.update_layout(
        plot_bgcolor='#2c2f33',
        paper_bgcolor='#2c2f33',
        font_color='#ffffff',
        title_x=0.5,
        hovermode="closest"
    )

    return fig, html.Div([
        html.Span("Predictions File Used: ", style={'color': 'bfd630'}),
        html.Span(current_file_name, style={'color': 'white'})
    ])

# Flask Route for File Upload and Prediction
@server.route('/')
def home():
    return render_template('index.html')

@server.route('/upload', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return {"ERROR": "No file uploaded."}, 400

    file = request.files['file']
    upload_folder = os.path.join(project_root, 'uploads')
    predictions_folder = os.path.join(project_root, 'predictions')

    # Create the necessary directories if they don't exist
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(predictions_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    try:
        df_new_inputs = pd.read_csv(filepath, sep=',', index_col='Claim Identifier')
        df_predictions = predict_new_inputs(df_new_inputs, best_model, objects_final_predictions)
    except Exception as e:
        return {"ERROR": str(e)}, 500

    # Save predictions to a new file with current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    result_filepath = os.path.join(predictions_folder, f'predictions_{current_date}.csv')
    df_predictions.to_csv(result_filepath, index=True)

    # Send the file for download directly after processing
    return send_file(result_filepath, as_attachment=True, download_name=f'predictions_{current_date}.csv')

if __name__ == '__main__':
    server.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))