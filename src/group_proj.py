# %% [markdown]
# Group members:
# 
# 
# Zhang Fengzhi A0298678X (Responsible for data processing and confidence band filling)
# 
# 
# Xie Conghui A0297300J   (Responsible for line animation and slider)
# 
# 
# Deng Xinyi A0298862H    (Responsible for framework construction)
# 
# 
# Li Jinheng A0299039M    (Responsible for label buttons and adding time intervals)
# 
# 
# Chen Shu A0297139N      (Responsible for tail mark and axis modifications)

# %% [markdown]
# Reminder: Graphics can't be displayed on html pages, you need to run them locally!

# %%
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# %%
Global = pd.read_csv("./data/Global Temperature Anomalies.csv")
pi = pd.read_csv("./data/piControl.csv")
force = pd.read_csv("./data/forcings.csv")

Global = Global[Global['Hemisphere'] == 'Global'][["Year", "J-D"]]
Global.columns = ["Year","Observed"]

df = pd.merge(Global, force, on="Year")
print(df)

# %%
# Calculation of 1880-2005 length
years = len(df)

# Numerical modelling started in 1850, so 30 years of backward adjustment
pi = pi.iloc[30:years+30,:].reset_index(drop=True)

# %%
df.iloc[:,2:] = df.iloc[:,2:].sub(pi['Temp (K)']+0.1,axis=0)
print(df)

# %%
df = df[['Year','Observed','All forcings','Anthropogenic tropospheric aerosol','Greenhouse gases','Land use','Orbital changes','Ozone','Solar','Volcanic']]
print(df)

# %%
def bootstrap_ci(data, num_bootstrap_samples=1000, ci=95, width_multiplier=1.0):
    bootstrap_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) 
                       for _ in range(num_bootstrap_samples)]
    
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    mean_estimate = np.mean(bootstrap_means)
    lower_bound_adjusted = mean_estimate - (mean_estimate - lower_bound) * width_multiplier
    upper_bound_adjusted = mean_estimate + (upper_bound - mean_estimate) * width_multiplier
    
    return lower_bound_adjusted, upper_bound_adjusted

def calculate_bootstrap_ci(df, columns, window_size=1, num_bootstrap_samples=1000, ci=95, width_multiplier=1.0):
    df_bound = df.copy()

    for column in columns:
        ci_lower = []
        ci_upper = []

        column_data = df[column]

        for i in range(len(df)):
            # Extract the data within the rolling window
            window_data = column_data[max(0, i - window_size):min(len(column_data), i + window_size + 1)]

            # Calculate bootstrap confidence intervals
            lower, upper = bootstrap_ci(window_data, num_bootstrap_samples, ci, width_multiplier)
            ci_lower.append(lower)
            ci_upper.append(upper)

        # Add the lower and upper bounds to the DataFrame
        df_bound[f'{column} Lower'] = ci_lower
        df_bound[f'{column} Upper'] = ci_upper

    return df_bound



# %%
# Generate data every 3,5,10 years
df3 = df.groupby(df['Year'] // 3 * 3,as_index=False).mean()
df5 = df.groupby(df['Year'] // 5 * 5,as_index=False).mean()
df10 = df.groupby(df['Year'] // 10 * 10,as_index=False).mean()

# %%
df_bound = calculate_bootstrap_ci(df, columns=['All forcings'])
df_bound3 = calculate_bootstrap_ci(df3, columns=['All forcings'])
df_bound5 = calculate_bootstrap_ci(df5, columns=['All forcings'])
df_bound10 = calculate_bootstrap_ci(df10, columns=['All forcings'])

# %%
# Average 1880-1910
Mean = df['Observed'].iloc[:30].mean()
print(Mean)

# %%
app = Dash(__name__)

labels = [ 
    'Anthropogenic tropospheric aerosol', 'Greenhouse gases', 
    'Land use', 'Orbital changes', 'Ozone', 'Solar', 'Volcanic'
]
colors = [ '#34495e', 
    '#e67e22', '#f1c40f', '#95a5a6', '#d35400', '#2ecc71', '#8e44ad'
]


app.layout = html.Div([
        dcc.Dropdown(
        id='interval-dropdown',
        options=[
            {'label': 'Every year', 'value': 'df'},
            {'label': 'Average Every 3 Years', 'value': 'df3'},
            {'label': 'Average Every 5 Years', 'value': 'df5'},
            {'label': 'Average Every 10 Years', 'value': 'df10'}
        ],
        value='df', 
        style={'width': '50%', 'margin': 'auto'}
    ),
    dcc.Graph(
        id='line-chart',
        config={'displayModeBar': False},
        style={'backgroundColor': 'white', 'width': '70%', 'height': '80vh', 'margin': 'auto'}
    ),
    html.Div([
        html.Button(label, id={'type': 'label-button', 'index': label}, n_clicks=0, style={
            'backgroundColor': color,
            'color': 'white',
            'padding': '5px 10px',
            'border-radius': '5px',
            'margin': '3px',
            'text-align': 'center',
            'font-size': '12px',
            'border': 'none',
            'cursor': 'pointer'
        }) for label, color in zip(labels, colors)
    ], style={
        'text-align': 'center',
        'display': 'flex',
        'flex-direction': 'row',
        'flex-wrap': 'wrap',
        'justify-content': 'center',
        'align-items': 'center',
        'margin-top': '20px'
    })
])

dataframes = {'df': df, 'df3': df3, 'df5': df5, 'df10': df10}
dataframes_bound = {'df': df_bound, 'df3': df_bound3, 'df5': df_bound5, 'df10': df_bound10}


@app.callback(
    Output('line-chart', 'figure'),
    [Input({'type': 'label-button', 'index': label}, 'n_clicks') for label in labels] +
    [Input('interval-dropdown', 'value')]
)

def update_chart(*args):
    selected_df_key = args[-1]
    selected_df = dataframes[selected_df_key]
    df_confidence = dataframes_bound[selected_df_key]
    
    fig = go.Figure()
    frames = []
    
    observed_trace = go.Scatter(
        x=selected_df['Year'],
        y=selected_df['Observed'],
        mode='lines+text',
        name='Observed',
        line=dict(width=2, color='black'),
        text=[None] * (len(selected_df['Year']) - 1) + ['Observed'],
        textposition='top right',
        showlegend=False 
    )
    
    all_forcings_trace = go.Scatter(
        x=selected_df['Year'],
        y=selected_df['All forcings'],
        mode='lines+text',
        name='All forcings',
        line=dict(width=2, color='red'),
        text=[None] * (len(selected_df['Year']) - 1) + ['All forcings'],
        textposition='top right',
        showlegend=False 
    )
    

    all_forcings_upper = go.Scatter(
        x=df_confidence['Year'],
        y=df_confidence['All forcings Upper'], 
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    )
    all_forcings_lower = go.Scatter(
        x=df_confidence['Year'],
        y=df_confidence['All forcings Lower'], 
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)', 
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    )
    
    fig.add_trace(observed_trace)
    fig.add_trace(all_forcings_trace)
    fig.add_trace(all_forcings_upper)
    fig.add_trace(all_forcings_lower)

    active_traces = []
    for i, (label, clicks) in enumerate(zip(labels, args[:-1])):
        if clicks % 2 == 1:  
            active_traces.append(go.Scatter(
                x=selected_df['Year'],
                y=selected_df[label],
                mode='lines+text',
                name=label,
                line=dict(width=2, color=colors[i]),
                text=[None] * (len(selected_df['Year']) - 1) + [label],  
                textposition='top right',
                showlegend=False
            ))

    fig.add_traces(active_traces)

    for frame_idx in range(len(selected_df['Year'])):
        frame_data = [observed_trace,
            all_forcings_trace,
            all_forcings_upper,
            all_forcings_lower
        ] + [
            go.Scatter(
                x=selected_df['Year'][:frame_idx + 1],
                y=selected_df[trace.name][:frame_idx + 1],
                mode='lines+text',
                name=trace.name,
                line=dict(width=2, color=trace.line.color),
                text=[None] * frame_idx + [trace.name],
                textposition='top right',
                showlegend=False
            ) for trace in active_traces
        ]
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(selected_df['Year'][frame_idx])
        ))

# def update_chart(*args):
#     selected_df_key = args[-1]
#     selected_df = dataframes[selected_df_key]
    
#     fig = go.Figure()
#     frames = []
    
#     observed_trace = go.Scatter(
#         x=selected_df['Year'],
#         y=selected_df['Observed'],
#         mode='lines',
#         name='Observed',
#         line=dict(width=2, color='black'),
#         showlegend=False 
#     )
#     all_forcings_trace = go.Scatter(
#         x=selected_df['Year'],
#         y=selected_df['All forcings'],
#         mode='lines',
#         name='All forcings',
#         line=dict(width=2, color='red'),
#         showlegend=False 
#     )
    
#     all_forcings_upper = go.Scatter(
#         x=df_bound['Year'],
#         y=df_bound['All forcings Upper'],  
#         mode='lines',
#         line=dict(width=0),
#         showlegend=False,
#         hoverinfo='skip'
#     )
#     all_forcings_lower = go.Scatter(
#         x=df_bound['Year'],
#         y=df_bound['All forcings Lower'], 
#         mode='lines',
#         fill='tonexty',  
#         fillcolor='rgba(255, 0, 0, 0.2)', 
#         line=dict(width=0),
#         showlegend=False,
#         hoverinfo='skip'
#     )
    
#     fig.add_trace(observed_trace)
#     fig.add_trace(all_forcings_trace)
#     fig.add_trace(all_forcings_upper)
#     fig.add_trace(all_forcings_lower)
    
#     active_traces = []
#     for i, (label, clicks) in enumerate(zip(labels, args[:-1])):
#         if clicks % 2 == 1:  # Only add trace if clicks are odd
#             active_traces.append(go.Scatter(
#                 x=selected_df['Year'],
#                 y=selected_df[label],
#                 mode='lines',
#                 name=label,
#                 line=dict(width=2, color=colors[i]),
#                 showlegend=False
#             ))

#     fig.add_traces(active_traces)

#     for frame_idx in range(len(selected_df['Year'])):
#         frame_data = [
#             observed_trace,
#             all_forcings_trace,
#             all_forcings_upper,
#             all_forcings_lower
#         ] + [
#             go.Scatter(
#                 x=selected_df['Year'][:frame_idx + 1],
#                 y=selected_df[trace.name][:frame_idx + 1],
#                 mode='lines',
#                 name=trace.name,
#                 line=dict(width=2, color=trace.line.color),
#                 showlegend=False
#             ) for trace in active_traces
#         ]
        
#         frames.append(go.Frame(
#             data=frame_data,
#             name=str(selected_df['Year'][frame_idx])
#         ))

    fig.update_layout(
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            visible=False,
            range=[selected_df['Year'].min(), selected_df['Year'].max()+30],
            tickvals=[df['Year'].min(), df['Year'].max()],
            ticktext=["1880", "2005"],
            tickangle=45,
        ),
        yaxis=dict(
            range=[Mean - 2, Mean + 2],
            showticklabels=True,
            showline=True,
            linecolor='gray',
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickcolor='gray',
            tickvals=[Mean - 2, Mean, Mean + 2],
            ticktext=["-2℉", "1880-1910\nAverage", "+2°C"],
            tickangle=0,
            linewidth=2
            ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": -0.1,
            "y": 0.2,
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                    "label": "play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                    "transition": {"duration": 0}}], 
                    "label": "pause",
                    "method": "animate"
                }
            ]
        }],
        sliders=[{
            "y": -0.05,
            "steps": [
                {"args": [[str(year)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                 "label": str(year),
                 "method": "animate"} for year in selected_df['Year']
            ],
            "active": 0,
        }],
        annotations=[
            dict(
                x=0.01, y=Mean + 2,
                xref="paper", yref="y",
                text="Hotter", showarrow=False,
                font=dict(size=12, color="black"),
            ),
            dict(
                x=0.01, y=Mean - 2,
                xref="paper", yref="y",
                text="Colder", showarrow=False,
                font=dict(size=12, color="black"),
            ),
            dict(
                x=0.5, y=-0.2,
                xref="paper", yref="paper",
                text="1880-2005", showarrow=False,
                font=dict(size=25, color="gray"),
            )
        ],
        shapes=[
            dict(
                type='line',
                x0=df['Year'].min(), y0=Mean,
                x1=df['Year'].max(), y1=Mean,
                line=dict(color='gray', width=2)
            ) 
        ],
    )

    fig.update(frames=frames)
    return fig


# %%
## !!!!! Please Read:
## Click on the button to show the corresponding line, click again to disappear, support multiple lines playback at the same time
## Try to use slide track, the tail markers will blink when the play button is playing.

## if __name__ == '__main__':
##     app.run_server(debug=True, port=8063)

if __name__ == '__main__':
    app.run_server(debug=False,port = 6500)
    
## import streamlit as st
## st.title("Streamlit Dashboard with Embedded Dash App")
## # 使用 iframe 嵌入 Dash 应用
## st.components.v1.iframe(src="http://127.0.0.1:9000", width=800, height=600)


# %% [markdown]
# We acknowledge the use of ChatGPT to debug code that was subsequently included in modified form in my report (https://chatgpt.com/share/6731c4a5-c804-8004-91b9-ccb8a51e7929). We entered the following prompt(s) on November 11, 2024: How to animate data.......

# %% [markdown]
# We acknowledge the use of ChatGPT to debug code that was subsequently included in modified form in my report (https://chatgpt.com/share/67320d24-3aa0-8002-b8ce-105d7f840bfe). We entered the following prompt(s) on November 11, 2024: Using dash and plotly for diagrams, how to add a drop down menu to show other diagrams. How to add a label button below the chart that can be interactively clicked to display the line......


