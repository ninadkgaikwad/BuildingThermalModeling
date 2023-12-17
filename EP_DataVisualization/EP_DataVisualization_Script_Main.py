# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:06:14 2022

@author: ninad
"""

# Importing Required Modules
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
from datetime import date
import pickle
import pandas as pd

# Importing User-Defined Modules
import MyDashApp_Module as AppFuncs

# Instantiate our App and incorporate BOOTSTRAP theme Stylesheet
# Themes - https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/#available-themes
# Themes - https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/explorer/
# hackerthemes.com/bootstrap-cheatsheet/

app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

# App Layout using Dash Bootstrap


    # Converting text to numbers


    # # Row 3
    # dbc.Row([

app.layout = dbc.Container([

    # Row 1
    dbc.Row([

        dbc.Col([

            html.H1("Visualization of Thermodynamic System Variables",
                    className = 'text-center text-primary mb-4')

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "center", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 2
    dbc.Row([

        dbc.Col([

            html.H3("Make a Selection for Data Type:",
                    className = 'text-left text-secondary mb-4')

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "left", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 3
    dbc.Row([

        dbc.Col([

            dcc.RadioItems(
                    id = 'File_Type_RadioItem',
                    labelStyle = {'display': 'block'},
                    options = [
                        {'label' : "Raw Data", 'value' : 1},
                        {'label' : "Aggregated Data", 'value' : 2},
                        ]  ,
                     value = 1,
                     className = "mb-2"
                ),

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "center", align = "center"),


    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 4
    dbc.Row([

        dbc.Col([

            html.H3("Folder Path for Uploading Files:",
                    className = 'text-left text-secondary mb-4')

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "left", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 5, upload files
    dcc.Upload(
        id='User_Input_File',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select the Data File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload1'),


    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 7
    dbc.Row([

        dbc.Col([

            html.H3("Date Range from Uploaded File:",
                    className = 'text-left text-secondary mb-4')

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "left", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 8
    dbc.Row([

    dbc.Col([

        html.Label(children = "Start Date:"),

    ], xs=6, sm=6, md=6, lg=3, xl=3),  # width = 12

    dbc.Col([

        html.Label("", id = 'Start_Date_Label'),

    ], xs=6, sm=6, md=6, lg=3, xl=3),  # width = 12

    dbc.Col([

        html.Label("End Date:"),

    ], xs=6, sm=6, md=6, lg=3, xl=3),  # width = 12

    dbc.Col([

        html.Label("", id = 'End_Date_Label'),

    ], xs=6, sm=6, md=6, lg=3, xl=3),  # width = 12

    ], justify = "center", align = "center"),


    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 9
    dbc.Row([

        dbc.Col([

            html.H3("Select Date Range for Visualization:",
                    className = 'text-left text-secondary mb-4')

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "left", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 10
    dbc.Row([

        dbc.Col([
            dcc.DatePickerRange(
                id='User_Input_Date_Range',
                min_date_allowed=date(2000, 1, 1),
                max_date_allowed=date(2021, 12, 31),
                initial_visible_month=date(2020, 1, 1),
                end_date=date(2020, 12, 31)
            ),
        ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

    ], justify = "left", align = "center"),


    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),


    # Row 11
    dbc.Row([

        dbc.Col([

            html.H3("Time Series Plot:",
                    className = 'text-left text-secondary mb-4')

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "left", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 12
    dbc.Row([

        dbc.Col([

            html.H3("Select Variable:",
                    className = 'text-left text-secondary mb-4')

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "left", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 13
    dcc.Dropdown(
        options = [],
        id = 'User_Input_TimePlot_Variable',
        multi=False
        ),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 12
    dbc.Row([

        dbc.Col([

            html.H3("Select Column / Columns within Variable:",
                    className='text-left text-secondary mb-4')

        ], xs=12, sm=12, md=12, lg=12, xl=12),  # width = 12

    ], justify="left", align="center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

        ], width=12),

    ]),

    # Row 14
    dcc.Dropdown(
        options = [],
        id = 'User_Input_TimePlot_Variable_Column',
        multi = True
    ),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

        ], width=12),

    ]),

    # Row 15
    dbc.Row([

        dbc.Col([

            html.Button('Plot', id = 'Button_TimePlot',
                        className = "btn btn-primary btn-lg col-12") ,

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "center", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 15
    dbc.Row([

        dbc.Col([

            dcc.Graph(id = 'Graph_TimePlot', figure ={}),

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "center", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 16
    dbc.Row([

        dbc.Col([

            html.H3("Scatter Plot:",
                    className = 'text-left text-secondary mb-4')

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "left", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 12
    dbc.Row([

        dbc.Col([

            html.H3("Select Variable X:",
                    className = 'text-left text-secondary mb-4')

            ], xs = 6, sm = 6, md = 6, lg = 6, xl = 6), # width = 12

        dbc.Col([

            html.H3("Select Variable Y:",
                    className='text-left text-secondary mb-4')

        ], xs=6, sm=6, md=6, lg=6, xl=6),  # width = 12

        ], justify = "left", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 13
    dbc.Row([
        dbc.Col([

            dcc.Dropdown(
                options=[],
                id='User_Input_ScatterPlot_Variable_X',
                multi=False
            ),
        ], xs=6, sm=6, md=6, lg=6, xl=6),  # width = 12

        dbc.Col([

            dcc.Dropdown(
                options=[],
                id='User_Input_ScatterPlot_Variable_Y',
                multi=False
            ),
        ], xs=6, sm=6, md=6, lg=6, xl=6),  # width = 12

    ],justify = "left", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 12
    dbc.Row([

        dbc.Col([

            html.H3("Select a Column within Variable X:",
                    className='text-left text-secondary mb-4')

        ], xs=6, sm=6, md=6, lg=6, xl=6),  # width = 12

        dbc.Col([

            html.H3("Select a Column within Variable Y:",
                    className='text-left text-secondary mb-4')

        ], xs=6, sm=6, md=6, lg=6, xl=6),  # width = 12

    ], justify="left", align="center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

        ], width=12),

    ]),

    # Row 13
    dbc.Row([
        dbc.Col([

            dcc.Dropdown(
                options=[],
                id='User_Input_ScatterPlot_Variable_X_Column',
                multi=False
            ),
        ], xs=6, sm=6, md=6, lg=6, xl=6),  # width = 12

        dbc.Col([

            dcc.Dropdown(
                options=[],
                id='User_Input_ScatterPlot_Variable_Y_Column',
                multi=False
            ),
        ], xs=6, sm=6, md=6, lg=6, xl=6),  # width = 12

    ], justify="left", align="center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

        ], width=12),

    ]),

    # Row 14
    dbc.Row([

        dbc.Col([

            html.Button('Plot', id = 'Button_ScatterPlot',
                        className = "btn btn-primary btn-lg col-12") ,

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "center", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),

    # Row 15
    dbc.Row([

        dbc.Col([

            dcc.Graph(id = 'Graph_ScatterPlot', figure ={}),

            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12

        ], justify = "center", align = "center"),

    # Break Row
    dbc.Row([

        dbc.Col([

            html.Br()

            ], width = 12),

        ]),
])


@app.callback(
    Output(component_id = 'Start_Date_Label', component_property = 'children'),
    Output(component_id = 'End_Date_Label', component_property = 'children'),
    Output(component_id = 'User_Input_Date_Range', component_property = 'figure'), # Need to check the property
    Output(component_id = 'User_Input_TimePlot_Variable', component_property = 'options'),
    Output(component_id = 'Uset_Input_ScatterPlot_Variable_X', component_property = 'options'),
    Output(component_id = 'User_Input_ScatterPlot_Variable_Y', component_property = 'options'),
    Input(component_id = 'User_Input_File', component_property = 'contents'),
)
def Input_File_Callback_Function(User_File):
    content_type, content_string = User_File.split(',')
    Decoded = base64.b64decode(content_string)

    IDF_OutputVariable_Dict = pickle.load(io.StringIO(Decoded.decode('utf-8')))

    return 0,0,0,0,0,0



    
# Running the App
 
if __name__ == '__main__': 
    app.run_server(port=4052)