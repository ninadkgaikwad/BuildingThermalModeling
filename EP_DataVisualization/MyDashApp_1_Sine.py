# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:06:14 2022

@author: ninad
"""

# Importing Required Modules
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# Importing User-Defined Modules
import MyDashApp_Module as AppFuncs

# Instantiate our App and incorporate BOOTSTRAP theme Stylesheet
# Themes - https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/#available-themes
# Themes - https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/explorer/
# hackerthemes.com/bootstrap-cheatsheet/

app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

# App Layout using Dash Bootstrap

app.layout = dbc.Container([
    
    # Row 1
    dbc.Row([
        
        dbc.Col([
            
            html.H1("Sine Plotter-Calculator", 
                    className = 'text-center text-primary mb-4')
            
            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12
        
        ], justify = "center", align = "center"),
    
    # Row 2
    dbc.Row([
        
        dbc.Col([
            
            html.H3("Simulation Parameters:",
                    className = 'text-left text-secondary mb-4')
            
            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12     
        
        ], justify = "left", align = "center"),
    
    # Row 3
    dbc.Row([
        
        dbc.Col([
            
            html.Label("Time Duration:",
                    className = 'text-left text-secondary mb-4'), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12         
        
        dbc.Col([
            
            dcc.Input(
                id = "TimeDuration",
                type = "text",
                placeholder = "Input the duration of simulation in seconds",
            ),  
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12         
        
        dbc.Col([
            
            html.Label("Time Step:"),  
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12           
        
        dbc.Col([
            
            dcc.Input(
                id = "TimeStep",
                type = "text",
                placeholder = "Input time step of simulation in seconds",
            ),  
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12            
        
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
            
            html.H3("Sine Parameters:",
                    className = 'text-left text-secondary mb-4')
            
            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12      
        
        ], justify = "left", align = "center"), 
    
    
    # Row 5
    dbc.Row([
        
        dbc.Col([
            
            html.Label("Sine 1 - Amplitude:", className = "bg-white border border-danger"), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12            
        
        dbc.Col([
            
            dcc.Input(
                id = "A1",
                type = "text",
                placeholder = "Input the amplitude",
                className = "text-light bg-dark"
            ), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12           
        
        dbc.Col([
            
            html.Label("Sine 2 - Amplitude:"), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12           
        
        dbc.Col([
            
            dcc.Input(
                id = "A2",
                type = "text",
                placeholder = "Input the amplitude",
            ), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12             
        
        ], justify = "center", align = "center"),    
    
    # Break Row
    dbc.Row([
        
        dbc.Col([
            
            html.Br()
            
            ], width = 12),
        
        ]),    
    
    # Row 6
    dbc.Row([
        
        dbc.Col([
            
            html.Label("Sine 1 - Frequency:"), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12            
        
        dbc.Col([
            
            dcc.Input(
                id = "F1",
                type = "text",
                placeholder = "Input the frequency in cycles/sec",
            ), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12           
        
        dbc.Col([
            
            html.Label("Sine 2 - Frequency:"), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12           
        
        dbc.Col([
            
            dcc.Input(
                id = "F2",
                type = "text",
                placeholder = "Input the frequency in cycles/sec",  
            ), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12              
        
        ], justify = "center", align = "center"),     
    
    # Break Row
    dbc.Row([
        
        dbc.Col([
            
            html.Br()
            
            ], width = 12),
        
        ]),    
    
    
    # Row 7
    dbc.Row([
        
        dbc.Col([
            
            html.Label("Sine 1 - Phase:"), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12          
        
        dbc.Col([
            
            dcc.Input(
                id = "P1",
                type = "text",
                placeholder = "Input phase in degrees",
            ), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12           
        
        dbc.Col([
            
            html.Label("Sine 2 - Phase:"), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12           
        
        dbc.Col([
            
            dcc.Input(
                id = "P2",
                type = "text",
                placeholder = "Input phase in degrees",
            ), 
            
            ], xs = 6, sm = 6, md = 6, lg = 3, xl = 3), # width = 12              
        
        ], justify = "center", align = "center"),   
    
    # Break Row
    dbc.Row([
        
        dbc.Col([
            
            html.Br()
            
            ], width = 12),
        
        ]),      
    
    
    # Row 8
    dbc.Row([
        
        dbc.Col([
            
            html.H3("Select Operation:",
                    className = 'text-left text-secondary mb-4')
            
            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12      
        
        ], justify = "left", align = "center"),     
    
    # Break Row
    dbc.Row([
        
        dbc.Col([
            
            html.Br()
            
            ], width = 12),
        
        ]),  

    # Row 9
    dbc.Row([
        
        dbc.Col([
            
            dcc.RadioItems(
                    id = 'RadioItem1',
                    labelStyle = {'display': 'block'},
                    options = [
                        {'label' : "Add the Sine Waves", 'value' : 1},
                        {'label' : "Subtract Sine Waves", 'value' : 2},
                        {'label' : "Multiply Sine Waves", 'value' : 3}
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

    # Row 8
    dbc.Row([
        
        dbc.Col([
            
            html.H3("Sine Graph:",
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
            
            html.Button('Compute', id = 'Button_1', 
                        className = "btn btn-primary btn-lg col-12") ,
            
            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12
        
        ], justify = "center", align = "center"),   

    # Break Row
    dbc.Row([
        
        dbc.Col([
            
            html.Br()
            
            ], width = 12),
        
        ]),       

    # Row 11
    dbc.Row([
        
        dbc.Col([
            
            dcc.Graph(id = 'SineGraph', figure ={}),
            
            ], xs = 12, sm = 12, md = 12, lg = 12, xl = 12), # width = 12
        
        ], justify = "center", align = "center"),    
    
    
], fluid = False)


# App Callbacks - Providing Functionality

@app.callback(    
    Output(component_id = 'SineGraph', component_property = 'figure'),
    Input(component_id = 'Button_1', component_property = 'n_clicks'),
    State(component_id = 'RadioItem1', component_property = 'value'),
    State(component_id = 'TimeDuration', component_property = 'value'),
    State(component_id = 'TimeStep', component_property = 'value'),
    State(component_id = 'A1', component_property = 'value'),
    State(component_id = 'A2', component_property = 'value'),
    State(component_id = 'F1', component_property = 'value'),
    State(component_id = 'F2', component_property = 'value'),
    State(component_id = 'P1', component_property = 'value'),
    State(component_id = 'P2', component_property = 'value'),  
    prevent_initial_call = False)
def CreateSineGraph(N_Clicks_B1, Computation_Option, TimeDuration, TimeStep, A1, A2, F1, F2, P1, P2):
    
    # Converting text to numbers
    TimeDuration =float(TimeDuration)
    TimeStep =float(TimeStep)
    A1 =float(A1)
    A2 =float(A2)
    F1 =float(F1)
    F2 =float(F2)
    P1 =float(P1)
    P2 =float(P2)
    
    # Create Time Vector
    TimeVector = AppFuncs.CreateTimeVector(TimeDuration, TimeStep)
    
    # Create the Two Sines
    Sine1 = AppFuncs.CreateSine(TimeVector, A1, F1, P1)
    
    Sine2 = AppFuncs.CreateSine(TimeVector, A2, F2, P2)
    
    # Compute with Sines
    Sines_DF = AppFuncs.Compute_with_Sines(TimeVector, Sine1, Sine2, Computation_Option)
    
    # Plotting Sines
    fig = px.line(Sines_DF, x="Time", y=["Sine_1","Sine_2","Sine_New"], labels={"Time": "Time (sec)", "Sine_1": "Sine Wave 1", "Sine_2": "Sine Wave 2", "Sine_New": "Sine Wave New"}, title = 'Sine Wave Graph')
    
    return fig
    
# Running the App
 
if __name__ == '__main__': 
    app.run_server(port=4051)