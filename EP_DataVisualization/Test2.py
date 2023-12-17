# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 11:11:29 2022

@author: ninad
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from datetime import datetime as dt

# Colors
colors = { 'Text_Color' : "#567ABC",
           'Plot_Color' : '#D3D3D3',
           'Paper_Color': '#D3D3D3'
    }

# Graph Functions
def get_bar_chart():
   barChart =  dcc.Graph(
            id = 'sample-graph',
            figure = {
                'data' : [
                    {'x': [1,2,3,4,5], 'y' : [1,4,9,16,25], 'type' : 'bar', 'name': 'First Chart'},
                    {'x': [1,2,3,4,5], 'y' : [1,2,3,4,5], 'type' : 'bar', 'name': 'First Chart'}
                    ],
                      'layout' : { 
                          'title':"Simple Bar Chart",
                          'plot_bgcolor' : colors['Plot_Color'],
                          'paper_bgcolor' : colors['Paper_Color'],
                          'font' : {
                              'color' : colors['Text_Color']
                              }
                        
                          }
                      }
            )
   return barChart

def get_scatter_chart():
    scatterChart = dcc.Graph(
        id = 'Scatter_Plot',
        figure = {
            'data' : [
                
                go.Scatter(
                    x = [1,2,3,4,5],
                    y = [1,4,9,16,25],
                    mode = 'markers'
                    )
                
                ],
                  'layout' : go.Layout(
                      title = 'Scatter Plot',
                      xaxis = {'title': "X - Axis"},
                      yaxis = {'title': "Y - Axis"}
                      )
                  }
        )
    return scatterChart

# App Code
app= dash.Dash()
app.layout= html.Div([
    
    html.H1(children = 'Hello Dash',
            style = {
                'textAlign' : 'center',
                'color' : "#567ABC"
                }),
    html.Br(),
    
    html.Div(children = 'Dash- A Data Product framewwork from Plotly',
            style = {
                'textAlign' : 'center',
                'color' : colors['Text_Color']
                }
             ),
    
    html.Br(),
    
    html.Label('Choose a City'),
    
    html.Br(),
    
    dcc.Dropdown(
        id = 'first-dropdown',
        options = [
            {'label' : "San Francisco", 'value' : 'SF'},
            {'label' : "NewYork City", 'value' : 'NYC'},
            {'label' : "Raleigh Durham", 'value' : 'RDU', 'disabled' : True}
            ],
        value = 'NYC',
        placeholder = 'Choose a City',
        multi = True,
        # disabled = True
        
        ),
    
 html.Br(),
 
 html.Label("Slider"),
 
 dcc.Slider(
     id = 'first-slider',
     min = 1,
     max = 10,
     value = 5,
     marks = {i: i for i in range(10)},
     step = 0.5
     
     ),
 
 html.Br(),
 
  html.Label("Range Slider"),
 
 dcc.RangeSlider(
     id = 'first-range-slider',
     min = 1,
     max = 10,
     value = [3,7],
     marks = {i: i for i in range(10)},
     step = 0.5
     
     ),
 
 html.Br(),
 
 html.Label("Input Box"),
 
 dcc.Input(
     id = 'Input1',
     placeholder = "Input your name",
     type = 'text',
     value = ''
     ),
 
 html.Br(),
  html.Br(),
  
  dcc.Input(
      id = 'TextArea1',
     placeholder = "Input your feedback",
     value = 'Placeholder for text',
     style = {'width': '50%'}
     ),  
  
 html.Br(),
  html.Br(),  
    
  dcc.Checklist(
        id = 'first-checklist',
        options = [
            {'label' : "San Francisco", 'value' : 'SF'},
            {'label' : "NewYork City", 'value' : 'NYC'},
            {'label' : "Raleigh Durham", 'value' : 'RDU', 'disabled' : True}
            ],
        value = ['NYC','NYC']     
      ),
    
 html.Br(),
  html.Br(),  
 
dcc.RadioItems(
        id = 'first-radioitems',
        options = [
            {'label' : "San Francisco", 'value' : 'SF'},
            {'label' : "NewYork City", 'value' : 'NYC'},
            {'label' : "Raleigh Durham", 'value' : 'RDU', 'disabled' : True}
            ]  ,
         value = 'NYC'
    ),
   
 html.Br(),
  html.Br(), 
  
html.Button('Submit', id = 'Button_1')  ,
  
 html.Br(),
  html.Br(),    

dcc.DatePickerSingle(
    id = 'dt-pick-single',
    date = dt(2022,10,10)
    )    ,
  
 html.Br(),
  html.Br(), 
  
 html.Br(),
  html.Br(),    

dcc.DatePickerRange(
    id = 'dt-pick-range',
    start_date = dt(2022,10,10),
    end_date_placeholder_text = 'Choose end date'
    )    ,
  
 html.Br(),
  html.Br(),  
  
dcc.Markdown(
    '''
    # Dash Supports Markdown    
    '''
    )  ,
  
 html.Br(),
  html.Br(),    
    
 dcc.Graph(
            id = 'sample-graph1',
            figure = {
                'data' : [
                    {'x': [1,2,3,4,5], 'y' : [1,4,9,16,25], 'type' : 'bar', 'name': 'First Chart'},
                    {'x': [1,2,3,4,5], 'y' : [1,2,3,4,5], 'type' : 'bar', 'name': 'First Chart'}
                    ],
                      'layout' : { 
                          'title':"Simple Bar Chart",
                          'plot_bgcolor' : colors['Plot_Color'],
                          'paper_bgcolor' : colors['Paper_Color'],
                          'font' : {
                              'color' : colors['Text_Color']
                              }
                        
                          }
                      }
            ),

html.Br(),
 
dcc.Graph(
        id = 'Scatter_Plot',
        figure = {
            'data' : [
                
                go.Scatter(
                    x = [1,2,3,4,5],
                    y = [1,4,9,16,25],
                    mode = 'markers'
                    )
                
                ],
                  'layout' : go.Layout(
                      title = 'Scatter Plot',
                      xaxis = {'title': "X - Axis"},
                      yaxis = {'title': "Y - Axis"}
                      )
                  }
        ),   
    
    ])

if __name__ == '__main__': 
    app.run_server(port=4050)