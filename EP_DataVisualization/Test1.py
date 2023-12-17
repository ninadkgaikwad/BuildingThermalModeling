# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 11:11:29 2022

@author: ninad
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

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
    
    html.Div(children = 'Dash- A Data Product framewwork from Plotly',
            style = {
                'textAlign' : 'center',
                'color' : colors['Text_Color']
                }
             ),
    
    get_bar_chart(),
    get_scatter_chart(),   
    
    ])

if __name__ == '__main__': 
    app.run_server(port=4050)