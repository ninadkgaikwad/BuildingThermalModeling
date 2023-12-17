# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import copy
import math
import time

# =============================================================================
# User Inputs - Required File
# =============================================================================

# User Input: Simulation Name
Simulation_Name = "test1"


# User Input: Aggregation Unit Number
Total_Aggregation_Zone_Number = 1

# =============================================================================
# User Inputs - Plotting Options
# =============================================================================

Plot_FileType = '.png'

Title_FontSize = 20

Axis_FontSize = 16

Label_FontSize = 12

LegendPosition = 'best' # 'upper left', 'upper right', 'lower left', 'lower right' 'upper center', 'lower center', 'center left', 'center right'

XTick_RotationDeg = 30

SaveFigures_Indicator = 1

# =============================================================================
# User Inputs - DateTime X Axis Plotting
# =============================================================================

# Creating Figure
ax = plt.gca()

# Formatting X axis DateTime Ticks
# ax.xaxis.set_major_locator(matplotlib.dates.HourLocator((12,23)))
ax.xaxis.set_major_locator(matplotlib.dates.DayLocator((1,2,3,4,5,6,7)))
# ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d/%Y'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%Y'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d - %H:%M'))


# =============================================================================
# Getting Required Data from Sim_ProcessedData
# =============================================================================

# Getting Current File Directory Path
Current_FilePath = os.path.dirname(__file__)

# Getting Raw Data
Sim_ProcessedData_FolderPath = os.path.join(Current_FilePath,  '..',  '..', 'Results', 'Processed_BuildingSim_Data', Simulation_Name, 'Sim_ProcessedData')

IDF_OutputVariable_Dict_file = open(os.path.join(Sim_ProcessedData_FolderPath,'IDF_OutputVariables_DictDF.pickle'),"rb")

IDF_OutputVariable_Dict = pickle.load(IDF_OutputVariable_Dict_file)


# Getting Aggregated Data
Sim_AggregatedData_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath, 'Sim_AggregatedData')

Aggregated_Dict_File = open(os.path.join(Sim_AggregatedData_FolderPath, Aggregation_File_Name), "rb")

Aggregated_Dict = pickle.load(Aggregated_Dict_File)



# Getting Aggregated Train and Test Data Folder Path
Sim_ProcessedData_FolderPath_AggregatedTestTrain = os.path.join(Current_FilePath, '..', '..', 'Results', 'Processed_BuildingSim_Data', Simulation_Name, 'Sim_TrainingTestingData')



# Getting Regression Train and Test Data Folder Path
Sim_ProcessedData_FolderPath_Regression = os.path.join(Current_FilePath, '..', '..', 'Results', 'Processed_BuildingSim_Data', Simulation_Name, 'Sim_RegressionModelData')

# Getting ANN Output Data Folder Path
Sim_ProcessedData_FolderPath_ANN = os.path.join(Current_FilePath, '..', '..', 'Results', 'Processed_BuildingSim_Data', Simulation_Name, 'Sim_ANNModelData')




# =============================================================================
# Figure 1: Actual Temperature Plot from Raw and Aggregated Data
# =============================================================================




# =============================================================================
# Figure 2: Regression Model Scatter Plot
# =============================================================================




# =============================================================================
# Figure 3: Predicted Temperature Plot from Predicted Data
# =============================================================================




# =============================================================================
# Figure 4: Predicted PHVAC Plot from Predicted Data
# =============================================================================




# =============================================================================
# Table 1: Error Table
# =============================================================================