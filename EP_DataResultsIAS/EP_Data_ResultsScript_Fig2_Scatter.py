# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import pandas as pd
import scipy.io
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib


# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Total_Aggregation_Zone_Number = 1

Training_RegressionModel_File_Name = 'TrainingData_RegressionModel_Dict_Aggregation_Dict_1Zone_1.pickle'

ANN_HeatInput_Train_DF_File_Name = 'ANN_HeatInput_Train_DF_1Zone_1.pickle'


# =============================================================================
# User Inputs - Plotting Options
# =============================================================================

Plot_FileType = '.png'

Title_FontSize = 20

Axis_FontSize = 16

Label_FontSize = 10

Marker_Gap = 100

LegendPosition = 'best'  # 'upper left', 'upper right', 'lower left', 'lower right' 'upper center', 'lower center', 'center left', 'center right'

XTick_RotationDeg = 30

SaveFigures_Indicator = 1


# =============================================================================
# User Inputs - DateTime X Axis Plotting
# =============================================================================

# Creating Figure
# ax = plt.gca()

# Formatting X axis DateTime Ticks
# ax.xaxis.set_major_locator(matplotlib.dates.HourLocator((12,23)))
# ax.xaxis.set_major_locator(matplotlib.dates.DayLocator((1, 2, 3, 4, 5, 6, 7)))
# ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d/%Y'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%Y'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d - %H:%M'))


# =============================================================================
# User Interactive Dynamic Plotting
# =============================================================================

## Create Sim_DataExploration Folder to save Figures ##

# Getting Current File Directory Path
Current_FilePath = os.path.dirname(__file__)

# Getting desired Simulation Folder Path
Processed_BuildingSim_Simulation_FolderPath = os.path.join(Current_FilePath, '..', '..', 'Results',
                                                           'Processed_BuildingSim_Data', Simulation_Name)

# Create Folder to save Figures depending on SaveFigures_Indicator
if (SaveFigures_Indicator == 1):

    # New Folder name for saving Figures
    Sim_Results_FolderName = 'Sim_Results_IAS'

    # Getting Folder Path for saving Figures
    Sim_Results_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath,
                                                     Sim_Results_FolderName)

    # Checking if Folders Exist if not create Folders
    if (os.path.isdir(Sim_Results_FolderPath)):

        # Folders Exist
        z = None

    else:

        os.mkdir(Sim_Results_FolderPath)

else:

    # Do not save Figures
    z = None

## Loading the Data File ##

# Getting Sim_ProcessedData_FolderPath
Sim_ProcessedData_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath, 'Sim_ProcessedData')

# Getting Sim_RegressionData_FolderPath
Sim_TrainingTestingData_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath, 'Sim_TrainingTestingData')

# Getting the Regression Data
TrainingData_RegressionModel_Dict_File = open(os.path.join(Sim_TrainingTestingData_FolderPath, Training_RegressionModel_File_Name), "rb")
TrainingData_RegressionModel_Dict = pickle.load(TrainingData_RegressionModel_Dict_File)

# =============================================================================
# Scatter Plot: QSol1 and GHI
# =============================================================================

# Getting the Values
Variable1_DF = (TrainingData_RegressionModel_Dict['QSol1'].iloc[:,0]).to_frame() # GHI
Variable2_DF = (TrainingData_RegressionModel_Dict['QSol1'].iloc[:,1]).to_frame() # QSol1

#Variable2_DF['QSol1'] = Variable2_DF['QSol1'].str.get(0)


# Plotting
plt.scatter(Variable1_DF.abs(), Variable2_DF)

# Plot Embellishments
plt.xlabel('$GHI \: (W/m^2)$', fontsize=Axis_FontSize, multialignment='center')
plt.ylabel('$Q_{SOL1} \: (W)$', fontsize=Axis_FontSize, multialignment='center')

plt.xticks(fontsize=Axis_FontSize)
plt.yticks(fontsize=Axis_FontSize)

plt.tight_layout()

plt.grid()

# Saving Figure dependent on SaveFigures_Indicator
# Creating File Path
Plot_File_Path = os.path.join(Sim_Results_FolderPath)

# Saving Plot
plt.savefig(os.path.join(Plot_File_Path, 'GHI_QSol1' + '_ScatterPlot' + '.png'), dpi=300)

# Showing Figure in Console
plt.show()



# =============================================================================
# Scatter Plot: QZir_P and Schedule_People
# =============================================================================

# Getting the Values
Variable1_DF = (TrainingData_RegressionModel_Dict['QZir_P'].iloc[:,0]).to_frame() # Schedule_People
Variable2_DF = (TrainingData_RegressionModel_Dict['QZir_P'].iloc[:,1]).to_frame() # QZir_P

# Plotting
plt.scatter(Variable1_DF, Variable2_DF)

# Plot Embellishments
plt.xlabel('$People \: Schedule$', fontsize=Axis_FontSize, multialignment='center')
plt.ylabel('$Q_{IR}^P  \: (W)$', fontsize=Axis_FontSize, multialignment='center')

plt.xticks(fontsize=Axis_FontSize)
plt.yticks(fontsize=Axis_FontSize)

plt.tight_layout()

plt.grid()

# Saving Figure dependent on SaveFigures_Indicator
# Creating File Path
Plot_File_Path = os.path.join(Sim_Results_FolderPath)

# Saving Plot
plt.savefig(os.path.join(Plot_File_Path, 'QZirP_SchedulePeople' + '_ScatterPlot' + '.png'), dpi=300)

# Showing Figure in Console
plt.show()



# =============================================================================
# Scatter Plot: QZir_P and Schedule_People
# =============================================================================

# Getting the Values
Variable1_DF = (TrainingData_RegressionModel_Dict['QAC'].iloc[:,0]).to_frame() # QHVAC
Variable1_DF = Variable1_DF.abs() # |QHVAC|
Variable2_DF = (TrainingData_RegressionModel_Dict['QAC'].iloc[:,1]).to_frame() # PHVAC

# Plotting
plt.scatter(Variable1_DF, Variable2_DF)

# Plot Embellishments
plt.xlabel('$|Q_{HVAC}|  \: (W)$', fontsize=Axis_FontSize, multialignment='center')
plt.ylabel('$P_{HVAC}  \: (W)$', fontsize=Axis_FontSize, multialignment='center')

plt.xticks(fontsize=Axis_FontSize)
plt.yticks(fontsize=Axis_FontSize)

plt.tight_layout()

plt.grid()

# Saving Figure dependent on SaveFigures_Indicator
# Creating File Path
Plot_File_Path = os.path.join(Sim_Results_FolderPath)

# Saving Plot
plt.savefig(os.path.join(Plot_File_Path, 'PHVAC_QHVAC' + '_ScatterPlot' + '.png'), dpi=300)

# Showing Figure in Console
plt.show()