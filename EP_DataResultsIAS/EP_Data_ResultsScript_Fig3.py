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

Sim_ANNModelData_Case_Folder_Name = 'Ind_1Z_100EP_25B_100REG_Bias'

Total_Aggregation_Zone_Number = 1


# =============================================================================
# User Inputs - Plotting Options
# =============================================================================

Plot_FileType = '.png'

Title_FontSize = 20

Axis_FontSize = 16

Label_FontSize = 10

LegendPosition = 'best'  # 'upper left', 'upper right', 'lower left', 'lower right' 'upper center', 'lower center', 'center left', 'center right'

XTick_RotationDeg = 30

SaveFigures_Indicator = 1


# =============================================================================
# User Inputs - DateTime X Axis Plotting
# =============================================================================

# Creating Figure
ax = plt.gca()

# Formatting X axis DateTime Ticks
# ax.xaxis.set_major_locator(matplotlib.dates.HourLocator((12,23)))
# ax.xaxis.set_major_locator(matplotlib.dates.DayLocator((1, 2, 3, 4, 5, 6, 7)))
# ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d/%Y'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%Y'))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d - %H:%M'))


# =============================================================================
# User Interactive Dynamic Plotting
# =============================================================================

## Create Sim_DataExploration Folder to save Figures ##

# Getting Current File Directory Path
Current_FilePath = os.path.dirname(__file__)

# Getting desired Simulation Folder Path
Processed_BuildingSim_Simulation_FolderPath = os.path.join(Current_FilePath, '..', '..', 'Results','Processed_BuildingSim_Data', Simulation_Name)

# Create Folder to save Figures depending on SaveFigures_Indicator
if (SaveFigures_Indicator == 1):

    # New Folder name for saving Figures
    Sim_Results_FolderName = 'Sim_Results_IAS'

    # Getting Folder Path for saving Figures
    Sim_Results_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath,Sim_Results_FolderName)

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

# Getting Sim_AggregatedData_FolderPath
Sim_AggregatedData_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath, 'Sim_AggregatedData')

# Getting Sim_ANNModelData_FolderPath
Sim_ANNModelData_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath, 'Sim_ANNModelData')

Sim_ANNModelData_Case_FolderPath = os.path.join(Sim_ANNModelData_FolderPath, Sim_ANNModelData_Case_Folder_Name)

# Getting Predict_Actual_Y_DF_withoutSim_File
Predict_Actual_Y_DF_withoutSim_File = open(os.path.join(Sim_ANNModelData_Case_FolderPath, 'Predict_Actual_Y_DF_1Zone_1_withoutSim.pickle'),
                                    "rb")

Predict_Actual_Y_DF_withoutSim = pickle.load(Predict_Actual_Y_DF_withoutSim_File)

# Getting Predict_Actual_Y_DF_withSim_File
Predict_Actual_Y_DF_withSim_File = open(os.path.join(Sim_ANNModelData_Case_FolderPath, 'Predict_Actual_Y_DF_1Zone_1_withSim.pickle'),
                                    "rb")

Predict_Actual_Y_DF_withSim = pickle.load(Predict_Actual_Y_DF_withSim_File)

# Getting DF_OutputVariablesFull_DictDF.mat containing all Data and DateTime
Aggregated_Dict_File_Name = 'Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.pickle'

Aggregated_Dict_File = open(os.path.join(Sim_AggregatedData_FolderPath, Aggregated_Dict_File_Name), "rb")

Aggregated_Dict = pickle.load(Aggregated_Dict_File)


# Getting DateTime List from IDF_OutputVariables_DictDF
DateTime_List = Aggregated_Dict['DateTime_List']

Duration = datetime.timedelta(days=1)

# FOR LOOP: Correcting DateTime_List for 24th Hour Error
for ii in range(len(DateTime_List)):
    DT = DateTime_List[ii]
    if DT.hour == 0 and DT.minute == 0:
        DT1 = datetime.datetime(DT.year, DT.month, DT.day, 0, 0, 0) + Duration
        DateTime_List[ii] = DT1

# Getting Start and End Dates for the Dataset
StartDate_Dataset = DateTime_List[0]
EndDate_Dataset = DateTime_List[-1]

# Getting the File Resolution from DateTime_List
DateTime_Delta = DateTime_List[1] - DateTime_List[0]

FileResolution_Minutes = DateTime_Delta.seconds / 60


## User Select: Date Range ##
StartDate_User = input('Enter Start Date as m/d/y : Dataset range is ' + StartDate_Dataset.strftime(
    "%m/%d/%Y") + ' to ' + EndDate_Dataset.strftime("%m/%d/%Y") + ' : ')
EndDate_User = input('Enter End Date as m/d/y : Dataset range is ' + StartDate_Dataset.strftime(
    "%m/%d/%Y") + ' to ' + EndDate_Dataset.strftime("%m/%d/%Y") + ' : ')

# Convert User Date Strings to DateTime Objects
StartDate_User = datetime.datetime.strptime(StartDate_User, '%m/%d/%Y')
EndDate_User = datetime.datetime.strptime(EndDate_User, '%m/%d/%Y')

# User Dates Corrected
StartDate_User_Corrected = datetime.datetime(StartDate_User.year, StartDate_User.month, StartDate_User.day, 0,
                                             int(FileResolution_Minutes), 0)
EndDate_User_Corrected = datetime.datetime(EndDate_User.year, EndDate_User.month, EndDate_User.day, 23,
                                           60 - int(FileResolution_Minutes), 0)

# Getting Start/End Datetime Indices
Start_DateTime_Index = DateTime_List.index(StartDate_User_Corrected)
End_DateTime_Index = DateTime_List.index(EndDate_User_Corrected) + 1

# Creating DateTime List to Plot
# DateTime_List_Plot = matplotlib.dates.date2num(DateTime_List[Start_DateTime_Index:End_DateTime_Index])
DateTime_List_Plot = DateTime_List[Start_DateTime_Index:End_DateTime_Index]

# Time Series Plots

# Get the Actual Temperature Data
Actual_Temperature = Predict_Actual_Y_DF_withoutSim.iloc[:,1]
Actual_Temperature_Plot = Actual_Temperature.iloc[0:2015]

# Get the Predicted Temperature Data
Predicted_Temperature_withoutSim = Predict_Actual_Y_DF_withoutSim.iloc[:,0]
Predicted_Temperature_withoutSim_Plot = Predict_Actual_Y_DF_withoutSim.iloc[0:2015]

Predicted_Temperature_withSim = Predict_Actual_Y_DF_withSim.iloc[:,0]
Predicted_Temperature_withSim_Plot = Predict_Actual_Y_DF_withSim.iloc[0:2015]

# Plotting Data

X_Axis_Data = DateTime_List_Plot
Y_Axis_Data_List = [Actual_Temperature_Plot, Predicted_Temperature_withSim_Plot.iloc[:,0]]
Y_Axis_Data = pd.concat(Y_Axis_Data_List, axis=1)
Y_Axis_Data.rename(columns={'Actual_Y':'Actual Temperature', 'Predict_Y_BiasNotAccounted': 'Predicted Temperature'},inplace = True)
Y_Axis_Data_Columns = Y_Axis_Data.columns

Plot_Marker_Type = ['v', 'o', 's', 'd', '*']
Marker_Gap = [101,347,45,247,177]
Color = ['k','r','b']

for ii in range(len(Y_Axis_Data_Columns)):

    if (ii == 0):

        plt.plot(X_Axis_Data, Y_Axis_Data[Y_Axis_Data_Columns[ii]], color='black')

    else:

        plt.plot(X_Axis_Data, Y_Axis_Data[Y_Axis_Data_Columns[ii]], linestyle='dashed', dashes=(5, 2), color=Color[ii], marker=Plot_Marker_Type[ii], markevery=Marker_Gap[ii], fillstyle = 'none')


# Plot Embellishments
plt.xlabel('$Time$', fontsize=Axis_FontSize)
plt.ylabel('$Zone \: Air \: Temperature \: (\N{DEGREE SIGN}C)$', fontsize=Axis_FontSize)
plt.xticks(rotation=XTick_RotationDeg, fontsize=Axis_FontSize)
plt.yticks(fontsize=Axis_FontSize)
plt.legend(Y_Axis_Data_Columns, loc=LegendPosition, fontsize=Label_FontSize, borderaxespad=0)
plt.tight_layout()
plt.grid()

# Saving Figure dependent on SaveFigures_Indicator

# Creating File Path
Plot_File_Path = os.path.join(Sim_Results_FolderPath)

# Saving Plot
Plot_Name = 'PredictedActualTemperature_TimeSeriesPlot_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_TestJuly' + '.png'
plt.savefig(os.path.join(Plot_File_Path, Plot_Name), dpi=300)

# Showing Figure in Console
plt.show() #plot.show() must be called after plt.savefig, otherwise blank image will be saved


