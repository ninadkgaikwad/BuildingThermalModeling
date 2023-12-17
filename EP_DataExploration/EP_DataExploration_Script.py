# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:55:35 2022

@author: ninad gaikwad 
"""

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

# Custom Modules

# Setting Latex Rendering for MAtplotlib depending on Latex_Indicator
'''
if (Latex_Indicator):

    plt.rcParams.update({
        "text.usetex": Latex_Indicator,
        "font.family": Latex_FontFamily,
        "font.sans-serif": [Latex_FontType]})
    
else:

    plt.rcParams.update({
        "text.usetex": Latex_Indicator}) 
    
'''
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
# User Interactive Dynamic Plotting
# =============================================================================

# User Selection: Raw or Aggregated Data File
Raw_Aggregated_DataFile_Option = input('Enter selection, 1 - Raw Data File , 2 - Aggregated Data File: ')

Raw_Aggregated_DataFile_Option = int(Raw_Aggregated_DataFile_Option)

# User Provides: Appropriate Folder Name
Simulation_Name = input('Enter Appropriate Simulation Folder Name: ')

## Create Sim_DataExploration Folder to save Figures ##

# Getting Current File Directory Path
Current_FilePath = os.path.dirname(__file__)

# Getting desired Simulation Folder Path
Processed_BuildingSim_Simulation_FolderPath = os.path.join(Current_FilePath,  '..',  '..', 'Results', 'Processed_BuildingSim_Data', Simulation_Name)

# Create Folder to save Figures depending on SaveFigures_Indicator
if (SaveFigures_Indicator == 1):
        
    # New Folder name for saving Figures
    Sim_IDFDataExploration_FolderName = 'Sim_DataExploration'
    
    # Getting Folder Path for saving Figures
    Sim_IDFDataExploration_FolderPath =  os.path.join(Processed_BuildingSim_Simulation_FolderPath, Sim_IDFDataExploration_FolderName)
    
    # Checking if Folders Exist if not create Folders
    if (os.path.isdir(Sim_IDFDataExploration_FolderPath)):
    
        # Folders Exist    
        z = None
        
    else:
        
        os.mkdir(Sim_IDFDataExploration_FolderPath)    
    
else:

    # Do not save Figures    
    z = None 

## IF ELSE LOOP: For Raw or Aggregated Data File Plotting ##
if (Raw_Aggregated_DataFile_Option == 1): # Raw File

    ## Loading the Data File ##

    # Getting Sim_ProcessedData_FolderPath
    Sim_ProcessedData_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath, 'Sim_ProcessedData') 
    
    # Getting IDF_OutputVariables_DictDF.pickle containing all Data and DateTime
    IDF_OutputVariable_Dict_File = open(os.path.join(Sim_ProcessedData_FolderPath,'IDF_OutputVariables_DictDF.pickle'),"rb")

    IDF_OutputVariable_Dict = pickle.load(IDF_OutputVariable_Dict_File)
    
    # Getting DateTime List from IDF_OutputVariables_DictDF
    DateTime_List = IDF_OutputVariable_Dict['DateTime_List']

    Duration = datetime.timedelta(days=1)

    #FOR LOOP: Correcting DateTime_List for 24th Hour Error
    for ii in range(len(DateTime_List)):
        DT = DateTime_List[ii]
        if DT.hour == 0 and DT.minute == 0:
            DT1 = datetime.datetime(DT.year,DT.month,DT.day,0,0,0) + Duration
            DateTime_List[ii] = DT1
    
    # Getting Start and End Dates for the Dataset
    StartDate_Dataset = DateTime_List[0]
    EndDate_Dataset = DateTime_List[-1]
    
    # Getting the File Resolution from DateTime_List
    DateTime_Delta = DateTime_List[1] - DateTime_List[0]

    FileResolution_Minutes = DateTime_Delta.seconds/60 
    
    ## User Input: Type of Graph ##
    TypeOfGraph = input('Select Type of Graph, 1 - Time Series , 2 - Scatter Plot: ')
    
    TypeOfGraph = int(TypeOfGraph)
    
    ## User Select: Date Range ##
    StartDate_User = input('Enter Start Date as m/d/y : Dataset range is '+ StartDate_Dataset.strftime("%m/%d/%Y") + ' to ' + EndDate_Dataset.strftime("%m/%d/%Y") + ' : ')
    EndDate_User = input('Enter End Date as m/d/y : Dataset range is '+ StartDate_Dataset.strftime("%m/%d/%Y") + ' to ' + EndDate_Dataset.strftime("%m/%d/%Y") + ' : ')

    # Convert User Date Strings to DateTime Objects
    StartDate_User = datetime.datetime.strptime(StartDate_User,'%m/%d/%Y')
    EndDate_User = datetime.datetime.strptime(EndDate_User,'%m/%d/%Y')
    
    # User Dates Corrected
    StartDate_User_Corrected = datetime.datetime(StartDate_User.year,StartDate_User.month,StartDate_User.day,0,int(FileResolution_Minutes),0)
    EndDate_User_Corrected = datetime.datetime(EndDate_User.year,EndDate_User.month,EndDate_User.day,23,60-int(FileResolution_Minutes),0)

    # Getting Start/End Datetime Indices
    Start_DateTime_Index = DateTime_List.index(StartDate_User_Corrected)
    End_DateTime_Index = DateTime_List.index(EndDate_User_Corrected) + 1

    # Creating DateTime List to Plot
    # DateTime_List_Plot = matplotlib.dates.date2num(DateTime_List[Start_DateTime_Index:End_DateTime_Index])
    DateTime_List_Plot = DateTime_List[Start_DateTime_Index:End_DateTime_Index]

    # IF ELSE LOOP: Based on type of Graph
    if (TypeOfGraph == 1): # Time Series Plots
    
        # Print Level1 Variable Names
        Level1_VariableNames_List = IDF_OutputVariable_Dict.keys()
        
        print('\n')
        print('Level-1 Variable Names List:')
        print('\n')
        for VarName in Level1_VariableNames_List:
            print(VarName)
            
        print('\n')
        
        # User Selection: Select Level1 Variable 
        Level1_Variable_Name = input('Select ONE Level-1 Variable from above List: ')
        
        # Get the Level1 Variable
        Level1_Variable = IDF_OutputVariable_Dict[Level1_Variable_Name]
        
        # Print Level2 Variable Names
        Level2_VariableNames_List = Level1_Variable.columns

        print('\n')
        print('Level-2 Variable Names List:')
        print('\n')
        for VarName in Level2_VariableNames_List:
            print(VarName)
            
        print('\n')       
        
        # User Selection: Select Level2 Variable 
        Level2_Variable_Name = input('Select ONE or MORE Level-2 Variables from above List (Comma separated no spaces): ')        
    
        # Creating a List of Columns from
        Level2_VariableColumn_List = Level2_Variable_Name.split(',')
        
        # Getting required columns from Level1_Variable for Plotting
        Variables_Plot_DF = Level1_Variable[Level2_VariableColumn_List].iloc[Start_DateTime_Index:End_DateTime_Index,:]

        # Plotting Data

        plt.plot(DateTime_List_Plot, Variables_Plot_DF)

        # Plot Embellishments
        plt.title('Time Series: '+Level1_Variable_Name.replace('_',' '), fontsize=Title_FontSize)
        plt.xlabel('Time', fontsize=Axis_FontSize)
        plt.ylabel(Level1_Variable_Name.replace('_',' '), fontsize=Axis_FontSize)

        plt.xticks(rotation=XTick_RotationDeg)

        plt.legend(Level2_VariableColumn_List,loc=LegendPosition, fontsize=Label_FontSize, borderaxespad=0)

        plt.tight_layout()

        # Saving Figure dependent on SaveFigures_Indicator

        # Creating File Path
        Plot_File_Path = os.path.join(Sim_IDFDataExploration_FolderPath)

        # Saving Plot
        plt.savefig(os.path.join(Plot_File_Path, 'Raw_' + Level1_Variable_Name + '_TimeSeriesPlot' + '.png'), dpi=300)

        # Showing Figure in Console
        # plt.show()

    elif (TypeOfGraph == 2): # Scatter Plots
    
        # Initialize XY_Variable_List
        XY_Variable_List = []
        
        Level1_Variable_Name_List = []
        
        Level2_Variable_Name_List = []
        
        # FOR LOOP; For X and Y Variable Selection
        for ii in range(2): 
            
            # IF ELSE LOOP: For Variable_Stem
            if (ii==0):
                Variable_Stem = 'X'
            elif (ii == 1):
                Variable_Stem = 'Y'
            
            # Print Level1 Variable Names
            Level1_VariableNames_List = IDF_OutputVariable_Dict.keys()
            
            print('\n')
            print('Level-1 '+Variable_Stem+' Variable Names List:')
            print('\n')
            for VarName in Level1_VariableNames_List:
                print(VarName)
                
            print('\n')
            
            # User Selection: Select Level1 Variable 
            Level1_Variable_Name = input('Select ONE Level-1 '+ Variable_Stem +' Variable from above List: ')
            
            # Get the Level1 Variable
            Level1_Variable = IDF_OutputVariable_Dict[Level1_Variable_Name]
            
            # Print Level2 Variable Names
            Level2_VariableNames_List = Level1_Variable.columns
    
            print('\n')
            print('Level-2 '+Variable_Stem+' Variable Names List:')
            print('\n')
            for VarName in Level2_VariableNames_List:
                print(VarName)
                
            print('\n')       
            
            # User Selection: Select Level2 Variable 
            Level2_Variable_Name = input('Select ONE Level-2 '+ Variable_Stem +' Variables from above List: ')

            # Getting required columns from Level1_Variable for Plotting
            Variables_Plot_DF = Level1_Variable[Level2_Variable_Name].iloc[Start_DateTime_Index:End_DateTime_Index]
        
            # Filling up XY_Variable_List
            XY_Variable_List.append(Variables_Plot_DF)
            
            Level1_Variable_Name_List.append(Level1_Variable_Name)
            
            Level2_Variable_Name_List.append(Level2_Variable_Name)
         
        # Plotting
        plt.scatter(XY_Variable_List[0], XY_Variable_List[1])

        # Plot Embellishments
        plt.title('Scatter Plot: ', fontsize=Title_FontSize)
        plt.xlabel(Level2_Variable_Name_List[0].replace('_',' '), fontsize=Axis_FontSize)
        plt.ylabel(Level2_Variable_Name_List[1].replace('_',' '), fontsize=Axis_FontSize)

        plt.xticks(rotation=XTick_RotationDeg)

        plt.tight_layout()

        # Saving Figure dependent on SaveFigures_Indicator
        # Creating File Path
        Plot_File_Path = os.path.join(Sim_IDFDataExploration_FolderPath)

        # Saving Plot
        plt.savefig(os.path.join(Plot_File_Path, 'Raw_' + '_ScatterPlot' + '.png'), dpi=300)

        # Showing Figure in Console
        # plt.show()
    

elif (Raw_Aggregated_DataFile_Option == 2): # Aggregated File
    
    ## Loading the Data File ##

    # Getting Sim_AggregatedData_FolderPath
    Sim_AggregatedData_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath, 'Sim_AggregatedData') 
    
    # Getting DF_OutputVariablesFull_DictDF.mat containing all Data and DateTime
    Aggregated_Dict_File = open(os.path.join(Sim_AggregatedData_FolderPath,'Aggregation_Dict.pickle'),"rb")

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

    FileResolution_Minutes = DateTime_Delta.seconds/60 
    
    ## User Input: Aggregation Unit Number ##
    Aggregation_UnitNumber = input('Enter Aggregation Unit Number: ')

    Aggregation_UnitNumber = int(Aggregation_UnitNumber)

    # Creating the Correct key based on Aggregation_UnitNumber
    AggregationDf_Key = 'Aggregation_Zone_'+str(Aggregation_UnitNumber)
    
    # Getting appropriate Aggregation_DF based on AggregationDf_Key
    Aggregation_DF = Aggregated_Dict[AggregationDf_Key]
    
    ## User Input: Type of Graph ##
    TypeOfGraph = input('Select Type of Graph, 1 - Time Series , 2 - Scatter Plot: ')
    
    TypeOfGraph = int(TypeOfGraph)
    
    ## User Select: Date Range ##
    StartDate_User = input('Enter Start Date as m/d/y : Dataset range is '+ StartDate_Dataset.strftime("%m/%d/%Y") + ' to ' + EndDate_Dataset.strftime("%m/%d/%Y") + ' : ')
    EndDate_User = input('Enter End Date as m/d/y : Dataset range is '+ StartDate_Dataset.strftime("%m/%d/%Y") + ' to ' + EndDate_Dataset.strftime("%m/%d/%Y") + ' : ')

    # Convert User Date Strings to DateTime Objects
    StartDate_User = datetime.datetime.strptime(StartDate_User,'%m/%d/%Y')
    EndDate_User = datetime.datetime.strptime(EndDate_User,'%m/%d/%Y')
    
    # User Dates Corrected
    StartDate_User_Corrected = datetime.datetime(StartDate_User.year,StartDate_User.month,StartDate_User.day,0,int(FileResolution_Minutes),0)
    EndDate_User_Corrected = datetime.datetime(EndDate_User.year,EndDate_User.month,EndDate_User.day,23,60-int(FileResolution_Minutes),0)

    # Getting Start/End Datetime Indices
    Start_DateTime_Index = DateTime_List.index(StartDate_User_Corrected)
    End_DateTime_Index = DateTime_List.index(EndDate_User_Corrected)+1

    # Creating DateTime List to Plot
    DateTime_List_Plot = matplotlib.dates.date2num(DateTime_List[Start_DateTime_Index:End_DateTime_Index])
    
    # IF ELSE LOOP: Based on type of Graph
    if (TypeOfGraph == 1): # Time Series Plots

        # Print Level1 Variable Names
        Level1_VariableNames_List = Aggregation_DF.columns
        
        print('\n')
        print('Variable Names List:')
        print('\n')
        for VarName in Level1_VariableNames_List:
            print(VarName)
            
        print('\n')
        
        # User Selection: Select Level1 Variable 
        Level1_Variable_Name = input('Select ONE Variable from above List: ')
        
        # Get the Level1 Variable
        Level1_Variable = Aggregation_DF[Level1_Variable_Name].iloc[Start_DateTime_Index:End_DateTime_Index]

        # Plotting Data
        plt.plot(DateTime_List_Plot, Level1_Variable)

        # Plot Embellishments
        plt.title('Time Series: '+Level1_Variable_Name.replace('_',' '), fontsize=Title_FontSize, multialignment = 'center')
        plt.xlabel('Time', fontsize=Axis_FontSize, multialignment = 'center')
        plt.ylabel(Level1_Variable_Name.replace('_',' '), fontsize=Axis_FontSize, multialignment = 'center')

        plt.xticks(rotation=XTick_RotationDeg)

        plt.tight_layout()

        # Saving Figure dependent on SaveFigures_Indicator



        # Creating File Path
        Plot_File_Path = os.path.join(Sim_IDFDataExploration_FolderPath)

        # Saving Plot
        plt.savefig(os.path.join(Plot_File_Path, 'Agg_' + Level1_Variable_Name + 'TimeSeriesPlot' + '.png'), dpi=300)

        # Showing Figure in Console
        # plt.show()

    elif(TypeOfGraph == 2): # Scatter Plots    
    
        # Initialize XY_Variable_List
        XY_Variable_List = []     
        
        Level1_Variable_Name_List = []
        
        # FOR LOOP; For X and Y Variable Selection
        for ii in range(2): 
            
            # IF ELSE LOOP: For Variable_Stem
            if (ii==0):
                Variable_Stem = 'X'
            elif (ii == 1):
                Variable_Stem = 'Y'
            
            # Print Level1 Variable Names
            Level1_VariableNames_List = Aggregation_DF.columns
            
            print('\n')
            print(Variable_Stem + 'Variable Names List:')
            print('\n')
            for VarName in Level1_VariableNames_List:
                print(VarName)
                
            print('\n')
            
            # User Selection: Select Level1 Variable 
            Level1_Variable_Name = input('Select ONE '+ Variable_Stem +' Variable from above List: ')
            
            # Get the Level1 Variable
            Level1_Variable = Aggregation_DF[Level1_Variable_Name].iloc[Start_DateTime_Index:End_DateTime_Index]
            
            # Getting required columns from Level1_Variable for Plotting
            Variables_Plot_DF = Level1_Variable    
        
            # Filling up XY_Variable_List
            XY_Variable_List.append(Variables_Plot_DF)   
            
            Level1_Variable_Name_List.append(Level1_Variable_Name) 

        # Plotting
        plt.scatter(XY_Variable_List[0], XY_Variable_List[1])

        # Plot Embellishments
        plt.title('Scatter Plot: ', fontsize=Title_FontSize, multialignment = 'center')
        plt.xlabel(Level1_Variable_Name_List[0].replace('_',' '), fontsize=Axis_FontSize, multialignment = 'center')
        plt.ylabel(Level1_Variable_Name_List[1].replace('_',' '), fontsize=Axis_FontSize, multialignment = 'center')

        plt.xticks(rotation=XTick_RotationDeg)

        plt.tight_layout()

        # Saving Figure dependent on SaveFigures_Indicator
        # Creating File Path
        Plot_File_Path = os.path.join(Sim_IDFDataExploration_FolderPath)

        # Saving Plot
        plt.savefig(os.path.join(Plot_File_Path,'Agg_' + '_ScatterPlot' + '.png'),dpi=300)

        # Showing Figure in Console
        plt.show()
         
            
