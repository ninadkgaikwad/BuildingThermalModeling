# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:48:16 2022

@author: ninad gaikwad 
"""

# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import pandas as pd
import numpy as np
import pickle
import datetime
import copy


# Custom Modules


# =============================================================================
# User Inputs
# =============================================================================

Simulation_Name = "test1"

Aggregation_Zone_List = [['CORE_ZN','PERIMETER_ZN_1','PERIMETER_ZN_2','PERIMETER_ZN_3','PERIMETER_ZN_4']] # DOE Small Office: CORE_ZN , PERIMETER_ZN_1 , PERIMETER_ZN_2, PERIMETER_ZN_3 , PERIMETER_ZN_4

Aggregation_Zone_NameStem = 'Aggregation_Zone'

Aggregation_File_Name = 'Aggregation_Dict_1Zone.pickle'

Type_Aggregation = 1 # 1 : Average , 2 : Weighted Floor Area Average, 3 : Weighted Volume Average

SystemNode_Name = 'DIRECT AIR INLET NODE'

Aggregation_VariableNames_List = ['Schedule_Value_',
                                  'Facility_Total_HVAC_Electric_Demand_Power_',
                                  'Site_Diffuse_Solar_Radiation_Rate_per_Area_',
                                  'Site_Direct_Solar_Radiation_Rate_per_Area_',
                                  'Site_Outdoor_Air_Drybulb_Temperature_',
                                  'Site_Solar_Altitude_Angle_',
                                  'Surface_Inside_Face_Internal_Gains_Radiation_Heat_Gain_Rate_',
                                  'Surface_Inside_Face_Lights_Radiation_Heat_Gain_Rate_',
                                  'Surface_Inside_Face_Solar_Radiation_Heat_Gain_Rate_',
                                  'Surface_Inside_Face_Temperature_',
                                  'Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_',
                                  'Zone_Air_Temperature_',
                                  'Zone_People_Convective_Heating_Rate_',
                                  'Zone_Lights_Convective_Heating_Rate_',
                                  'Zone_Electric_Equipment_Convective_Heating_Rate_',
                                  'Zone_Gas_Equipment_Convective_Heating_Rate_',
                                  'Zone_Other_Equipment_Convective_Heating_Rate_',
                                  'Zone_Hot_Water_Equipment_Convective_Heating_Rate_',
                                  'Zone_Steam_Equipment_Convective_Heating_Rate_',
                                  'Zone_People_Radiant_Heating_Rate_',
                                  'Zone_Lights_Radiant_Heating_Rate_',
                                  'Zone_Electric_Equipment_Radiant_Heating_Rate_',
                                  'Zone_Gas_Equipment_Radiant_Heating_Rate_',
                                  'Zone_Other_Equipment_Radiant_Heating_Rate_',
                                  'Zone_Hot_Water_Equipment_Radiant_Heating_Rate_',
                                  'Zone_Steam_Equipment_Radiant_Heating_Rate_',
                                  'Zone_Lights_Visible_Radiation_Heating_Rate_',
                                  'Zone_Total_Internal_Convective_Heating_Rate_',
                                  'Zone_Total_Internal_Radiant_Heating_Rate_',
                                  'Zone_Total_Internal_Total_Heating_Rate_',
                                  'Zone_Total_Internal_Visible_Radiation_Heating_Rate_',
                                  'Zone_Air_System_Sensible_Cooling_Rate_',
                                  'Zone_Air_System_Sensible_Heating_Rate_',
                                  'System_Node_Temperature_',
                                  'System_Node_Mass_Flow_Rate_']


# =============================================================================
# Getting Required Data from Sim_ProcessedData
# =============================================================================

# Getting Current File Directory Path
Current_FilePath = os.path.dirname(__file__)

# Getting Sim_ProcessedData Folder Path
Sim_ProcessedData_FolderPath = os.path.join(Current_FilePath,  '..',  '..', 'Results', 'Processed_BuildingSim_Data', Simulation_Name, 'Sim_ProcessedData')

# Get Required Files from Sim_ProcessedData_FolderPath
IDF_OutputVariable_Dict_file = open(os.path.join(Sim_ProcessedData_FolderPath,'IDF_OutputVariables_DictDF.pickle'),"rb")

IDF_OutputVariable_Dict = pickle.load(IDF_OutputVariable_Dict_file)

Eio_OutputFile_Dict_file = open(os.path.join(Sim_ProcessedData_FolderPath,'Eio_OutputFile.pickle'),"rb")

Eio_OutputFile_Dict = pickle.load(Eio_OutputFile_Dict_file)

# Getting DateTime_List
DateTime_List = IDF_OutputVariable_Dict['DateTime_List']


# =============================================================================
# Creating Unique Zone Name List and Associated Areas and Volume Dicts 
# =============================================================================

# Creating Unique List of Zones
Total_Zone_List = []

# FOR LOOP: For each element of Aggregation_Zone_List
for CurrentZone_List in Aggregation_Zone_List:
    
    # FOR LOOP: For each element of CurrentZone_List
    for CurrentZone in CurrentZone_List:
        
        # Appending CurrentZone to Total_Zone_List
        Total_Zone_List.append(CurrentZone)
        
# Creating Unique Zone List
Unique_Zone_List = list(set(Total_Zone_List))

# IF ELSE LOOP: For cheking Type_Aggregation
if (Type_Aggregation == 1): # Normal Aggregation

    # Do nothing
    Do_Nothing = 0

elif (Type_Aggregation == 2): # Weighted Area Aggregation

    # Initializing Unique_Zone_Area_Dict and Unique_Zone_Volume_Dict
    Unique_Zone_Area_Dict = {}
    
    # Getiing Zone Area and Volumes from Eio_OutputFile_Dict
    
    # FOR LOOP: For each element of Unique_Zone_List
    for Unique_Zone in Unique_Zone_List:
        
        Unique_Zone_Area_Dict[Unique_Zone] = float(Eio_OutputFile_Dict['Zone Information'].query('`Zone Name` == Unique_Zone')['Floor Area {m2}'])
    
    # Creating Zone_TotalArea_List
    Zone_TotalArea_List = []
    
    # FOR LOOP: For each Element in Aggregation_Zone_List
    for Aggregation_Zone_List1 in Aggregation_Zone_List:
        
        # Initializing TotalArea
        TotalArea = 0
        
        # FOR LOOP: For each Element in Aggregation_Zone_List1
        for element in Aggregation_Zone_List1:
            
            # Summing Up Zone Area
            TotalArea = TotalArea + Unique_Zone_Area_Dict[element]
            
        # Appending Zone_TotalArea_List
        Zone_TotalArea_List.append(TotalArea)
    
elif (Type_Aggregation == 3): # Weighted Volume Aggregation

    # Initializing Unique_Zone_Area_Dict and Unique_Zone_Volume_Dict    
    Unique_Zone_Volume_Dict = {}
    
    # Getiing Zone Area and Volumes from Eio_OutputFile_Dict
    
    # FOR LOOP: For each element of Unique_Zone_List
    for Unique_Zone in Unique_Zone_List:
        
        Unique_Zone_Volume_Dict[Unique_Zone] = float(Eio_OutputFile_Dict['Zone Information'].query('`Zone Name` == Unique_Zone')['Volume {m3}'])

    # Creating Zone_TotalVolume_List
    Zone_TotalVolume_List = []
    
    # FOR LOOP: For each Element in Aggregation_Zone_List
    for Aggregation_Zone_List1 in Aggregation_Zone_List:
        
        # Initializing TotalArea
        TotalVolume = 0
        
        # FOR LOOP: For each Element in Aggregation_Zone_List1
        for element in Aggregation_Zone_List1:
            
            # Summing Up Zone Area
            TotalVolume = TotalVolume + Unique_Zone_Volume_Dict[element]
            
        # Appending Zone_TotalArea_List
        Zone_TotalVolume_List.append(TotalVolume)
        
        
# =============================================================================
# Creating Aggregation_DF with relevant Columns to hold Aggregated Data
# =============================================================================
    
# Creating Equipment List
Equipment_List = ['People', 'Lights', 'ElectricEquipment', 'GasEquipment', 'OtherEquipment', 'HotWaterEquipment', 'SteamEquipment']

# Initializing Aggregation_DF
Aggregation_DF = pd.DataFrame()

# FOR LOOP: For each Variable Name in Aggregation_VariableNames_List
for key in Aggregation_VariableNames_List:
    
    # IF LOOP: For the Variable Name Schedule_Value_
    if (key == 'Schedule_Value_'): # Create Schedule Columns which are needed
    
        # FOR LOOP: For each element in Equipment_List
        for element in Equipment_List:
            
            # Creating Current_EIO_Dict_Key
            Current_EIO_Dict_Key = element + ' ' + 'Internal Gains Nominal'
            
            # IF LOOP: To check if Current_EIO_Dict_Key is present in Eio_OutputFile_Dict
            if (Current_EIO_Dict_Key in Eio_OutputFile_Dict): # Key present in Eio_OutputFile_Dict            
            
                # Creating key1 for column Name
                key1 = key + element
            
                # Initializing Aggregation_Dict with None
                Aggregation_DF[key1] = None    
    
    else: # For all other Columns
    
        # Initializing Aggregation_Dict with None
        Aggregation_DF[key] = None

# Initializing Aggregation_DF_Equipment
Aggregation_DF_Equipment = pd.DataFrame()        
        
# FOR LOOP: For each element in Equipment_List
for element in Equipment_List:
    
    # Creating Current_EIO_Dict_Key
    Current_EIO_Dict_Key = element + ' ' + 'Internal Gains Nominal'
    
    # IF LOOP: To check if Current_EIO_Dict_Key is present in Eio_OutputFile_Dict
    if (Current_EIO_Dict_Key in Eio_OutputFile_Dict): # Key present in Eio_OutputFile_Dict            
    
        # Creating key1 for column Name
        key1 =  element + '_Level'
    
        # Initializing Aggregation_Dict with None
        Aggregation_DF_Equipment[key1] = None  
        

# =============================================================================
# Creating Aggregation_Dict to hold Aggregated Data 
# =============================================================================
        
# Initializing Aggregation_Dict
Aggregation_Dict = {'DateTime_List': DateTime_List}

# Initializing Counter
Counter = 0

# FOR LOOP: For each element in Aggregation_Zone_List
for element in Aggregation_Zone_List:
    
    # Incrementing Counter
    Counter = Counter + 1
    
    # Creating Aggregated Zone name 1 : For the Aggregated Time Series
    Aggregated_Zone_Name_1 = Aggregation_Zone_NameStem + "_" + str(Counter)
    
    # Creating Aggregated Zone name 2 : For the Aggregated Equipment 
    Aggregated_Zone_Name_2 = Aggregation_Zone_NameStem + "_Equipment_" + str(Counter)    
    
    # Appending empty Aggregation_DF to Aggregation_Dict
    Aggregation_Dict[Aggregated_Zone_Name_1] = copy.deepcopy(Aggregation_DF)
    
    Aggregation_Dict[Aggregated_Zone_Name_2] = copy.deepcopy(Aggregation_DF_Equipment)


# =============================================================================
# Creating Aggregated Data
# =============================================================================
    
# Initializing Counter
Counter = 0
    
# FOR LOOP: For each Aggregated Zone in Aggregation_Zone_List
for Current_Aggregated_Zone_List in Aggregation_Zone_List:
    
    # Incrementing Counter
    Counter = Counter + 1
    
    # Creating Aggregated Zone name
    Aggregated_Zone_Name_1 = Aggregation_Zone_NameStem + "_" + str(Counter)
    
    Aggregated_Zone_Name_2 = Aggregation_Zone_NameStem + "_Equipment_" + str(Counter) 
    
    # FOR LOOP: For each Aggregation_VariableName in Aggregation_VariableNames_List
    for Current_Aggregation_VariableName in Aggregation_Dict[Aggregated_Zone_Name_1].columns:
       
        # Getting Current_Aggregation_Variable Type
        Current_Aggregation_Variable_Type = Current_Aggregation_VariableName.split('_')[0]
        
        # Aggregation Based on Current_Aggregation_Variable_Type
        if (Current_Aggregation_Variable_Type == 'Site' or Current_Aggregation_Variable_Type == 'Facility'): # Site

            # Getting Current_Aggregation_Variable from IDF_OutputVariable_Dict
            Current_Aggregation_Variable = IDF_OutputVariable_Dict[Current_Aggregation_VariableName[:-1]]
                        
            # Filling Aggregation_Dict with Current_Aggregation_Variable
            Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName] = Current_Aggregation_Variable.iloc[:, [1]]
            
        elif (Current_Aggregation_Variable_Type == 'Zone'): # Zone

            if Current_Aggregation_VariableName[:-1] in IDF_OutputVariable_Dict.keys():
                # Getting Current_Aggregation_Variable from IDF_OutputVariable_Dict
                Current_Aggregation_Variable = IDF_OutputVariable_Dict[Current_Aggregation_VariableName[:-1]]

                # Getting Dataframe subset based on Current_Aggregated_Zone_List
                Current_DF_Cols_Desired = []

                #  Getting Current_Aggregation_Variable_ColName_List
                Current_Aggregation_Variable_ColName_List = Current_Aggregation_Variable.columns

                # FOR LOOP: For each element in Current_Aggregated_Zone_List
                for ColName1 in Current_Aggregated_Zone_List:

                    # FOR LOOP: For each element in Current_Aggregation_Variable_ColName_List
                    for ColName2 in Current_Aggregation_Variable_ColName_List:

                        # IF LOOP: For checking presence of ColName1 in ColName2
                        if (ColName2.find(ColName1) >= 0): # ColName1 present in ColName2

                            # Appending ColName2 to Current_DF_Cols_Desired
                            Current_DF_Cols_Desired.append(ColName2)

                            # IF ELSE LOOP: For Type_Aggregation
                            if (Type_Aggregation == 1): # Normal Aggregation

                                # Do Nothing
                                Do_Nothing = 0

                            elif (Type_Aggregation == 2): # Weighted Area Aggregation

                                # Aggregate by Area
                                Current_Aggregation_Variable[ColName2] = Unique_Zone_Area_Dict[ColName1] * Current_Aggregation_Variable[ColName2]

                            elif (Type_Aggregation == 3): # Weighted Volume Aggregation

                                # Aggregate by Volume
                                Current_Aggregation_Variable[ColName2] = Unique_Zone_Volume_Dict[ColName1] * Current_Aggregation_Variable[ColName2]

                # IF ELSE LOOP: For aggregating according to Type_Aggregation and storing in Aggregation_Dict
                if (Type_Aggregation == 1): # Normal Aggregation

                    # Filling Aggregation_Dict with Current_Aggregation_Variable
                    Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName] = Current_Aggregation_Variable[Current_DF_Cols_Desired].mean(1)

                elif (Type_Aggregation == 2): # Weighted Area Aggregation

                    # Filling Aggregation_Dict with Current_Aggregation_Variable
                    Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName]  = (Current_Aggregation_Variable[Current_DF_Cols_Desired].sum(1))/(Zone_TotalArea_List[Counter])

                elif (Type_Aggregation == 3): # Weighted Volume Aggregation

                    # Filling Aggregation_Dict with Current_Aggregation_Variable
                    Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName]  = (Current_Aggregation_Variable[Current_DF_Cols_Desired].sum(1))/(Zone_TotalVolume_List[Counter])

        elif (Current_Aggregation_Variable_Type == 'Surface'): # Surface

            # Getting Current_Aggregation_Variable from IDF_OutputVariable_Dict
            Current_Aggregation_Variable = IDF_OutputVariable_Dict[Current_Aggregation_VariableName[:-1]]
                          
            # Getting Dataframe subset based on Current_Aggregated_Zone_List
            Current_DF_Cols_Desired = []
            
            # Initializing Current_DF
            Current_DF = pd.DataFrame()
            
            #  Getting Current_Aggregation_Variable_ColName_List 
            Current_Aggregation_Variable_ColName_List = Current_Aggregation_Variable.columns
            
            # FOR LOOP: For each element in Current_Aggregated_Zone_List
            for ColName1 in Current_Aggregated_Zone_List:
                
                # FOR LOOP: For each element in Current_Aggregation_Variable_ColName_List
                for ColName2 in Current_Aggregation_Variable_ColName_List:
                    
                    # IF LOOP: For checking presence of ColName1 in ColName2
                    if (ColName2.find(ColName1) >= 0): # ColName1 present in ColName2
                    
                        # Appending ColName2 to Current_DF_Cols_Desired
                        Current_DF_Cols_Desired.append(ColName2)
                        
                        # IF ELSE LOOP: For Type_Aggregation
                        if (Type_Aggregation == 1): # Normal Aggregation
                            
                            # Do Nothing
                            Do_Nothing = 0
                            
                        elif (Type_Aggregation == 2): # Weighted Area Aggregation
                                
                            # Aggregate by Area
                            Current_Aggregation_Variable[ColName2] = Unique_Zone_Area_Dict[ColName1] * Current_Aggregation_Variable[ColName2]
                                
                        elif (Type_Aggregation == 3): # Weighted Volume Aggregation
                            
                            # Aggregate by Volume
                            Current_Aggregation_Variable[ColName2] = Unique_Zone_Volume_Dict[ColName1] * Current_Aggregation_Variable[ColName2]
                            
                # IF ELSE LOOP: For filling Up Current_DF according to Current_Aggregation_VariableName
                if ((Current_Aggregation_VariableName.find('Heat') >= 0) or (Current_Aggregation_VariableName.find('Gain') >= 0) or (Current_Aggregation_VariableName.find('Rate') >= 0) or (Current_Aggregation_VariableName.find('Power') >= 0) or (Current_Aggregation_VariableName.find('Energy') >= 0)): # Its an additive Variable
                    
                    # Adding Column to Current_DF
                    Current_DF[ColName1] = Current_Aggregation_Variable[Current_DF_Cols_Desired].sum(1)

                else: # It's a mean Variable
                    
                    # Addding Column to Current_DF
                    Current_DF[ColName1] = Current_Aggregation_Variable[Current_DF_Cols_Desired].mean(1)
  
            # IF ELSE LOOP: For aggregating according to Type_Aggregation and storing in Aggregation_Dict
            if (Type_Aggregation == 1): # Normal Aggregation
            
                # Filling Aggregation_Dict with Current_Aggregation_Variable
                Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName]  = Current_DF[Current_Aggregated_Zone_List].mean(1)
            
            elif (Type_Aggregation == 2): # Weighted Area Aggregation
                
                # Filling Aggregation_Dict with Current_Aggregation_Variable
                Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName]  = (Current_DF[Current_Aggregated_Zone_List].sum(1))/(Zone_TotalArea_List[Counter])
          
            elif (Type_Aggregation == 3): # Weighted Volume Aggregation

                # Filling Aggregation_Dict with Current_Aggregation_Variable
                Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName]  = (Current_DF[Current_Aggregated_Zone_List].sum(1))/(Zone_TotalVolume_List[Counter])
           
            
        elif (Current_Aggregation_Variable_Type == 'System'): # System Node

            # Getting Current_Aggregation_Variable from IDF_OutputVariable_Dict
            Current_Aggregation_Variable = IDF_OutputVariable_Dict[Current_Aggregation_VariableName[:-1]]
                          
            # Getting Dataframe subset based on Current_Aggregated_Zone_List
            Current_DF_Cols_Desired = []
            
            #  Getting Current_Aggregation_Variable_ColName_List 
            Current_Aggregation_Variable_ColName_List = Current_Aggregation_Variable.columns
            
            # FOR LOOP: For each element in Current_Aggregated_Zone_List
            for ColName1 in Current_Aggregated_Zone_List:
                
                # FOR LOOP: For each element in Current_Aggregation_Variable_ColName_List
                for ColName2 in Current_Aggregation_Variable_ColName_List:
                    
                    # IF LOOP: For checking presence of ColName1 in ColName2
                    if ((ColName2.find(ColName1) >= 0) and (ColName2.find(SystemNode_Name) >= 0)): # ColName1 present in ColName2
                    
                        # Appending ColName2 to Current_DF_Cols_Desired
                        Current_DF_Cols_Desired.append(ColName2)
                        
                        # IF ELSE LOOP: For Type_Aggregation
                        if (Type_Aggregation == 1): # Normal Aggregation
                            
                            # Do Nothing
                            Do_Nothing = 0
                            
                        elif (Type_Aggregation == 2): # Weighted Area Aggregation
                                
                            # Aggregate by Area
                            Current_Aggregation_Variable[ColName2] = Unique_Zone_Area_Dict[ColName1] * Current_Aggregation_Variable[ColName2]
                                
                        elif (Type_Aggregation == 3): # Weighted Volume Aggregation
                            
                            # Aggregate by Volume
                            Current_Aggregation_Variable[ColName2] = Unique_Zone_Volume_Dict[ColName1] * Current_Aggregation_Variable[ColName2]
                            
            # IF ELSE LOOP: For aggregating according to Type_Aggregation and storing in Aggregation_Dict
            if (Type_Aggregation == 1): # Normal Aggregation
            
                # Filling Aggregation_Dict with Current_Aggregation_Variable
                Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName]  = Current_Aggregation_Variable[Current_DF_Cols_Desired].mean(1)
            
            elif (Type_Aggregation == 2): # Weighted Area Aggregation
                
                # Filling Aggregation_Dict with Current_Aggregation_Variable
                Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName]  = (Current_Aggregation_Variable[Current_DF_Cols_Desired].sum(1))/(Zone_TotalArea_List[Counter])
          
            elif (Type_Aggregation == 3): # Weighted Volume Aggregation

                # Filling Aggregation_Dict with Current_Aggregation_Variable
                Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName]  = (Current_Aggregation_Variable[Current_DF_Cols_Desired].sum(1))/(Zone_TotalVolume_List[Counter])
           
            
        elif (Current_Aggregation_Variable_Type == 'Schedule'): # Schedule            

 
            # Getting Dataframe subset based on Current_Aggregated_Zone_List
            Current_DF_Cols_Desired = []  
            
            # Create a CurrentLevel_List
            CurrentLevel_List = []

            # Creating Current_VariableName_1
            Current_Aggregation_VariableName_1 = Current_Aggregation_VariableName.split('_')[0] + '_' + Current_Aggregation_VariableName.split('_')[1]
                        
            # Get Current_Element
            Current_Element = Current_Aggregation_VariableName.split('_')[2]
            
            # Creating Current_EIO_Dict_Key
            Current_EIO_Dict_Key = Current_Element + ' ' + 'Internal Gains Nominal'

            # Creating Current_EIO_Dict_Key
            Current_EIO_Dict_Key_Level = Current_Element + '_' + 'Level'
            
            # IF ELSE LOOP: For creating Current_EIO_Dict_Key_Level_ColName based on Current_Element
            if (Current_Element == 'People'): # People
            
                Current_EIO_Dict_Key_Level_ColName = 'Number of People {}'
            
            elif (Current_Element == 'Lights'): # Lights
            
                Current_EIO_Dict_Key_Level_ColName = 'Lighting Level {W}'
            
            else: # ElectricEquipment, OtherEquipment, HotWaterEquipment, SteamEquipment
            
                Current_EIO_Dict_Key_Level_ColName = 'Equipment Level {W}'
            
            # Getting Current_EIO_Dict_DF
            Current_EIO_Dict_DF = Eio_OutputFile_Dict[Current_EIO_Dict_Key]
            
            # Getting Current_Aggregation_Variable from IDF_OutputVariable_Dict
            Current_Aggregation_Variable = IDF_OutputVariable_Dict[Current_Aggregation_VariableName_1]

            #  Getting Current_Aggregation_Variable_ColName_List 
            Current_Aggregation_Variable_ColName_List = Current_Aggregation_Variable.columns
                                   
            # FOR LOOP: For each element in Current_Aggregated_Zone_List
            for ColName1 in Current_Aggregated_Zone_List:
                
                # Getting ColName2 from the 'Schedule Name' Column of Current_EIO_Dict_DF
                ColName2 = str(Current_EIO_Dict_DF[Current_EIO_Dict_DF['Zone Name'] == ColName1]['Schedule Name'].iloc[0])
           
                # Appending ColName2 to Current_DF_Cols_Desired
                Current_DF_Cols_Desired.append(ColName2)
                
                # Getting Equipment Level
                Current_EquipmentLevel = float(Current_EIO_Dict_DF[Current_EIO_Dict_DF['Zone Name'] == ColName1][Current_EIO_Dict_Key_Level_ColName].iloc[0])
                       
                # Appending Current_EquipmentLevel to CurrentLevel_List
                CurrentLevel_List.append(Current_EquipmentLevel)

            # FOR LOOP: Getting Corrected Current_DF_Cols_Desired
            Current_DF_Cols_Desired_Corrected = []

            for ColName3 in Current_DF_Cols_Desired:
                for ColName4 in Current_Aggregation_Variable.columns:
                    if (ColName4.find(ColName3) >= 0):
                        Current_DF_Cols_Desired_Corrected.append(ColName4)

            # Filling Aggregation_Dict with Current_Aggregation_Variable and Current_EIO_Dict_Key_Level
            Aggregation_Dict[Aggregated_Zone_Name_1][Current_Aggregation_VariableName] = Current_Aggregation_Variable[Current_DF_Cols_Desired_Corrected].mean(1)

            Aggregation_Dict[Aggregated_Zone_Name_2][Current_EIO_Dict_Key_Level] = pd.DataFrame(np.array([sum(CurrentLevel_List)/len(CurrentLevel_List)]))
        # else: # Any other Variable
        
        

# =============================================================================
# Creating Sim_AggregatedData Folder
# =============================================================================

# Making Additional Folders for storing Aggregated Files
Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath,  '..',  '..', 'Results', 'Processed_BuildingSim_Data')

Sim_IDFAggregatedData_FolderName = 'Sim_AggregatedData'

# Checking if Folders Exist if not create Folders
if (os.path.isdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_IDFAggregatedData_FolderName))):

    # Folders Exist    
    z = None
    
else:
    
    os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_IDFAggregatedData_FolderName))
    
# Creating Sim_AggregatedData Folder Path
Sim_AggregatedData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_IDFAggregatedData_FolderName)


# =============================================================================
# Storing Aggregated Data in Sim_AggregatedData Folder
# =============================================================================

# Saving Aggregation_Dict as a .pickle File in Results Folder
pickle.dump(Aggregation_Dict, open(os.path.join(Sim_AggregatedData_FolderPath,Aggregation_File_Name), "wb"))
