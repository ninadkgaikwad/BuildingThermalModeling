# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:48:16 2022

@author: sajjad uddin mahmud
"""

# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import pandas as pd
import numpy as np
import pickle
import cloudpickle
import datetime

# Custom Modules


# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Aggregation_VariableNames_List = ['Schedule_Value_',
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

# Getting desired Simulation Folder Path
Processed_BuildingSim_Simulation_FolderPath = os.path.join(Current_FilePath,  '..',  '..', 'Results', 'Processed_BuildingSim_Data', Simulation_Name)

## Loading the Data File ##

# Getting Sim_AggregatedData_FolderPath
Sim_AggregatedData_FolderPath = os.path.join(Processed_BuildingSim_Simulation_FolderPath, 'Sim_AggregatedData')

# Getting DF_OutputVariablesFull_DictDF.mat containing all Data and DateTime
Aggregated_Dict_File = open(os.path.join(Sim_AggregatedData_FolderPath, 'Aggregation_Dict.pickle'), "rb")

Aggregated_Dict = pickle.load(Aggregated_Dict_File)

# Getting DateTime List from IDF_OutputVariables_DictDF
DateTime_List = Aggregated_Dict['DateTime_List']


# =============================================================================
# Date Segregation for Test and Train Data
# =============================================================================

# Method - First Week from Each Month
Year = DateTime_List[0].year
TestData_DateRange_Index = []
TrainData_DateRange_Index = []

# FOR LOOP:
for Index in range(len(DateTime_List)):
    if DateTime_List[Index].month <= 12:
        if DateTime_List[Index].day <= 7:
            TestData_DateRange_Index.append(Index)
        else:
            TrainData_DateRange_Index.append(Index)


# =============================================================================
# Regression Model for QSol1 and QSol2 (Heat Gain from Solar Radiation)
# =============================================================================

# Reading Data
DNI = IDF_OutputVariable_Dict['Site_Direct_Solar_Radiation_Rate_per_Area']
Theta = IDF_OutputVariable_Dict['Site_Solar_Altitude_Angle']
DHI = IDF_OutputVariable_Dict['Site_Diffuse_Solar_Radiation_Rate_per_Area']

# Initialization
DNI_Test = []
DNI_Train = []
Theta_Test = []
Theta_Train = []
DHI_Test = []
DHI_Train = []

# FOR LOOP: Test Data Segregation
for DateIndex in TestData_DateRange_Index:
    # Reading Current Data
    Current_DNI_Test = DNI.iloc[DateIndex, 1]
    Current_Theta_Test = Theta.iloc[DateIndex, 1]
    Current_DHI_Test = DHI.iloc[DateIndex, 1]

    # Appending Current Data into List
    DNI_Test.append(Current_DNI_Test)
    Theta_Test.append(Current_Theta_Test)
    DHI_Test.append(Current_DHI_Test)

# Converting into Numpy Array
DNI_Test = np.array(DNI_Test)
Theta_Test = np.array(Theta_Test)
DHI_Test = np.array(DHI_Test)

# FOR LOOP: DNI Train Data Segregation
for DateIndex in TrainData_DateRange_Index:
    # Reading Current Data
    Current_DNI_Train = DNI.iloc[DateIndex, 1]
    Current_Theta_Train = Theta.iloc[DateIndex, 1]
    Current_DHI_Train = DHI.iloc[DateIndex, 1]

    # Appending Current Data into List
    DNI_Train.append(Current_DNI_Train)
    Theta_Train.append(Current_Theta_Train)
    DHI_Train.append(Current_DHI_Train)

# Converting into Numpy Array
DNI_Train = np.array(DNI_Train)
Theta_Train = np.array(Theta_Train)
DHI_Train = np.array(DHI_Train)

# Calculating GHI
GHI_Test = (DNI_Test * np.cos(Theta_Test)) + DHI_Test
GHI_Train = (DNI_Train * np.cos(Theta_Train)) + DHI_Train


# =============================================================================
# Regression Model for QInt_C (Internal Heat Gain Convective)
# =============================================================================

# Reading Data
Zone_People_Convective_Heating_Rate = IDF_OutputVariable_Dict['Zone_People_Convective_Heating_Rate']
Zone_Light_Convective_Heating_Rate = IDF_OutputVariable_Dict['Zone_Lights_Convective_Heating_Rate']
Zone_ElectricEquipment_Convective_Heating_Rate = IDF_OutputVariable_Dict['Zone_Electric_Equipment_Convective_Heating_Rate']

# Initialization
Zone_People_Convective_Heating_Rate_Test = []
Zone_People_Convective_Heating_Rate_Train = []
Zone_Light_Convective_Heating_Rate_Test = []
Zone_Light_Convective_Heating_Rate_Train = []
Zone_ElectricEquipment_Convective_Heating_Rate_Test = []
Zone_ElectricEquipment_Convective_Heating_Rate_Train = []


# FOR LOOP: Test Data Segregation
for DateIndex in TestData_DateRange_Index:
    # Reading Current Data
    Current_Zone_People_Convective_Heating_Rate_Test = Zone_People_Convective_Heating_Rate.iloc[DateIndex, 1]
    Current_Zone_Light_Convective_Heating_Rate_Test = Zone_Light_Convective_Heating_Rate.iloc[DateIndex, 1]
    Current_Zone_ElectricEquipment_Convective_Heating_Rate_Test = Zone_ElectricEquipment_Convective_Heating_Rate.iloc[DateIndex, 1]

    # Appending Current Data into List
    Zone_People_Convective_Heating_Rate_Test.append(Current_Zone_People_Convective_Heating_Rate_Test)
    Zone_Light_Convective_Heating_Rate_Test.append(Current_Zone_Light_Convective_Heating_Rate_Test)
    Zone_ElectricEquipment_Convective_Heating_Rate_Test.append(Current_Zone_ElectricEquipment_Convective_Heating_Rate_Test)


# Converting into Numpy Array
Zone_People_Convective_Heating_Rate_Test = np.array(Zone_People_Convective_Heating_Rate_Test)
Zone_Light_Convective_Heating_Rate_Test = np.array(Zone_Light_Convective_Heating_Rate_Test)
Zone_ElectricEquipment_Convective_Heating_Rate_Test = np.array(Zone_ElectricEquipment_Convective_Heating_Rate_Test)


# FOR LOOP: Train Data Segregation
for DateIndex in TrainData_DateRange_Index:
    # Reading Current Data
    Current_Zone_People_Convective_Heating_Rate_Train = Zone_People_Convective_Heating_Rate.iloc[DateIndex, 1]
    Current_Zone_Light_Convective_Heating_Rate_Train = Zone_Light_Convective_Heating_Rate.iloc[DateIndex, 1]
    Current_Zone_ElectricEquipment_Convective_Heating_Rate_Train = Zone_ElectricEquipment_Convective_Heating_Rate.iloc[DateIndex, 1]


    # Appending Current Data into List
    Zone_People_Convective_Heating_Rate_Train.append(Current_Zone_People_Convective_Heating_Rate_Train)
    Zone_Light_Convective_Heating_Rate_Train.append(Current_Zone_Light_Convective_Heating_Rate_Train)
    Zone_ElectricEquipment_Convective_Heating_Rate_Train.append(Current_Zone_ElectricEquipment_Convective_Heating_Rate_Train)


# Converting into Numpy Array
Zone_People_Convective_Heating_Rate_Train = np.array(Zone_People_Convective_Heating_Rate_Train)
Zone_Light_Convective_Heating_Rate_Train = np.array(Zone_Light_Convective_Heating_Rate_Train)
Zone_ElectricEquipment_Convective_Heating_Rate_Train = np.array(Zone_ElectricEquipment_Convective_Heating_Rate_Train)


# Calculating QInt_C (Internal Heat Gain Convective)
Zone_Total_Internal_Convective_Heating_Rate_Test = Zone_People_Convective_Heating_Rate_Test + Zone_Light_Convective_Heating_Rate_Test + Zone_ElectricEquipment_Convective_Heating_Rate_Test
Zone_Total_Internal_Convective_Heating_Rate_Train = Zone_People_Convective_Heating_Rate_Train + Zone_Light_Convective_Heating_Rate_Train + Zone_ElectricEquipment_Convective_Heating_Rate_Train


# =============================================================================
# Regression Model for QInt_R (Internal Heat Gain Radiant)
# =============================================================================

# Reading Data
Zone_People_Radiant_Heating_Rate = IDF_OutputVariable_Dict['Zone_People_Radiant_Heating_Rate']
Zone_Light_Radiant_Heating_Rate = IDF_OutputVariable_Dict['Zone_Lights_Radiant_Heating_Rate']
Zone_ElectricEquipment_Radiant_Heating_Rate = IDF_OutputVariable_Dict['Zone_Electric_Equipment_Radiant_Heating_Rate']
Zone_Lights_Visible_Radiation_Heating_Rate = IDF_OutputVariable_Dict['Zone_Lights_Visible_Radiation_Heating_Rate']

# Initialization
Zone_People_Radiant_Heating_Rate_Test = []
Zone_People_Radiant_Heating_Rate_Train = []
Zone_Light_Radiant_Heating_Rate_Test = []
Zone_Light_Radiant_Heating_Rate_Train = []
Zone_ElectricEquipment_Radiant_Heating_Rate_Test = []
Zone_ElectricEquipment_Radiant_Heating_Rate_Train = []
Zone_Lights_Visible_Radiation_Heating_Rate_Test = []
Zone_Lights_Visible_Radiation_Heating_Rate_Train = []


# FOR LOOP: Test Data Segregation
for DateIndex in TestData_DateRange_Index:
    # Reading Current Data
    Current_Zone_People_Radiant_Heating_Rate_Test = Zone_People_Radiant_Heating_Rate.iloc[DateIndex, 1]
    Current_Zone_Light_Radiant_Heating_Rate_Test = Zone_Light_Radiant_Heating_Rate.iloc[DateIndex, 1]
    Current_Zone_ElectricEquipment_Radiant_Heating_Rate_Test = Zone_ElectricEquipment_Radiant_Heating_Rate.iloc[DateIndex, 1]
    Current_Zone_Lights_Visible_Radiation_Heating_Rate_Test = Zone_Lights_Visible_Radiation_Heating_Rate.iloc[DateIndex, 1]

    # Appending Current Data into List
    Zone_People_Radiant_Heating_Rate_Test.append(Current_Zone_People_Radiant_Heating_Rate_Test)
    Zone_Light_Radiant_Heating_Rate_Test.append(Current_Zone_Light_Radiant_Heating_Rate_Test)
    Zone_ElectricEquipment_Radiant_Heating_Rate_Test.append(Current_Zone_ElectricEquipment_Radiant_Heating_Rate_Test)
    Zone_Lights_Visible_Radiation_Heating_Rate_Test.append(Current_Zone_Lights_Visible_Radiation_Heating_Rate_Test)

# Converting into Numpy Array
Zone_People_Radiant_Heating_Rate_Test = np.array(Zone_People_Radiant_Heating_Rate_Test)
Zone_Light_Radiant_Heating_Rate_Test = np.array(Zone_Light_Radiant_Heating_Rate_Test)
Zone_ElectricEquipment_Radiant_Heating_Rate_Test = np.array(Zone_ElectricEquipment_Radiant_Heating_Rate_Test)
Zone_Lights_Visible_Radiation_Heating_Rate_Test = np.array(Zone_Lights_Visible_Radiation_Heating_Rate_Test)

# FOR LOOP: Train Data Segregation
for DateIndex in TrainData_DateRange_Index:
    # Reading Current Data
    Current_Zone_People_Radiant_Heating_Rate_Train = Zone_People_Radiant_Heating_Rate.iloc[DateIndex, 1]
    Current_Zone_Light_Radiant_Heating_Rate_Train = Zone_Light_Radiant_Heating_Rate.iloc[DateIndex, 1]
    Current_Zone_ElectricEquipment_Radiant_Heating_Rate_Train = Zone_ElectricEquipment_Radiant_Heating_Rate.iloc[
        DateIndex, 1]
    Current_Zone_Lights_Visible_Radiation_Heating_Rate_Train = Zone_Lights_Visible_Radiation_Heating_Rate.iloc[DateIndex, 1]

    # Appending Current Data into List
    Zone_People_Radiant_Heating_Rate_Train.append(Current_Zone_People_Radiant_Heating_Rate_Train)
    Zone_Light_Radiant_Heating_Rate_Train.append(Current_Zone_Light_Radiant_Heating_Rate_Train)
    Zone_ElectricEquipment_Radiant_Heating_Rate_Train.append(Current_Zone_ElectricEquipment_Radiant_Heating_Rate_Train)
    Zone_Lights_Visible_Radiation_Heating_Rate.append(Current_Zone_Lights_Visible_Radiation_Heating_Rate_Train)

# Converting into Numpy Array
Zone_People_Radiant_Heating_Rate_Train = np.array(Zone_People_Radiant_Heating_Rate_Train)
Zone_Light_Radiant_Heating_Rate_Train = np.array(Zone_Light_Radiant_Heating_Rate_Train)
Zone_ElectricEquipment_Radiant_Heating_Rate_Train = np.array(Zone_ElectricEquipment_Radiant_Heating_Rate_Train)
Zone_Lights_Visible_Radiation_Heating_Rate_Train = np.array(Zone_Lights_Visible_Radiation_Heating_Rate_Train)

# Calculating QZIR (Zone Total Internal Radiant Heating Rate)
Zone_Total_Internal_Radiant_Heating_Rate_Test = Zone_People_Radiant_Heating_Rate_Test + Zone_Light_Radiant_Heating_Rate_Test + Zone_ElectricEquipment_Radiant_Heating_Rate_Test
Zone_Total_Internal_Radiant_Heating_Rate_Train = Zone_People_Radiant_Heating_Rate_Train + Zone_Light_Radiant_Heating_Rate_Train + Zone_ElectricEquipment_Radiant_Heating_Rate_Train

# Calculating QZIVR (Zone Total Internal Visible Radiant Heating Rate)
Zone_Total_Internal_Visible_Radiation_Heating_Rate_Test = Zone_Lights_Visible_Radiation_Heating_Rate_Test
Zone_Total_Internal_Visible_Radiation_Heating_Rate_Train = Zone_Lights_Visible_Radiation_Heating_Rate_Train

# Calculating QInt_R (Internal Heat Gain Radiant)

# =============================================================================
# Regression Model for QHVAC (HVAC Heat Gain)
# =============================================================================

