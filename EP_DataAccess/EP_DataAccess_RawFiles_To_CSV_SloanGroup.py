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

Total_Zone_Num = 5

Aggregation_File_Name_Stem = 'DOE_SmallOffice_Data_Zone_'

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

# Getting DateTime_List
DateTime_List = IDF_OutputVariable_Dict['DateTime_List']

Date_Time_Dict = {'Date_Time': DateTime_List}

# =============================================================================
# Creating Zone Wise CSV Files in the format required by Sloan Group
# =============================================================================

# Getting required data from IDF_OutputVariable_Dict
Zone_Temperature_DF = IDF_OutputVariable_Dict['Zone_Air_Temperature'].iloc[:,1:]

Site_Temperature = IDF_OutputVariable_Dict['Site_Outdoor_Air_Drybulb_Temperature'].iloc[:,1:]

Site_HVAC_Power = IDF_OutputVariable_Dict['Facility_Total_HVAC_Electric_Demand_Power'].iloc[:,1:]

# FOR LOOP: For each Zone
for ii in range(Total_Zone_Num):

    # Initialization
    Current_Zone_Data_DF = copy.deepcopy(pd.DataFrame())

    # Creating Q_HVAC
    Zone_HVAC_Heating = IDF_OutputVariable_Dict['Zone_Air_System_Sensible_Heating_Rate'].iloc[:,ii+2]

    Zone_HVAC_Cooling = IDF_OutputVariable_Dict['Zone_Air_System_Sensible_Cooling_Rate'].iloc[:,ii+2]

    Zone_HVAC_HeatCool_Power = Zone_HVAC_Heating - Zone_HVAC_Cooling

    Zone_HVAC_HeatCool_Power_Dict = {'HVAC_HeatCool_Power':Zone_HVAC_HeatCool_Power.tolist()}

    # Concatenating DFs to get Current Zone Data DF
    Current_Zone_Data_DF = pd.concat([Current_Zone_Data_DF, pd.DataFrame(Date_Time_Dict), Zone_Temperature_DF.iloc[:, ii+1], Site_Temperature, pd.DataFrame(Zone_HVAC_HeatCool_Power_Dict), Site_HVAC_Power],axis=1)

    # Create Current CSV File Name
    Current_CSV_File_Name = Aggregation_File_Name_Stem + str(ii+1) + '.csv'

    # Creating CSV File
    Current_Zone_Data_DF.to_csv(Current_CSV_File_Name)

