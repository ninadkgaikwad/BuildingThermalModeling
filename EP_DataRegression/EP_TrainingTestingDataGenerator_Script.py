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
import datetime
import copy

# For debugging
import matplotlib.pyplot as plt

# Custom Modules


# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Aggregation_File_Name = 'Aggregation_Dict_5Zone.pickle'

## User Input: Aggregation Unit Number ##
Aggregation_UnitNumber_Total = 5

# Aggregation Zone NameStem Input
Aggregation_Zone_NameStem = 'Aggregation_Zone'

## User Input: User Defined Testing DateTime Range List ##
TestingDateTime_Range_List = [('01/25/2013', '01/31/2013'),('02/22/2013', '02/28/2013'),('03/25/2013', '03/31/2013'),('04/24/2013', '04/30/2013'),('05/25/2013', '05/31/2013'),('06/24/2013', '06/30/2013'),('07/25/2013', '07/31/2013'),('08/25/2013', '08/31/2013'),('09/24/2013', '09/30/2013'),('10/25/2013', '10/31/2013'),('11/24/2013', '11/30/2013'),('12/25/2013', '12/31/2013')]

''' Everything Else Left Is Used for Training Data'''

TrainingData_RegressionModel_Dict_File_Name1 = 'TrainingData_RegressionModel_Dict'

TestingData_RegressionModel_Dict_File_Name1 = 'TestingData_RegressionModel_Dict'

Aggregation_DF_Train_File_Name1 = 'Aggregation_DF_Train'

Aggregation_DF_Test_File_Name1 = 'Aggregation_DF_Test'

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

# LOOP: For Each Aggregated Sub-Zone
for ii in range(Aggregation_UnitNumber_Total):

    Aggregation_UnitNumber = ii + 1

    # =============================================================================
    # Creating Result File Names
    # =============================================================================

    TrainingData_RegressionModel_Dict_File_Name = TrainingData_RegressionModel_Dict_File_Name1 + '_' + Aggregation_File_Name.split('.')[0] + '_' + str(Aggregation_UnitNumber) + '.pickle'

    TestingData_RegressionModel_Dict_File_Name = TestingData_RegressionModel_Dict_File_Name1 + '_' + Aggregation_File_Name.split('.')[0] + '_' + str(Aggregation_UnitNumber) + '.pickle'

    Aggregation_DF_Train_File_Name = Aggregation_DF_Train_File_Name1 + '_' + Aggregation_File_Name.split('.')[0] + '_' + str(Aggregation_UnitNumber) + '.pickle'

    Aggregation_DF_Test_File_Name = Aggregation_DF_Test_File_Name1 + '_' + Aggregation_File_Name.split('.')[0] + '_' + str(Aggregation_UnitNumber) + '.pickle'

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
    Aggregated_Dict_File = open(os.path.join(Sim_AggregatedData_FolderPath, Aggregation_File_Name), "rb")

    Aggregated_Dict = pickle.load(Aggregated_Dict_File)

    # Getting DateTime List from IDF_OutputVariables_DictDF
    DateTime_List = Aggregated_Dict['DateTime_List']

    # Creating the Correct key based on Aggregation_UnitNumber
    AggregationDf_Key = Aggregation_Zone_NameStem + "_" + str(Aggregation_UnitNumber)

    # Creating Aggregated Zone name 2 : For the Aggregated Equipment
    Aggregated_Equipment_Key = Aggregation_Zone_NameStem + "_Equipment_" + str(Aggregation_UnitNumber)

    # Getting appropriate Aggregation_DF based on AggregationDf_Key
    Aggregation_DF = copy.deepcopy(Aggregated_Dict[AggregationDf_Key])

    # Getting appropriate Aggregation_DF based on AggregationDf_Key
    Aggregation_DF_Equipment = copy.deepcopy(Aggregated_Dict[Aggregated_Equipment_Key])


    # =============================================================================
    # Basic Computation
    # =============================================================================

    Duration = datetime.timedelta(days=1)

    # FOR LOOP: Correcting DateTime_List for 24th Hour Error
    for ii in range(len(DateTime_List)):
        DT = DateTime_List[ii]
        if DT.hour == 0 and DT.minute == 0:
            DT1 = datetime.datetime(DT.year, DT.month, DT.day, 0, 0, 0) + Duration
            DateTime_List[ii] = DT1

    # Add DateTime to Aggregation_DF
    Aggregation_DF['DateTime'] = DateTime_List

    # Getting Start and End Dates for the Dataset
    StartDate_Dataset = DateTime_List[0]
    EndDate_Dataset = DateTime_List[-1]

    # Getting the File Resolution from DateTime_List
    DateTime_Delta = DateTime_List[1] - DateTime_List[0]

    FileResolution_Minutes = DateTime_Delta.seconds/60

    # Computing GHI
    DNI = np.array(Aggregation_DF['Site_Direct_Solar_Radiation_Rate_per_Area_'])
    Theta = np.array(Aggregation_DF['Site_Solar_Altitude_Angle_'])
    DHI = np.array(Aggregation_DF['Site_Diffuse_Solar_Radiation_Rate_per_Area_'])

    GHI = (DNI * np.abs(np.cos(Theta))) + DHI

    GHI = GHI.tolist()

    Aggregation_DF['GHI'] = GHI

    # Correcting Schedule with Equipment Level
    Schedule_Value_People_Corrected = Aggregation_DF['Schedule_Value_People'] * Aggregation_DF_Equipment['People_Level'].iloc[0]
    Schedule_Value_Lights_Corrected = Aggregation_DF['Schedule_Value_Lights'] * Aggregation_DF_Equipment['Lights_Level'].iloc[0]
    Schedule_Value_ElectricEquipment_Corrected = Aggregation_DF['Schedule_Value_ElectricEquipment'] * Aggregation_DF_Equipment['ElectricEquipment_Level'].iloc[0]

    Aggregation_DF['Schedule_Value_People_Corrected'] = Schedule_Value_People_Corrected
    Aggregation_DF['Schedule_Value_Lights_Corrected'] = Schedule_Value_Lights_Corrected
    Aggregation_DF['Schedule_Value_ElectricEquipment_Corrected'] = Schedule_Value_ElectricEquipment_Corrected

    # Computing HVAC Parameters
    Tz = np.array(Aggregation_DF['Zone_Air_Temperature_'])
    Ts = np.array(Aggregation_DF['System_Node_Temperature_'])
    M_Dot = np.array(Aggregation_DF['System_Node_Mass_Flow_Rate_'])
    Ca = 1.004

    # QHVAC_X = Ca * M_Dot * (Ts - Tz)

    QHVAC_X = 1000 * Ca * M_Dot * (Ts - Tz) # Debugger

    QHVAC_X = QHVAC_X.tolist()

    Aggregation_DF['QHVAC_X'] = QHVAC_X

    # QHVAC_Y = Aggregation_DF['Zone_Air_System_Sensible_Heating_Rate_'] - Aggregation_DF['Zone_Air_System_Sensible_Cooling_Rate_']

    QHVAC_Y = Aggregation_DF['Facility_Total_HVAC_Electric_Demand_Power_'] / Aggregation_UnitNumber_Total

    Aggregation_DF['QHVAC_Y'] = QHVAC_Y

    # Debugger
    plt.figure()    
    plt.plot(Aggregation_DF['Zone_Air_System_Sensible_Heating_Rate_'] )
    plt.plot(-Aggregation_DF['Zone_Air_System_Sensible_Cooling_Rate_'] )
    plt.plot(Aggregation_DF['QHVAC_X'] )
    plt.show()

    plt.figure()    
    plt.plot(M_Dot )
    plt.show()

    plt.figure()    
    plt.plot(Ts )
    plt.show()


    # =============================================================================
    # Separating Aggregation_DF into Test and Train Data
    # =============================================================================

    # Initializing DateRange List
    Test_DateRange_Index = []
    Train_DateRange_Index = []

    # FOR LOOP:
    for DateTime_Tuple in TestingDateTime_Range_List:

        # Getting Start and End Date
        StartDate = datetime.datetime.strptime(DateTime_Tuple[0],'%m/%d/%Y')
        EndDate = datetime.datetime.strptime(DateTime_Tuple[1],'%m/%d/%Y')

        # User Dates Corrected
        StartDate_Corrected = datetime.datetime(StartDate.year,StartDate.month,StartDate.day,0,int(FileResolution_Minutes),0)
        EndDate_Corrected = datetime.datetime(EndDate.year,EndDate.month,EndDate.day,23,60-int(FileResolution_Minutes),0)

        Counter_DateTime = -1

        # FOR LOOP:
        for Element in DateTime_List:
            Counter_DateTime = Counter_DateTime + 1

            if (Element >= StartDate_Corrected and Element <= EndDate_Corrected):
                Test_DateRange_Index.append(Counter_DateTime)

    # Getting Train DateRange Index
    Test_DateRange_Index_Set = set(Test_DateRange_Index)
    Total_DateRange_Index_Set = set(list(range(0,len(DateTime_List))))
    Train_DateRange_Index_Set = Total_DateRange_Index_Set.difference(Test_DateRange_Index_Set)
    Train_DateRange_Index = list(Train_DateRange_Index_Set)

    # Getting Train and Test Dataset
    Aggregation_DF_Train = copy.deepcopy(Aggregation_DF.iloc[Train_DateRange_Index,:])
    Aggregation_DF_Test = copy.deepcopy(Aggregation_DF.iloc[Test_DateRange_Index,:])


    # =============================================================================
    # Creating Dictionary for Train and Test Data for Regression Model
    # =============================================================================

    # Creating Main Dictionary
    TrainingData_RegressionModel_Dict = {}
    TestingData_RegressionModel_Dict = {}

    # Adding DateTime to the Dictionary
    TrainingData_RegressionModel_Dict['DateTime'] = Aggregation_DF_Train[['DateTime']]
    TestingData_RegressionModel_Dict['DateTime'] = Aggregation_DF_Test[['DateTime']]
    
    # Regression Model: QSol1
    TrainingData_RegressionModel_Dict['QSol1'] = Aggregation_DF_Train[['GHI','Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_']]
    TestingData_RegressionModel_Dict['QSol1'] = Aggregation_DF_Test[['GHI','Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_']]

    # Regression Model: QSol2
    TrainingData_RegressionModel_Dict['QSol2'] = Aggregation_DF_Train[['GHI','Surface_Inside_Face_Solar_Radiation_Heat_Gain_Rate_']]
    TestingData_RegressionModel_Dict['QSol2'] = Aggregation_DF_Test[['GHI','Surface_Inside_Face_Solar_Radiation_Heat_Gain_Rate_']]

    # Regression Model: QZic_P
    TrainingData_RegressionModel_Dict['QZic_P'] = Aggregation_DF_Train[['Schedule_Value_People_Corrected','Zone_People_Convective_Heating_Rate_']]
    TestingData_RegressionModel_Dict['QZic_P'] = Aggregation_DF_Test[['Schedule_Value_People_Corrected','Zone_People_Convective_Heating_Rate_']]

    # Regression Model: QZic_L
    TrainingData_RegressionModel_Dict['QZic_L'] = Aggregation_DF_Train[['Schedule_Value_Lights_Corrected','Zone_Lights_Convective_Heating_Rate_']]
    TestingData_RegressionModel_Dict['QZic_L'] = Aggregation_DF_Test[['Schedule_Value_Lights_Corrected','Zone_Lights_Convective_Heating_Rate_']]

    # Regression Model: QZic_EE
    TrainingData_RegressionModel_Dict['QZic_EE'] = Aggregation_DF_Train[['Schedule_Value_ElectricEquipment_Corrected','Zone_Electric_Equipment_Convective_Heating_Rate_']]
    TestingData_RegressionModel_Dict['QZic_EE'] = Aggregation_DF_Test[['Schedule_Value_ElectricEquipment_Corrected','Zone_Electric_Equipment_Convective_Heating_Rate_']]

    # Regression Model: QZir_P
    TrainingData_RegressionModel_Dict['QZir_P'] = Aggregation_DF_Train[['Schedule_Value_People_Corrected','Zone_People_Radiant_Heating_Rate_']]
    TestingData_RegressionModel_Dict['QZir_P'] = Aggregation_DF_Test[['Schedule_Value_People_Corrected','Zone_People_Radiant_Heating_Rate_']]

    # Regression Model: QZir_L
    TrainingData_RegressionModel_Dict['QZir_L'] = Aggregation_DF_Train[['Schedule_Value_Lights_Corrected','Zone_Lights_Radiant_Heating_Rate_']]
    TestingData_RegressionModel_Dict['QZir_L'] = Aggregation_DF_Test[['Schedule_Value_Lights_Corrected','Zone_Lights_Radiant_Heating_Rate_']]

    # Regression Model: QZir_EE
    TrainingData_RegressionModel_Dict['QZir_EE'] = Aggregation_DF_Train[['Schedule_Value_ElectricEquipment_Corrected','Zone_Electric_Equipment_Radiant_Heating_Rate_']]
    TestingData_RegressionModel_Dict['QZir_EE'] = Aggregation_DF_Test[['Schedule_Value_ElectricEquipment_Corrected','Zone_Electric_Equipment_Radiant_Heating_Rate_']]

    # Regression Model: QZivr_L
    TrainingData_RegressionModel_Dict['QZivr_L'] = Aggregation_DF_Train[['Schedule_Value_Lights_Corrected','Zone_Lights_Visible_Radiation_Heating_Rate_']]
    TestingData_RegressionModel_Dict['QZivr_L'] = Aggregation_DF_Test[['Schedule_Value_Lights_Corrected','Zone_Lights_Visible_Radiation_Heating_Rate_']]

    # Regression Model: QAC
    TrainingData_RegressionModel_Dict['QAC'] = Aggregation_DF_Train[['QHVAC_X', 'QHVAC_Y']]
    TestingData_RegressionModel_Dict['QAC'] = Aggregation_DF_Test[['QHVAC_X', 'QHVAC_Y']]


    # =============================================================================
    # Creating Sim_TrainingTestingData Folder
    # =============================================================================

    # Making Additional Folders for Storing TrainingTesting Files
    Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..', 'Results',
                                                         'Processed_BuildingSim_Data')

    Sim_TrainingTestingData_FolderName = 'Sim_TrainingTestingData'

    # Checking if Folders Exist if not create Folders
    if (
    os.path.isdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_TrainingTestingData_FolderName))):

        # Folders Exist
        z = None

    else:

        os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_TrainingTestingData_FolderName))

    # Creating Sim_TrainingTestingData Folder Path
    Sim_TrainingTestingData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name,
                                                 Sim_TrainingTestingData_FolderName)

    # =============================================================================
    # Storing TrainingTesting Data in Sim_TrainingTestingData Folder
    # =============================================================================

    # Saving TrainingTesting_Dict as a .pickle File in Results Folder
    pickle.dump(TrainingData_RegressionModel_Dict, open(os.path.join(Sim_TrainingTestingData_FolderPath, TrainingData_RegressionModel_Dict_File_Name), "wb"))
    pickle.dump(TestingData_RegressionModel_Dict, open(os.path.join(Sim_TrainingTestingData_FolderPath, TestingData_RegressionModel_Dict_File_Name), "wb"))


    # =============================================================================
    # Storing Aggregated TrainingTesting Data in Sim_TrainingTestingData Folder
    # =============================================================================

    # Saving Aggregated TrainingTesting_Dict as a .pickle File in Results Folder
    pickle.dump(Aggregation_DF_Train, open(os.path.join(Sim_TrainingTestingData_FolderPath, Aggregation_DF_Train_File_Name), "wb"))
    pickle.dump(Aggregation_DF_Test, open(os.path.join(Sim_TrainingTestingData_FolderPath, Aggregation_DF_Test_File_Name), "wb"))