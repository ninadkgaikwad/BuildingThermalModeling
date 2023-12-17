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
import math
import time



# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Learning_Rate = 0.01

Loss_Function = 'mean_squared_error'

Epochs = 20

Batch_Size = 100

Buffer_Input = 1000

Validation_Split = 0.2

## User Input: Aggregation Unit Number ##
Aggregation_UnitNumber = 1

Total_Aggregation_Zone_Number = 1

FeatureType = 0 # 0 - Remove no features, 1 - Remove Internal Heat, 2 - Remove Solar Heat, 3 - Remove Ambient Temp, 4 - Remove HVAC Heat, 5 - Remove Zone Temperature, 6 - Remove all but Zone Temperature, internal heat

# Aggregation Zone NameStem Input
Aggregation_Zone_NameStem = 'Aggregation_Zone'

ANNModel_Key = 'ANN_Model'

# Percentage Training Data to be used
Training_Data_Control = 0 # 0 = All Data, 1 = Not All Data
Training_Data_Percentage_Used = 1 # Values Between 0 and 1

# =============================================================================
# Initialization
# =============================================================================

PHVAC = np.zeros((1,1))

PHVAC1 = np.zeros((1,1))

# =============================================================================
# Getting Required Data from Sim_ProcessedData
# =============================================================================

# Getting Current File Directory Path
Current_FilePath = os.path.dirname(__file__)

# Getting Folder Path
Sim_ProcessedData_FolderPath_AggregatedTestTrain = os.path.join(Current_FilePath, '..', '..', 'Results',
                                                                'Processed_BuildingSim_Data', Simulation_Name,
                                                                'Sim_TrainingTestingData')
Sim_ProcessedData_FolderPath_Regression = os.path.join(Current_FilePath, '..', '..', 'Results',
                                                       'Processed_BuildingSim_Data', Simulation_Name,
                                                       'Sim_RegressionModelData')

# LOOP: Output Generation for Each Aggregated Zone


for kk in range(Total_Aggregation_Zone_Number):

    kk = kk + 1

    print("Current Unit Number: " + str(kk))

    # Creating Required File Names

    Aggregation_DF_Test_File_Name = 'Aggregation_DF_Test_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    Aggregation_DF_Train_File_Name = 'Aggregation_DF_Train_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    ANN_HeatInput_Test_DF_File_Name = 'ANN_HeatInput_Test_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    ANN_HeatInput_Train_DF_File_Name = 'ANN_HeatInput_Train_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    PHVAC_Regression_Model_File_Name = 'PHVAC_RegressionModel_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk)

    # Get Required Files from Sim_AggregatedTestTrainData_FolderPath

    #IF Loop to Control the Amount of Training Data
    if (Training_Data_Control == 1):
        AggregatedTest_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Test_File_Name), "rb")
        AggregatedTest_DF = pickle.load(AggregatedTest_Dict_File)

        AggregatedTrain_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Train_File_Name), "rb")
        AggregatedTrain_DF = pickle.load(AggregatedTrain_Dict_File)

        AggregatedTrain_DF = AggregatedTrain_DF[0:math.floor(Training_Data_Percentage_Used*len(AggregatedTrain_DF))]

        PHVAC_Regression_Model_File_Path = os.path.join(Sim_ProcessedData_FolderPath_Regression,
                                                        PHVAC_Regression_Model_File_Name)
        PHVAC_Regression_Model = tf.keras.models.load_model(PHVAC_Regression_Model_File_Path)

        # Get Required Files from Sim_RegressionModelData_FolderPath
        ANN_HeatInput_Test_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Test_DF_File_Name),
            "rb")
        ANN_HeatInput_Test_DF = pickle.load(ANN_HeatInput_Test_DF_File)

        ANN_HeatInput_Train_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Train_DF_File_Name), "rb")
        ANN_HeatInput_Train_DF = pickle.load(ANN_HeatInput_Train_DF_File)

        ANN_HeatInput_Train_DF = ANN_HeatInput_Train_DF[0:math.floor(Training_Data_Percentage_Used*len(ANN_HeatInput_Train_DF))]

    else:
        AggregatedTest_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Test_File_Name), "rb")
        AggregatedTest_DF = pickle.load(AggregatedTest_Dict_File)

        AggregatedTrain_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Train_File_Name), "rb")
        AggregatedTrain_DF = pickle.load(AggregatedTrain_Dict_File)

        PHVAC_Regression_Model_File_Path = os.path.join(Sim_ProcessedData_FolderPath_Regression, PHVAC_Regression_Model_File_Name)
        PHVAC_Regression_Model = tf.keras.models.load_model(PHVAC_Regression_Model_File_Path)

        # Get Required Files from Sim_RegressionModelData_FolderPath
        ANN_HeatInput_Test_DF_File = open(os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Test_DF_File_Name),
                                          "rb")
        ANN_HeatInput_Test_DF = pickle.load(ANN_HeatInput_Test_DF_File)

        ANN_HeatInput_Train_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Train_DF_File_Name), "rb")
        ANN_HeatInput_Train_DF = pickle.load(ANN_HeatInput_Train_DF_File)

    # =============================================================================
    # Creating Sim_ANNModelData Folder
    # =============================================================================

    # Making Additional Folders for storing Aggregated Files
    Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..', 'Results',
                                                         'Processed_BuildingSim_Data')

    Sim_ANNModelData_FolderName = 'Sim_ANNModelData'

    # Checking if Folders Exist if not create Folders
    if (
            os.path.isdir(
                os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName))):

        # Folders Exist
        z = None

    else:

        os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName))

    # Creating Sim_RegressionModelData Folder Path
    Sim_ANNModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name,
                                               Sim_ANNModelData_FolderName)

    # =============================================================================
    # Basic Computation
    # =============================================================================

    # Computing QZic and QZir Train

    # Initialization
    QZic_Train = []
    QZir_Train = []
    QZic_Test = []
    QZir_Test = []
    QSol1_Test = []
    QSol1_Train = []
    QSol2_Test = []
    QSol2_Train = []
    QAC_Test = []
    QAC_Train = []

    # FOR LOOP: Getting Summation
    for ii in range(ANN_HeatInput_Train_DF.shape[0]):
        QZic_Train_1 = ANN_HeatInput_Train_DF['QZic_P'][ii][0] + ANN_HeatInput_Train_DF['QZic_L'][ii][0] + \
                       ANN_HeatInput_Train_DF['QZic_EE'][ii][0]
        QZir_Train_1 = ANN_HeatInput_Train_DF['QZir_P'][ii][0] + ANN_HeatInput_Train_DF['QZir_L'][ii][0] + \
                       ANN_HeatInput_Train_DF['QZir_EE'][ii][0] + ANN_HeatInput_Train_DF['QZivr_L'][ii][0]
        QZic_Train.append(QZic_Train_1)
        QZir_Train.append(QZir_Train_1)

        QSol1_Train_1 = ANN_HeatInput_Train_DF['QSol1'][ii][0]
        QSol2_Train_1 = ANN_HeatInput_Train_DF['QSol2'][ii][0]
        # QAC_Train_1 = ANN_HeatInput_Train_DF['QAC'][ii][0]
        QAC_Train_1 = AggregatedTrain_DF['QHVAC_X'].iloc[ii]

        QSol1_Train.append(QSol1_Train_1)
        QSol2_Train.append(QSol2_Train_1)
        QAC_Train.append(QAC_Train_1)

    ANN_HeatInput_Train_DF.insert(2, 'QZic', QZic_Train)
    ANN_HeatInput_Train_DF.insert(2, 'QZir', QZir_Train)
    ANN_HeatInput_Train_DF.insert(2, 'QSol1_Corrected', QSol1_Train)
    ANN_HeatInput_Train_DF.insert(2, 'QSol2_Corrected', QSol2_Train)
    ANN_HeatInput_Train_DF.insert(2, 'QAC_Corrected', QAC_Train)

    # FOR LOOP: Getting Summation
    for ii in range(ANN_HeatInput_Test_DF.shape[0]):
        QZic_Test_1 = ANN_HeatInput_Test_DF['QZic_P'][ii][0] + ANN_HeatInput_Test_DF['QZic_L'][ii][0] + \
                      ANN_HeatInput_Test_DF['QZic_EE'][ii][0]
        QZir_Test_1 = ANN_HeatInput_Test_DF['QZir_P'][ii][0] + ANN_HeatInput_Test_DF['QZir_L'][ii][0] + \
                      ANN_HeatInput_Test_DF['QZir_EE'][ii][0] + ANN_HeatInput_Test_DF['QZivr_L'][ii][0]
        QZic_Test.append(QZic_Test_1)
        QZir_Test.append(QZir_Test_1)

        QSol1_Test_1 = ANN_HeatInput_Test_DF['QSol1'][ii][0]
        QSol2_Test_1 = ANN_HeatInput_Test_DF['QSol2'][ii][0]
        # QAC_Test_1 = ANN_HeatInput_Test_DF['QAC'][ii][0]
        QAC_Test_1 = AggregatedTest_DF['QHVAC_X'].iloc[ii]

        QSol1_Test.append(QSol1_Test_1)
        QSol2_Test.append(QSol2_Test_1)
        QAC_Test.append(QAC_Test_1)

    ANN_HeatInput_Test_DF.insert(2, 'QZic', QZic_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QZir', QZir_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QSol1_Corrected', QSol1_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QSol2_Corrected', QSol2_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QAC_Corrected', QAC_Test)

    # Training and Testing X and Y
    if (FeatureType == 0):

        # Train Data
        # Creating New Feature: Temperature Difference (Derivative Information)
        AggregatedTrain_DF_New1 = AggregatedTrain_DF.iloc[1:,:]
        AggregatedTrain_DF_New2 = AggregatedTrain_DF.iloc[:-1,:]

        # Temperature_Difference = AggregatedTrain_DF_New1['Zone_Air_Temperature_'] - AggregatedTrain_DF_New2['Zone_Air_Temperature_']
        # AggregatedTrain_DF_New1.insert(2, 'Temp_Difference', Temperature_Difference, allow_duplicates=False)
        AggregatedTrain_DF_New1.reset_index(drop=True, inplace=True)
        AggregatedTrain_DF_New2.reset_index(drop=True, inplace=True)

        # Computing Temperature Difference
        AggregatedTrain_DF_New1['Temp_Difference'] = AggregatedTrain_DF_New1['Zone_Air_Temperature_'] - AggregatedTrain_DF_New2['Zone_Air_Temperature_']

        # Test Data
        # Creating New Feature: Temperature Difference (Derivative Information)
        AggregatedTest_DF_New1 = AggregatedTest_DF.iloc[1:, :]
        AggregatedTest_DF_New2 = AggregatedTest_DF.iloc[:-1, :]

        # Temperature_Difference = AggregatedTest_DF_New1['Zone_Air_Temperature_'] - AggregatedTest_DF_New2['Zone_Air_Temperature_']
        # AggregatedTest_DF_New1.insert(2, 'Temp_Difference', Temperature_Difference, allow_duplicates=False)
        AggregatedTest_DF_New1.reset_index(drop=True, inplace=True)
        AggregatedTest_DF_New2.reset_index(drop=True, inplace=True)

        # Computing Temperature Difference
        AggregatedTest_DF_New1['Temp_Difference'] = AggregatedTest_DF_New1['Zone_Air_Temperature_'] - AggregatedTest_DF_New2['Zone_Air_Temperature_']

        # Creating Train and Test X,Y
        # Train Data
        ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
        AggregatedTrain_DF.reset_index(drop=True, inplace=True)

        ANN_HeatInput_Train_DF_Concat = ANN_HeatInput_Train_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']].iloc[1:-1,:]
        AggregatedTrain_DF_New1_Concat = AggregatedTrain_DF_New1[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_', 'Temp_Difference']].iloc[:-1, :]

        ANN_HeatInput_Train_DF_Concat.reset_index(drop=True, inplace=True)
        AggregatedTrain_DF_New1_Concat.reset_index(drop=True, inplace=True)
        
        Train_X = pd.concat([ANN_HeatInput_Train_DF_Concat,AggregatedTrain_DF_New1_Concat],axis=1)
        Train_Y = AggregatedTrain_DF['Zone_Air_Temperature_'].iloc[2:]

        # Test Data
        AggregatedTest_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)

        ANN_HeatInput_Test_DF_Concat = ANN_HeatInput_Test_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir','QAC_Corrected']].iloc[1:-1, :]
        AggregatedTest_DF_New1_Concat = AggregatedTest_DF_New1[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_','Temp_Difference']].iloc[:-1, :]

        ANN_HeatInput_Test_DF_Concat.reset_index(drop=True, inplace=True)
        AggregatedTest_DF_New1_Concat.reset_index(drop=True, inplace=True)

        Test_X = pd.concat([ANN_HeatInput_Test_DF_Concat, AggregatedTest_DF_New1_Concat], axis=1)

        Test_Y = AggregatedTest_DF['Zone_Air_Temperature_'].iloc[2:]

        # Resetting Index
        Train_X.reset_index(drop=True, inplace=True)
        Train_Y.reset_index(drop=True, inplace=True)
        Test_X.reset_index(drop=True, inplace=True)
        Test_Y.reset_index(drop=True, inplace=True)


    elif (FeatureType == 1):

        ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
        AggregatedTrain_DF.reset_index(drop=True, inplace=True)
        Train_X = pd.concat(
            [ANN_HeatInput_Train_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QAC_Corrected']].iloc[:-1,
             :],
             AggregatedTrain_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]],
            axis=1)
        Train_Y = AggregatedTrain_DF['Zone_Air_Temperature_'].iloc[1:]

        AggregatedTest_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
        Test_X = pd.concat(
            [ANN_HeatInput_Test_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QAC_Corrected']].iloc[:-1,
             :],
             AggregatedTest_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]],
            axis=1)
        Test_Y = AggregatedTest_DF['Zone_Air_Temperature_'].iloc[1:]

    elif (FeatureType == 2):

        ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
        AggregatedTrain_DF.reset_index(drop=True, inplace=True)
        Train_X = pd.concat(
            [ANN_HeatInput_Train_DF[['QZic', 'QZir', 'QAC_Corrected']].iloc[:-1,
             :],
             AggregatedTrain_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]],
            axis=1)
        Train_Y = AggregatedTrain_DF['Zone_Air_Temperature_'].iloc[1:]

        AggregatedTest_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
        Test_X = pd.concat(
            [ANN_HeatInput_Test_DF[['QZic', 'QZir', 'QAC_Corrected']].iloc[:-1,
             :],
             AggregatedTest_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]],
            axis=1)
        Test_Y = AggregatedTest_DF['Zone_Air_Temperature_'].iloc[1:]

    elif (FeatureType == 3):

        ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
        AggregatedTrain_DF.reset_index(drop=True, inplace=True)
        Train_X = pd.concat(
            [ANN_HeatInput_Train_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']].iloc[:-1,
             :],
             AggregatedTrain_DF[['Zone_Air_Temperature_']].iloc[:-1, :]],
            axis=1)
        Train_Y = AggregatedTrain_DF['Zone_Air_Temperature_'].iloc[1:]

        AggregatedTest_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
        Test_X = pd.concat(
            [ANN_HeatInput_Test_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']].iloc[:-1,
             :],
             AggregatedTest_DF[['Zone_Air_Temperature_']].iloc[:-1, :]],
            axis=1)
        Test_Y = AggregatedTest_DF['Zone_Air_Temperature_'].iloc[1:]

    elif (FeatureType == 4):

        ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
        AggregatedTrain_DF.reset_index(drop=True, inplace=True)
        Train_X = pd.concat(
            [ANN_HeatInput_Train_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir']].iloc[:-1,
             :],
             AggregatedTrain_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]],
            axis=1)
        Train_Y = AggregatedTrain_DF['Zone_Air_Temperature_'].iloc[1:]

        AggregatedTest_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
        Test_X = pd.concat(
            [ANN_HeatInput_Test_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir']].iloc[:-1,
             :],
             AggregatedTest_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]],
            axis=1)
        Test_Y = AggregatedTest_DF['Zone_Air_Temperature_'].iloc[1:]

    elif(FeatureType == 5):

        ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
        AggregatedTrain_DF.reset_index(drop=True, inplace=True)
        Train_X = pd.concat(
            [ANN_HeatInput_Train_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']].iloc[:-1,
             :],
             AggregatedTrain_DF[['Site_Outdoor_Air_Drybulb_Temperature_']].iloc[:-1, :]],
            axis=1)
        Train_Y = AggregatedTrain_DF['Zone_Air_Temperature_'].iloc[1:]

        AggregatedTest_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
        Test_X = pd.concat(
            [ANN_HeatInput_Test_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']].iloc[:-1,
             :],
             AggregatedTest_DF[['Site_Outdoor_Air_Drybulb_Temperature_']].iloc[:-1, :]],
            axis=1)
        Test_Y = AggregatedTest_DF['Zone_Air_Temperature_'].iloc[1:]

    elif (FeatureType == 6):

        ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
        AggregatedTrain_DF.reset_index(drop=True, inplace=True)
        Train_X = pd.concat(
            [ANN_HeatInput_Train_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QAC_Corrected', 'QZic', 'QZir']].iloc[:-1,
             :],
             AggregatedTrain_DF[['Site_Outdoor_Air_Drybulb_Temperature_']].iloc[:-1, :]],
            axis=1)
        Train_Y = AggregatedTrain_DF['Zone_Air_Temperature_'].iloc[1:]

        AggregatedTest_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
        Test_X = pd.concat(
            [ANN_HeatInput_Test_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QAC_Corrected', 'QZic', 'QZir']].iloc[:-1,
             :],
             AggregatedTest_DF[['Site_Outdoor_Air_Drybulb_Temperature_']].iloc[:-1, :]],
            axis=1)
        Test_Y = AggregatedTest_DF['Zone_Air_Temperature_'].iloc[1:]

    # =============================================================================
    # ANN Modelling
    # =============================================================================

    # Initializing Dataframe for Percentage Accuracy
    ANN_PercentageAccuracy_DF = pd.DataFrame(columns=['ANN Model Name', 'Training Accuracy', 'Testing Accuracy'])

    # Creating Training and Validation Sets
    Train_X1 = Train_X[0:math.floor(Train_X.shape[0] * (1 - Validation_Split))]
    Train_Y1 = Train_Y[0:math.floor(Train_Y.shape[0] * (1 - Validation_Split))]

    Train_Index = Train_X1.shape[0]

    Val_X1 = Train_X[Train_Index:Train_Index + math.floor(Train_X.shape[0] * (Validation_Split))]
    Val_Y1 = Train_Y[Train_Index:Train_Index + math.floor(Train_Y.shape[0] * (Validation_Split))]

    # Converting Train_X and Train_Y into Tensor
    Train_X_TF = tf.convert_to_tensor(Train_X1)
    Train_Y_TF = tf.convert_to_tensor(Train_Y1)

    # Converting Val_X and Val_Y into Tensor
    Val_X_TF = tf.convert_to_tensor(Val_X1)
    Val_Y_TF = tf.convert_to_tensor(Val_Y1)

    # Converting Text_X into Tensor
    Test_X_TF = tf.convert_to_tensor(Test_X)

    # Prepare the Training Dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((Train_X_TF, Train_Y_TF))
    train_dataset = train_dataset.shuffle(buffer_size=Buffer_Input).batch(Batch_Size)

    # Prepare the Validation Dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((Val_X_TF, Val_Y_TF))
    val_dataset = val_dataset.batch(Batch_Size)

    # Creating Normalization Layer
    Train_X_Array = np.array(Train_X_TF)

    Normalization_Layer = tf.keras.layers.Normalization(axis=-1)
    Normalization_Layer.adapt(Train_X_Array)

    tf.keras.regularizers.L2(
        l2=0.01)

    # Creating ANN Model
    Current_ANNModel = tf.keras.Sequential([
        Normalization_Layer,
        # layers.Dense(units=1)
        layers.Dense(10, activation='relu', kernel_regularizer='l2'),
        # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
        # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
        # layers.Dense(30, activation='relu'),
        # layers.Dense(30, activation='relu'),
        # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
        # layers.LSTM(10, return_sequences=True, return_state=True),
        layers.Dense(1)
    ])

    # Instantiate an Optimizer
    optimizer = keras.optimizers.SGD(learning_rate=Learning_Rate)

    # Instantiate a Loss Function.
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Printing out the Summary of Regression Model
    Current_ANNModel.summary()

    # Prepare the Metrics.
    train_acc_metric = keras.metrics.MeanSquaredError()
    val_acc_metric = keras.metrics.MeanSquaredError()

    Training_Loss_Value_Set = []
    Val_Loss_Value_Set = []
    Training_Error_Value_Set = []
    Val_Error_Value_Set = []

    for epoch in range(Epochs):

        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = Current_ANNModel(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, Current_ANNModel.trainable_weights)
            optimizer.apply_gradients(zip(grads, Current_ANNModel.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            # if step % 200 == 0:
            # print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))
            # print("Seen so far: %d samples" % ((step + 1) * Batch_Size))

        # training_loss_value[counter] = loss_value
        Training_Loss_Value_Set = tf.experimental.numpy.append(Training_Loss_Value_Set, loss_value)

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        # print("Training acc over epoch: %.4f" % (float(train_acc),))

        # training_error_value[counter] = train_acc
        Training_Error_Value_Set = tf.experimental.numpy.append(Training_Error_Value_Set, train_acc)

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = Current_ANNModel(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_logits)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        # print("Validation acc: %.4f" % (float(val_acc),))
        # print("Time taken: %.2fs" % (time.time() - start_time))

        Val_Loss_Value_Set = tf.experimental.numpy.append(Val_Loss_Value_Set, val_loss_value)
        Val_Error_Value_Set = tf.experimental.numpy.append(Val_Error_Value_Set, val_acc)

    Test_X_TF_Array = Test_X_TF.numpy()
    Test_X_TF_Array = np.reshape(Test_X_TF_Array, (Test_X_TF_Array.shape[0], Test_X_TF_Array.shape[1]))
    Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0]))

    # Initializing the Simulation
    PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0],1))

    Tz_Current = AggregatedTest_DF_New1['Zone_Air_Temperature_'].iloc[0]
    Tz_Diff_Current = AggregatedTest_DF_New1['Zone_Air_Temperature_'].iloc[0]
    Ts = AggregatedTest_DF_New1['System_Node_Temperature_'].to_numpy()
    M_Dot = AggregatedTest_DF_New1['System_Node_Mass_Flow_Rate_'].to_numpy()
    Ca = 1.004


    # LOOP: For Loop Simulation
    for jj in range(Test_X_TF_Array.shape[0]):

        # Computing QHVAC from Predicted Temperature
        QHVAC_Computed = Ca * M_Dot[jj] * (Ts[jj] - Tz_Current)

        # Creating X for Current Timestep
        Test_X_TF_Array[jj,4] = QHVAC_Computed
        Test_X_TF_Array[jj,6] = Tz_Current
        Test_X_TF_Array[jj,7] = Tz_Diff_Current

        # Predicting Next Timestep Temperature
        Test_Y_Predict[jj] = Current_ANNModel(Test_X_TF_Array[jj,:])

        # Computing PHVAC Current
        QHVAC_Computed_Array = np.abs(np.reshape(QHVAC_Computed,(1,1)))
        PHVAC_Current[jj,0] = PHVAC_Regression_Model(QHVAC_Computed_Array)

        # Feedback Step
        Tz_Diff_Current = Test_Y_Predict[jj] - Tz_Current
        Tz_Current = Test_Y_Predict[jj]


    # Computing PHVAC
    PHVAC = PHVAC + PHVAC_Current

    # Predicting on Training and Testing Set Using Trained Model
    Test_Y_Predict1 = Current_ANNModel.predict(Test_X)
    Train_Y_Predict1 = Current_ANNModel.predict(Train_X)
    PHVAC1_Current = PHVAC_Regression_Model.predict(Test_X['QAC_Corrected'])

    PHVAC1 = PHVAC1 + PHVAC1_Current

    # Computing Percentage Accuracy of the Model without Simulation
    Train_PercentageAccuracy = (np.absolute((np.mean(Train_Y_Predict1) - np.mean(Train_Y.to_numpy()))) / np.mean(
        Train_Y.to_numpy())) * 100
    Test_PercentageAccuracy = (np.absolute((np.mean(Test_Y_Predict1) - np.mean(Test_Y.to_numpy()))) / np.mean(
        Test_Y.to_numpy())) * 100

    # Computing Percentage Accuracy of the Model with Simulation
    Test_PercentageAccuracy_Sim = (np.absolute((np.mean(Test_Y_Predict) - np.mean(Test_Y.to_numpy()))) / np.mean(
        Test_Y.to_numpy())) * 100

    Train_PercentageAccuracy = Train_PercentageAccuracy.tolist()
    Test_PercentageAccuracy = Test_PercentageAccuracy.tolist()
    Test_PercentageAccuracy_Sim = Test_PercentageAccuracy_Sim.tolist()


    # Appending Percentage Accuracy into Table
    ANN_PercentageAccuracy_Current_DF = pd.DataFrame([[ANNModel_Key, Train_PercentageAccuracy, Test_PercentageAccuracy, Test_PercentageAccuracy_Sim]],
                                                     columns=['ANN Model Name', 'Training Accuracy', 'Testing Accuracy Without Simulation', 'Testing Accuracy With Simulation'])
    ANN_PercentageAccuracy_DF = pd.concat([ANN_PercentageAccuracy_DF, ANN_PercentageAccuracy_Current_DF], ignore_index=True)

    ## Plotting ##


    # Pair Plots for Training Data
    sns.pairplot(Train_X, diag_kind='kde')
    plt.gcf().set_size_inches(10, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_PairPlot' + '.png'), dpi=300)
    # plt.show()
    plt.close()

    # Training Plot
    plt.figure()
    plt.plot(Training_Loss_Value_Set, label='Loss')
    plt.plot(Val_Loss_Value_Set, label='Val_Loss')
    plt.title('Loss Plot: ' + ANNModel_Key + ' Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss', labelpad=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_LossPlot' + '.png'))
    # plt.show()
    plt.close()

    # Error Plot
    plt.figure()
    plt.plot(Training_Error_Value_Set, label='Loss')
    plt.plot(Val_Error_Value_Set, label='Val_Loss')
    plt.title('Error Plot: ' + ANNModel_Key + ' Model')
    plt.xlabel('Epoch')
    plt.ylabel('Error', labelpad=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_ErrorPlot' + '.png'))
    # plt.show()
    plt.close()

    # Prediction Plot without Simulation
    plt.figure()
    plt.plot(Test_Y_Predict1[0:2016], color='g', label='Predicted Temp')
    plt.plot(Test_Y[0:2016], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_PredictionPlot_withoutSim' + '.png'))
    #plt.show()
    plt.close()

    # Prediction Plot with Simulation
    plt.figure()
    plt.plot(Test_Y_Predict[0:2016], color='r', label='Predicted Temp')
    plt.plot(Test_Y[0:2016], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath,
                             ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(
                                 kk) + '_PredictionPlot_withSim' + '.png'))
    # plt.show()
    plt.close()

    # =============================================================================
    # Creating ANN Model Output Data in Sim_ANNModelData Folder
    # =============================================================================

    # Saving Output Data
    Predict_Y_List = np.reshape(Test_Y_Predict, (np.shape(Test_Y_Predict)[0])).tolist()
    Actual_Y_List = Test_Y.tolist()

    Predict_Actual_Y_Dict = {'Predict_Y': Predict_Y_List, 'Actual_Y': Actual_Y_List}
    Predict_Actual_Y_DF = pd.DataFrame(Predict_Actual_Y_Dict)

    Predict_Actual_Y_DF_File_Name = 'Predict_Actual_Y_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'



    # =============================================================================
    # Storing ANN Model Data in Sim_ANNModelData Folder
    # =============================================================================

    # Saving the Accuracy Table
    ANN_Model_Accuracy_File_Name = 'ANN_Model_Accuracy_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.csv'
    ANN_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, ANN_Model_Accuracy_File_Name), index=False)

    # Saving ANN Model Output Data as a .pickle File in Results Folder
    pickle.dump(Predict_Actual_Y_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_Y_DF_File_Name), "wb"))

    '''
    # Saving Sim_ANNModelData as a .pickle File in Results Folder
    pickle.dump(ANN_HeatInput_Train_DF, open(os.path.join(Sim_ANNModelData_FolderPath, "ANN_HeatInput_Train_DF.pickle"), "wb"))
    pickle.dump(ANN_HeatInput_Test_DF, open(os.path.join(Sim_ANNModelData_FolderPath, "ANN_HeatInput_Test_DF.pickle"), "wb"))
    '''


# =============================================================================
# PHVAC without Simulation: PHVAC1
# =============================================================================


# PHVAC Plot
plt.figure()
plt.plot(PHVAC1[0:2016], label='PHVAC Computed without Sim')
plt.plot(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:2016], label='PHVAC Actual')
plt.title('PHVAC Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Time')
plt.ylabel('PHVAC', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + 'PHVACPlot_withoutSim' + '.png'))
# plt.show()
plt.close()


# Computing Percentage Accuracy of the PHVAC
PHVAC1_PercentageAccuracy = (np.absolute((np.mean(PHVAC1) - np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].to_numpy()))) / np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].to_numpy())) * 100
PHVAC1_PercentageAccuracy = PHVAC1_PercentageAccuracy.tolist()

# Appending Percentage Accuracy into Table
PHVAC1_PercentageAccuracy_DF = pd.DataFrame([[ANNModel_Key, PHVAC1_PercentageAccuracy]],columns=['ANN Model Name', 'PHVAC1_PercentageAccuracy'])

Predict_Y_List = np.reshape(PHVAC1, (np.shape(PHVAC1)[0])).tolist()
Actual_Y_List = AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].tolist()
Actual_Y_List = Actual_Y_List[1:-1]

Predict_Actual_PHVAC1_Dict = {'Predict_Y': Predict_Y_List, 'Actual_Y': Actual_Y_List}
Predict_Actual_PHVAC1_DF = pd.DataFrame(Predict_Actual_PHVAC1_Dict)

Predict_Actual_PHVAC1_DF_File_Name = 'Predict_Actual_PHVAC_withoutSim_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.pickle'

# Saving PHVAC Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_PHVAC1_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_PHVAC1_DF_File_Name), "wb"))

# Saving the Accuracy Table as a .csv File in Results Folder
PHVAC1_Accuracy_File_Name = 'PHVAC_Model_Accuracy_withoutSim_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
PHVAC1_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, PHVAC1_Accuracy_File_Name), index=False)




# =============================================================================
# PHVAC from Simulation
# =============================================================================


# PHVAC Plot
plt.figure()
plt.plot(PHVAC[0:2016], label='PHVAC Computed with Sim')
plt.plot(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:2016], label='PHVAC Actual')
plt.title('PHVAC Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Time')
plt.ylabel('PHVAC', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + 'PHVACPlot_withSim' + '.png'))
# plt.show()
plt.close()


# Computing Percentage Accuracy of the PHVAC
PHVAC_PercentageAccuracy = (np.absolute((np.mean(PHVAC) - np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].to_numpy()))) / np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].to_numpy())) * 100
PHVAC_PercentageAccuracy = PHVAC_PercentageAccuracy.tolist()

# Appending Percentage Accuracy into Table
PHVAC_PercentageAccuracy_DF = pd.DataFrame([[ANNModel_Key, PHVAC_PercentageAccuracy]],columns=['ANN Model Name', 'PHVAC_PercentageAccuracy'])

Predict_Y_List = np.reshape(PHVAC, (np.shape(PHVAC)[0])).tolist()
Actual_Y_List = AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].tolist()
Actual_Y_List = Actual_Y_List[1:-1]

Predict_Actual_PHVAC_Dict = {'Predict_Y': Predict_Y_List, 'Actual_Y': Actual_Y_List}
Predict_Actual_PHVAC_DF = pd.DataFrame(Predict_Actual_PHVAC_Dict)

Predict_Actual_PHVAC_DF_File_Name = 'Predict_Actual_PHVAC_withSim_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.pickle'

# Saving PHVAC Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_PHVAC_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_PHVAC_DF_File_Name), "wb"))

# Saving the Accuracy Table as a .csv File in Results Folder
PHVAC_Accuracy_File_Name = 'PHVAC_Model_Accuracy_withSim_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
PHVAC_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, PHVAC_Accuracy_File_Name), index=False)

