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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import datetime
import math
import time
import tracemalloc



# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Shoter_ResultsPath = True  # True - Results are stored in shorter path , False - Results are stored in the appropriate directory

ANN_Type = 2  # 1 - ANN , 2 - Simple RNN , 3 - LSTM , 4 - GRU

Lag_Number = 4  # Number of lags for input X to predict one step future Y

TimeStepper_ModelType = 1  # 1 - Model Predicts Next State Directly ; 2 - Model Computes Current Derivative

Learning_Rate = 0.001

Loss_Function = 'mean_squared_error'

Epochs = 1

Batch_Size = 2

Buffer_Input = 1000

Validation_Split = 0.2

## User Input: Aggregation Unit Number ##
# Aggregation_UnitNumber = 1

Total_Aggregation_Zone_Number = 1

# FeatureType = 0 # 0 - Remove no features, 1 - Remove Internal Heat, 2 - Remove Solar Heat, 3 - Remove Ambient Temp, 4 - Remove HVAC Heat, 5 - Remove Zone Temperature, 6 - Remove all but Zone Temperature, internal heat

# Aggregation Zone NameStem Input
Aggregation_Zone_NameStem = 'Aggregation_Zone'

# ANNModel_Key = 'ANN_Model'

# Percentage Training Data to be used
Training_Data_Control = 1 # 0 = All Data, 1 = Not All Data
Training_Data_Percentage_Used = 0.001 # Values Between 0 and 1
Testing_Points = 100
Training_Points = 1000

if (ANN_Type == 1):

    # Renaming Ann ModelKey
    ANNModel_Key =  "MLP_L_" + str(Lag_Number) + "_TS_" + str(TimeStepper_ModelType)

elif (ANN_Type == 2):

    # Renaming Ann ModelKey
    ANNModel_Key =  "SimpleRNN_" + str(Lag_Number) + "_TS_" + str(TimeStepper_ModelType)

elif (ANN_Type == 3):

    # Renaming Ann ModelKey
    ANNModel_Key =  "LSTM_" + str(Lag_Number) + "_TS_" + str(TimeStepper_ModelType)

elif (ANN_Type == 4):

    # Renaming Ann ModelKey
    ANNModel_Key =  "GRU_" + str(Lag_Number) + "_TS_" + str(TimeStepper_ModelType)

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

    Aggregation_UnitNumber = kk

    print("Current Unit Number: " + str(kk))

    # Creating Required File Names

    Aggregation_DF_Test_File_Name = 'Aggregation_DF_Test_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    Aggregation_DF_Train_File_Name = 'Aggregation_DF_Train_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    ANN_HeatInput_Test_DF_File_Name = 'ANN_HeatInput_Test_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    ANN_HeatInput_Train_DF_File_Name = 'ANN_HeatInput_Train_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    PHVAC_Regression_Model_File_Name = 'QAC_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk)

    # Get Required Files from Sim_AggregatedTestTrainData_FolderPath

    #IF Loop to Control the Amount of Training Data
    if (Training_Data_Control == 1):

        AggregatedTest_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Test_File_Name), "rb")
        AggregatedTest_DF = pickle.load(AggregatedTest_Dict_File)

        AggregatedTest_DF = AggregatedTest_DF[0:Testing_Points]

        AggregatedTrain_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Train_File_Name), "rb")
        AggregatedTrain_DF = pickle.load(AggregatedTrain_Dict_File)

        # AggregatedTrain_DF = AggregatedTrain_DF[0:math.floor(Training_Data_Percentage_Used*len(AggregatedTrain_DF))]

        AggregatedTrain_DF = AggregatedTrain_DF[0:Training_Points]

        PHVAC_Regression_Model_File_Path = os.path.join(Sim_ProcessedData_FolderPath_Regression,
                                                        PHVAC_Regression_Model_File_Name)
        PHVAC_Regression_Model = tf.keras.models.load_model(PHVAC_Regression_Model_File_Path)

        # Get Required Files from Sim_RegressionModelData_FolderPath
        ANN_HeatInput_Test_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Test_DF_File_Name),
            "rb")
        ANN_HeatInput_Test_DF = pickle.load(ANN_HeatInput_Test_DF_File)

        ANN_HeatInput_Test_DF = ANN_HeatInput_Test_DF[0:Testing_Points]

        ANN_HeatInput_Train_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Train_DF_File_Name), "rb")
        ANN_HeatInput_Train_DF = pickle.load(ANN_HeatInput_Train_DF_File)

        # ANN_HeatInput_Train_DF = ANN_HeatInput_Train_DF[0:math.floor(Training_Data_Percentage_Used*len(ANN_HeatInput_Train_DF))]

        ANN_HeatInput_Train_DF = ANN_HeatInput_Train_DF[0:Training_Points]

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

    if (Shoter_ResultsPath == False):

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
    
    elif (Shoter_ResultsPath == True):
    
        ## Shorter Path
        
        Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..')

        Sim_ANNModelData_FolderName = 'ANN_Results_Project_CS570'

        Sim_ANNModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath,
                                                Sim_ANNModelData_FolderName)

    # =============================================================================
    # Basic Computation
    # =============================================================================

    # Getting DateTime Data
    DateTime_Train = AggregatedTrain_DF['DateTime']
    DateTime_Test = AggregatedTest_DF['DateTime']

    # Resetting
    ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
    ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)

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
        # print(ii)
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

    ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
    AggregatedTrain_DF.reset_index(drop=True, inplace=True)
    Train_X = pd.concat(
        [ANN_HeatInput_Train_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']].iloc[:-1,
            :],
            AggregatedTrain_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]],
        axis=1)
    Train_Y = AggregatedTrain_DF['Zone_Air_Temperature_'].iloc[Lag_Number:]

    AggregatedTest_DF.reset_index(drop=True, inplace=True)
    ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
    Test_X = pd.concat(
        [ANN_HeatInput_Test_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']].iloc[:-1,
            :],
            AggregatedTest_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]],
        axis=1)
    Test_Y = AggregatedTest_DF['Zone_Air_Temperature_'].iloc[Lag_Number:]

    # Getting Test_Y for MDot and Ts
    Test_X_M_Dot_Ts = pd.concat([AggregatedTest_DF['System_Node_Mass_Flow_Rate_'].iloc[:-1,], AggregatedTest_DF['System_Node_Temperature_'].iloc[:-1,] ], axis=1)
    
    # =============================================================================
    # ANN Data Formating
    # =============================================================================

    # Initializing Dataframe for Percentage Accuracy
    ANN_PercentageAccuracy_DF = pd.DataFrame(columns=['ANN Model Name', 'Training Mean Error', 'Testing Mean Error Without Simulation', 'Testing Mean Error With Simulation', 'Final Train Error', 'Final Train Loss', 'Final Val Error', 'Final Val Loss', 'Time/Iteration'])

    if (ANN_Type == 1):        

        # Converting Train_X and Train_Y into Tensor
        Train_X_TF_WithoutLag = tf.convert_to_tensor(Train_X)
        Train_Y_TF_1 = tf.convert_to_tensor(Train_Y)

        # Converting Text_X into Tensor
        Test_X_TF_WithoutLag = tf.convert_to_tensor(Test_X)
        Test_Y_TF = tf.convert_to_tensor(Test_Y)

        # Converting Test_X_M_Dot_Ts to Tensor
        Test_X_M_Dot_Ts_TF_WithoutLag = tf.convert_to_tensor(Test_X_M_Dot_Ts)

        # Creating Train/Test X's based on lags        
        for ii in range(Train_X_TF_WithoutLag.shape[0]-(Lag_Number-1)):

            for jj in range(Lag_Number):
                if (jj==0):
                    Current_TF = tf.reshape(Train_X_TF_WithoutLag[ii+jj,:],[1,Train_X_TF_WithoutLag.shape[1]])
                else:
                    Current_TF = tf.concat([Current_TF,tf.reshape(Train_X_TF_WithoutLag[ii+jj,:],[1,Train_X_TF_WithoutLag.shape[1]])],axis=1)

            if (ii==0):
                Train_X_TF_1 = Current_TF
            else:
                Train_X_TF_1 = tf.concat([Train_X_TF_1,Current_TF],axis=0)

        for ii in range(Test_X_TF_WithoutLag.shape[0]-(Lag_Number-1)):
            
            for jj in range(Lag_Number):
                if (jj==0):
                    Current_TF = tf.reshape(Test_X_TF_WithoutLag[ii+jj,:],[1,Test_X_TF_WithoutLag.shape[1]])
                else:
                    Current_TF = tf.concat([Current_TF,tf.reshape(Test_X_TF_WithoutLag[ii+jj,:],[1,Test_X_TF_WithoutLag.shape[1]])],axis=1)

            if (ii==0):
                Test_X_TF = Current_TF
            else:
                Test_X_TF = tf.concat([Test_X_TF,Current_TF],axis=0)

        for ii in range(Test_X_M_Dot_Ts_TF_WithoutLag.shape[0]-(Lag_Number-1)):
            
            for jj in range(Lag_Number):
                if (jj==0):
                    Current_TF = tf.reshape(Test_X_M_Dot_Ts_TF_WithoutLag[ii+jj,:],[1,Test_X_M_Dot_Ts_TF_WithoutLag.shape[1]])
                else:
                    Current_TF = tf.concat([Current_TF,tf.reshape(Test_X_M_Dot_Ts_TF_WithoutLag[ii+jj,:],[1,Test_X_M_Dot_Ts_TF_WithoutLag.shape[1]])],axis=1)

            if (ii==0):
                Test_X_M_Dot_Ts_TF = Current_TF
            else:
                Test_X_M_Dot_Ts_TF = tf.concat([Test_X_M_Dot_Ts_TF,Current_TF],axis=0)
        
        # Creating Training and Validation Sets
        Train_X_TF = Train_X_TF_1[0:math.floor(Train_X_TF_1.shape[0] * (1 - Validation_Split)),:]
        Train_Y_TF = Train_Y_TF_1[0:math.floor(Train_Y_TF_1.shape[0] * (1 - Validation_Split))]

        Train_Index = Train_X_TF.shape[0]

        Val_X_TF = Train_X_TF_1[Train_Index:Train_Index + math.floor(Train_X_TF_1.shape[0] * (Validation_Split)),:]
        Val_Y_TF = Train_Y_TF_1[Train_Index:Train_Index + math.floor(Train_Y_TF_1.shape[0] * (Validation_Split))]

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

    elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):

        # Scaling Data
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()

        Scaler_Train_X = scaler_X.fit(Train_X.to_numpy())
        Scaler_Train_Y = scaler_Y.fit(Train_Y.to_numpy().reshape(-1,1))   
        
        Train_X_Scaled = Scaler_Train_X.transform(Train_X.to_numpy())
        Train_Y_Scaled = Scaler_Train_Y.transform(Train_Y.to_numpy().reshape(-1,1))

        # Creating Training and Validation Sets
        Train_X1 = Train_X_Scaled[0:math.floor(Train_X.shape[0] * (1 - Validation_Split))]
        Train_Y1 = Train_Y_Scaled[0:math.floor(Train_Y.shape[0] * (1 - Validation_Split))]

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
        Test_X_TF_1 = tf.convert_to_tensor(Test_X)

        # Converting Test_X_M_Dot_Ts to Tensor
        Test_X_M_Dot_Ts_TF_1 = tf.convert_to_tensor(Test_X_M_Dot_Ts)

        # Creating Train_Dataset and Val_Dataset manually
        Train_Dataset_List = []
        Val_Dataset_List = []

        """ for ii in range(Train_X_TF.shape[0]):

            if (((Batch_Size*Lag_Number)*ii+(Batch_Size*Lag_Number)) <= Train_X_TF.shape[0]):

                Train_X_TF_batch = Train_X_TF[((Batch_Size*Lag_Number)*ii):((Batch_Size*Lag_Number)*ii+(Batch_Size*Lag_Number)),:]

                Train_X_TF_batch = tf.reshape(Train_X_TF_batch, [Batch_Size, Lag_Number, Train_X_TF.shape[1]])

                Train_Y_TF_batch = Train_Y_TF[(Batch_Size*ii):(Batch_Size*ii+Batch_Size)]

                Train_Dataset_List.append((Train_X_TF_batch, Train_Y_TF_batch))

        for ii in range(Val_X_TF.shape[0]):

            if (((Batch_Size*Lag_Number)*ii+(Batch_Size*Lag_Number)) <= Val_X_TF.shape[0]):

                Val_X_TF_batch = Val_X_TF[((Batch_Size*Lag_Number)*ii):((Batch_Size*Lag_Number)*ii+(Batch_Size*Lag_Number)),:]

                Val_X_TF_batch = tf.reshape(Val_X_TF_batch, [Batch_Size, Lag_Number, Val_X_TF.shape[1]])

                Val_Y_TF_batch = Val_Y_TF[(Batch_Size*ii):(Batch_Size*ii+Batch_Size)]

                Val_Dataset_List.append((Val_X_TF_batch, Val_Y_TF_batch))

        for ii in range(Test_X_TF_1.shape[0]):

            if (((Lag_Number)*ii+(Lag_Number)) <= Test_X_TF_1.shape[0]):

                Test_X_TF_batch = Test_X_TF_1[((Lag_Number)*ii):((Lag_Number)*ii+(Lag_Number)),:]

                Test_X_TF_batch = tf.reshape(Test_X_TF_batch, [1, Lag_Number, Test_X_TF_1.shape[1]])

                if (ii == 0):

                    Test_X_TF = Test_X_TF_batch

                else:

                    Test_X_TF = tf.concat([Test_X_TF, Test_X_TF_batch], axis=0)

        for ii in range(Test_X_M_Dot_Ts_TF_1.shape[0]):

            if (((Lag_Number)*ii+(Lag_Number)) <= Test_X_M_Dot_Ts_TF_1.shape[0]):

                Test_X_M_Dot_Ts_TF_batch = Test_X_M_Dot_Ts_TF_1[((Lag_Number)*ii):((Lag_Number)*ii+(Lag_Number)),:]

                Test_X_M_Dot_Ts_TF_batch = tf.reshape(Test_X_M_Dot_Ts_TF_batch, [1, Lag_Number, Test_X_M_Dot_Ts_TF_1.shape[1]])

                if (ii == 0):

                    Test_X_M_Dot_Ts_TF = Test_X_TF_batch

                else:

                    Test_X_M_Dot_Ts_TF = tf.concat([Test_X_M_Dot_Ts_TF, Test_X_TF_batch], axis=0) """


        for ii in range(Train_X_TF.shape[0]):

            if ((ii+(Batch_Size*Lag_Number)) <= Train_X_TF.shape[0]):

                Train_X_TF_batch = Train_X_TF[(ii):(ii+(Batch_Size*Lag_Number)),:]

                Train_X_TF_batch = tf.reshape(Train_X_TF_batch, [Batch_Size, Lag_Number, Train_X_TF.shape[1]])

                Train_Y_TF_batch = Train_Y_TF[(ii):(ii+Batch_Size)]

                Train_Dataset_List.append((Train_X_TF_batch, Train_Y_TF_batch))

        for ii in range(Val_X_TF.shape[0]):

            if ((ii+(Batch_Size*Lag_Number)) <= Val_X_TF.shape[0]):

                Val_X_TF_batch = Val_X_TF[(ii):(ii+(Batch_Size*Lag_Number)),:]

                Val_X_TF_batch = tf.reshape(Val_X_TF_batch, [Batch_Size, Lag_Number, Val_X_TF.shape[1]])

                Val_Y_TF_batch = Val_Y_TF[(ii):(ii+Batch_Size)]

                Val_Dataset_List.append((Val_X_TF_batch, Val_Y_TF_batch))

        for ii in range(Train_X_TF.shape[0]):

            if ((ii+(Lag_Number)) <= Train_X_TF.shape[0]):

                Train_X_TF_batch = Train_X_TF[(ii):(ii+(Lag_Number)),:]

                Train_X_TF_batch = tf.reshape(Train_X_TF_batch, [1, Lag_Number, Train_X_TF.shape[1]])

                if (ii == 0):

                    Train_X_TF_11 = Train_X_TF_batch

                else:

                    Train_X_TF_11 = tf.concat([Train_X_TF_11, Train_X_TF_batch], axis=0)

        for ii in range(Test_X_TF_1.shape[0]):

            if ((ii+(Lag_Number)) <= Test_X_TF_1.shape[0]):

                Test_X_TF_batch = Test_X_TF_1[(ii):(ii+(Lag_Number)),:]

                Test_X_TF_batch = tf.reshape(Test_X_TF_batch, [1, Lag_Number, Test_X_TF_1.shape[1]])

                if (ii == 0):

                    Test_X_TF = Test_X_TF_batch

                else:

                    Test_X_TF = tf.concat([Test_X_TF, Test_X_TF_batch], axis=0)

        for ii in range(Test_X_M_Dot_Ts_TF_1.shape[0]):

            if ((ii+(Lag_Number)) <= Test_X_M_Dot_Ts_TF_1.shape[0]):

                Test_X_M_Dot_Ts_TF_batch = Test_X_M_Dot_Ts_TF_1[(ii):(ii+(Lag_Number)),:]

                Test_X_M_Dot_Ts_TF_batch = tf.reshape(Test_X_M_Dot_Ts_TF_batch, [1, Lag_Number, Test_X_M_Dot_Ts_TF_1.shape[1]])

                if (ii == 0):

                    Test_X_M_Dot_Ts_TF = Test_X_M_Dot_Ts_TF_batch

                else:

                    Test_X_M_Dot_Ts_TF = tf.concat([Test_X_M_Dot_Ts_TF, Test_X_M_Dot_Ts_TF_batch], axis=0)

        train_dataset = Train_Dataset_List
        val_dataset = Val_Dataset_List

    # =============================================================================
    # ANN Modelling
    # =============================================================================
    
    # Defining Regularizer
    tf.keras.regularizers.L2(l2=0.01)

    # Creating ANN Model
    if (ANN_Type == 1): # MLP

        # Creating Model
        Current_ANNModel = tf.keras.Sequential([
            Normalization_Layer,
            # layers.Dense(units=1)
            layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(20, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(2, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            # layers.LSTM(10, return_sequences=True, return_state=True),
            layers.Dense(1)
        ])

    elif (ANN_Type == 2): # Simple RNN

        # Creating Model
        Current_ANNModel = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            layers.SimpleRNN(7*Lag_Number, input_shape=(Lag_Number,Train_X_TF.shape[1]), return_sequences=False),
            #layers.Dense(7, activation='relu', kernel_regularizer='l2'),
            layers.Dense(units=1, activation='linear')
        ])

    elif (ANN_Type == 3): # LSTM

         # Creating Model
        Current_ANNModel = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            layers.LSTM(7*Lag_Number, input_shape=(Lag_Number,Train_X_TF.shape[1]), return_sequences=False),
            layers.Dense(units=1, activation='linear')
        ])

    elif (ANN_Type == 4): # GRU

        # Creating Model
        Current_ANNModel = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            # layers.GRU(21, input_shape=(1,Train_X_TF.shape[1]), return_sequences=True, kernel_regularizer='l2'),
            # layers.GRU(14, input_shape=(1,Train_X_TF.shape[1]), return_sequences=True, kernel_regularizer='l2'),
            layers.GRU(7*Lag_Number, input_shape=(Lag_Number,Train_X_TF.shape[1]), return_sequences=False),
            layers.Dense(units=1, activation='linear')
        ])

    # =============================================================================
    # ANN Training
    # =============================================================================

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
        #for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            with tf.GradientTape() as tape:

                if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                    # Computing Model Output
                    logits = Current_ANNModel(x_batch_train, training=True)

                elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                    if (ANN_Type == 1): # ANN

                        # Getting Tz_Previous
                        x_batch_train_Tz_Prev = tf.cast(tf.reshape(x_batch_train[:,(Lag_Number*7)-1],[x_batch_train.shape[0],1]), tf.float32)

                        # Computing Model Output
                        logits = x_batch_train_Tz_Prev + Current_ANNModel(x_batch_train, training=True)

                    elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):  # LSTM, RNN, GRU

                        # Getting Tz_Previous
                        x_batch_train_Tz_Prev = tf.cast(tf.reshape(x_batch_train[:,Lag_Number-1,6],[x_batch_train.shape[0],1]), tf.float32)

                        # Computing Model Output
                        logits = x_batch_train_Tz_Prev + Current_ANNModel(x_batch_train, training=True)
                
                # Compute Loss
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
        #for x_batch_val, y_batch_val in val_dataset:
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

    # Getting Final Train/Val Loss/Error
    Final_Train_Loss = Training_Loss_Value_Set.numpy()[-1]
    Final_Train_Error = Training_Error_Value_Set.numpy()[-1]
    Final_Val_Loss = Val_Loss_Value_Set.numpy()[-1]
    Final_Val_Error = Val_Error_Value_Set.numpy()[-1]

    # =============================================================================
    # ANN Testing
    # =============================================================================
    
    if (ANN_Type == 1):

        # Converting Test_X tensors to numpy arrays
        Test_X_TF_Array = Test_X_TF.numpy()
        Test_X_TF_Array = np.reshape(Test_X_TF_Array, (Test_X_TF_Array.shape[0], Test_X_TF_Array.shape[1]))
        
        Test_X_M_Dot_Ts_TF_Array = Test_X_M_Dot_Ts_TF.numpy()
        Test_X_M_Dot_Ts_TF_Array = np.reshape(Test_X_M_Dot_Ts_TF_Array, (Test_X_M_Dot_Ts_TF_Array.shape[0], Test_X_M_Dot_Ts_TF_Array.shape[1]))

        Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0]))

        # Initializing the Simulation
        PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0],1))

        # Getting Appropriate Indices
        Tz_Indices = list(range(6, Test_X_TF_Array.shape[1]+1, 7))
        QHVAC_Indices = list(range(4, Test_X_TF_Array.shape[1]+1, 7))
        M_Dot_Indices = list(range(0, Test_X_M_Dot_Ts_TF_Array.shape[1], 2))
        Ts_Indices = list(range(1, Test_X_M_Dot_Ts_TF_Array.shape[1]+1, 2))

        # Initializing Initial Temp and Control Signals
        Tz_Current = Test_X_TF_Array[0,Tz_Indices]
        M_Dot = Test_X_M_Dot_Ts_TF_Array[:,M_Dot_Indices]
        Ts = Test_X_M_Dot_Ts_TF_Array[:,Ts_Indices]
        
        Ca = 1.004

        # Timing Simulation
        Sim_StartTime = time.time()

        # LOOP: Simulation for Loop with Considering Bias into Account
        for jj in range(Test_X_TF_Array.shape[0]):

            # Computing QHVAC from Predicted Temperature
            QHVAC_Computed = Ca * M_Dot[jj,:] * (Ts[jj,:] - Tz_Current)

            # Creating X for Current Timestep
            Test_X_TF_Array[jj,QHVAC_Indices] = QHVAC_Computed
            Test_X_TF_Array[jj,Tz_Indices] = Tz_Current

            # Predicting Next Timestep Temperature
            if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                Test_Y_Predict[jj] = Current_ANNModel(Test_X_TF_Array[jj,:])

            elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                Test_Y_Predict[jj] = Test_X_TF_Array[jj,-1] + Current_ANNModel(Test_X_TF_Array[jj,:])

            # Computing PHVAC Current
            QHVAC_Computed_Array = np.abs(np.reshape(QHVAC_Computed[-1],(1,1)))
            PHVAC_Current[jj,0] = PHVAC_Regression_Model(QHVAC_Computed_Array)

            # Feedback Step
            Tz_Current[:-1] = Tz_Current[1:]
            Tz_Current[-1] = Test_Y_Predict[jj]


        # Timing Simulation
        Sim_EndTime = time.time()

        SimTime = (Sim_EndTime - Sim_StartTime)/(Test_X_TF_Array.shape[0])

        # Computing PHVAC
        PHVAC = PHVAC + PHVAC_Current

        # Predicting on Training and Testing Set Using Trained Model without Simulation
        if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

            Test_Y_Predict1 = Current_ANNModel.predict(Test_X_TF_Array)
            Train_Y_Predict1 = Current_ANNModel.predict(Train_X_TF)

        elif (TimeStepper_ModelType == 2):  # Model Computes Current Derivative
            
            Test_Y_Predict1 = tf.reshape(Test_X_TF_Array[:,-1],[Test_X_TF_Array[:,-1].shape[0],1]) + Current_ANNModel.predict(Test_X_TF_Array)
            Train_Y_Predict1 =  tf.reshape(Train_X_TF[:,-1],[Train_X_TF[:,-1].shape[0],1]) + Current_ANNModel.predict(Train_X_TF)

        PHVAC1_Current = PHVAC_Regression_Model.predict(Test_X['QAC_Corrected'].iloc[Lag_Number-1:].abs())

        PHVAC1 = PHVAC1 + PHVAC1_Current


    elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):

        # Converting Test_X tensors to numpy arrays
        Test_X_TF_Array = Test_X_TF.numpy()
        Test_X_TF_Array = np.reshape(Test_X_TF_Array, (Test_X_TF_Array.shape[0], Test_X_TF_Array.shape[1], Test_X_TF_Array.shape[2]))
        
        Test_X_M_Dot_Ts_TF_Array = Test_X_M_Dot_Ts_TF.numpy()
        Test_X_M_Dot_Ts_TF_Array = np.reshape(Test_X_M_Dot_Ts_TF_Array, (Test_X_M_Dot_Ts_TF_Array.shape[0], Test_X_M_Dot_Ts_TF_Array.shape[1], Test_X_M_Dot_Ts_TF_Array.shape[2]))

        Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0]))

        # Initializing the Simulation
        PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0],1))

        # Initializing Initial Temp and Control Signals
        Tz_Current = Test_X_TF_Array[0,:,6]
        M_Dot = Test_X_M_Dot_Ts_TF_Array[:,:,0]
        Ts = Test_X_M_Dot_Ts_TF_Array[:,:,1]
        
        Ca = 1.004

        # Timing Simulation
        Sim_StartTime = time.time()

        # LOOP: For Loop Simulation
        for jj in range(Test_X_TF_Array.shape[0]):

            # Computing QHVAC from Predicted Temperature
            QHVAC_Computed = Ca * M_Dot[jj,:] * (Ts[jj,:] - Tz_Current)

            # Creating X for Current Timestep
            Test_X_TF_Array[jj,:,4] = QHVAC_Computed
            Test_X_TF_Array[jj,:,6] = Tz_Current

            # Scaling of Input
            Test_X_Scaled = Scaler_Train_X.transform(Test_X_TF_Array[jj,:,:].reshape([Lag_Number, Test_X_TF_Array.shape[2]]))

            # Reshaping Input for RNN
            Test_X_TF_Array_Reshaped = tf.reshape(Test_X_Scaled, [1,Lag_Number,Test_X_TF_Array.shape[2]])

            # Predicting Next Timestep Temperature
            if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                Test_Y_Predict[jj] = Current_ANNModel(Test_X_TF_Array_Reshaped)

            elif (TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                Test_Y_Predict[jj] = tf.cast(tf.reshape(Test_X_TF_Array_Reshaped[0,Lag_Number-1,6],[1,1]), tf.float32) + Current_ANNModel(Test_X_TF_Array_Reshaped)

            # Undo Scaling of Output
            Test_Y_Predict[jj] = Scaler_Train_Y.inverse_transform(np.array(Test_Y_Predict[jj]).reshape(1,-1))

            # Computing PHVAC Current
            QHVAC_Computed_Array = np.abs(np.reshape(QHVAC_Computed[-1],(1,1)))
            PHVAC_Current[jj,0] = PHVAC_Regression_Model(QHVAC_Computed_Array)

            # Feedback Step
            Tz_Current[:-1] = Tz_Current[1:]
            Tz_Current[-1] = Test_Y_Predict[jj]

        
        # Timing Simulation
        Sim_EndTime = time.time()

        SimTime = (Sim_EndTime - Sim_StartTime)/(Test_X_TF_Array.shape[0])

        # Computing PHVAC 
        PHVAC = PHVAC + PHVAC_Current

        # Predicting on Training and Testing Set Using Trained Model
        # Test_X_TF_Scaled = Scaler_Train_X.transform(Test_X_TF)

        for kk in range(Test_X_TF.shape[0]):

            Test_X_TF_Scaled_batch = Scaler_Train_X.transform(Test_X_TF[kk,:,:])

            if (kk == 0):

                Test_X_TF_reshaped= tf.reshape(Test_X_TF_Scaled_batch, [1,Lag_Number,Test_X_TF.shape[2]])

            else:

                Test_X_TF_reshaped = tf.concat([Test_X_TF_reshaped, tf.reshape(Test_X_TF_Scaled_batch, [1,Lag_Number,Test_X_TF.shape[2]])], axis=0)
        
        if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

            Test_Y_Predict1 = Current_ANNModel.predict(Test_X_TF_reshaped)           
            Train_Y_Predict1 = Current_ANNModel.predict(Train_X_TF_11)

            Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(Test_Y_Predict1.reshape(-1,1))
            Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(Train_Y_Predict1.reshape(-1,1))

        elif (TimeStepper_ModelType == 2):  # Model Computes Current Derivative            

            Test_Y_Predict1 = tf.reshape(Test_X_TF_reshaped[:,Lag_Number-1,6],[Test_X_TF_reshaped[:,Lag_Number-1,6].shape[0],1]) + Current_ANNModel.predict(Test_X_TF_reshaped)           
            Train_Y_Predict1 = tf.reshape(Train_X_TF_11[:,Lag_Number-1,6],[Train_X_TF_11[:,Lag_Number-1,6].shape[0],1])  + Current_ANNModel.predict(Train_X_TF_11)

            Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(tf.reshape(Test_Y_Predict1,[-1,1]))
            Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(tf.reshape(Train_Y_Predict1,[-1,1]))

        PHVAC1_Current = PHVAC_Regression_Model.predict(Test_X['QAC_Corrected'].iloc[Lag_Number-1:].abs()) ## Created problems in PHVAC without SIm Results

        PHVAC1 = PHVAC1 + PHVAC1_Current

        Train_Y_TF = Scaler_Train_Y.inverse_transform(tf.reshape(Train_Y_TF, [-1,1]))    

    # Computing Percentage Accuracy of the Model without Simulation
    Train_PercentageAccuracy = np.mean((np.absolute((Train_Y_Predict1-np.reshape(Train_Y_TF,(Train_Y_Predict1.shape[0],1)))/(np.reshape(Train_Y_TF,(Train_Y_Predict1.shape[0],1))))) *100)

    Test_PercentageAccuracy = np.mean((np.absolute((Test_Y_Predict1-np.reshape(Test_Y.to_numpy(),(Test_Y_Predict1.shape[0],1)))/(np.reshape(Test_Y.to_numpy(),(Test_Y_Predict1.shape[0],1))))) *100)

    # Computing Percentage Accuracy of the Model with Simulation
    Test_PercentageAccuracy_Sim = np.mean((np.absolute((Test_Y_Predict-np.reshape(Test_Y.to_numpy(),(Test_Y_Predict.shape[0],1)))/(np.reshape(Test_Y.to_numpy(),(Test_Y_Predict.shape[0],1))))) *100)


    Train_PercentageAccuracy = Train_PercentageAccuracy.tolist()
    Test_PercentageAccuracy = Test_PercentageAccuracy.tolist()
    Test_PercentageAccuracy_Sim = Test_PercentageAccuracy_Sim.tolist()


    # Appending Percentage Accuracy into Table
    ANN_PercentageAccuracy_Current_DF = pd.DataFrame([[ANNModel_Key, Train_PercentageAccuracy, Test_PercentageAccuracy, Test_PercentageAccuracy_Sim, Final_Train_Error, Final_Train_Loss, Final_Val_Error, Final_Val_Loss, SimTime]],columns=['ANN Model Name', 'Training Mean Error', 'Testing Mean Error Without Simulation', 'Testing Mean Error With Simulation', 'Final Train Error', 'Final Train Loss', 'Final Val Error', 'Final Val Loss', 'Time/Iteration'])
    ANN_PercentageAccuracy_DF = pd.concat([ANN_PercentageAccuracy_DF, ANN_PercentageAccuracy_Current_DF], ignore_index=True)

    # =============================================================================
    # Results Plotting
    # =============================================================================

    # Pair Plots for Training Data
    sns.pairplot(Train_X, diag_kind='kde')
    plt.gcf().set_size_inches(10, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_PairPlot' + '.png'), dpi=300)
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
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_LossPlot' + '.png'))
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
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_ErrorPlot' + '.png'))
    # plt.show()
    plt.close()

    # Prediction Plot for Training
    plt.figure()
    # plt.plot(Test_Y_Predict1[0:], color='g', label='Predicted Temp')
    # plt.plot(Test_Y[0:], color='b', label='Actual Temp', linestyle='dashed')
    plt.plot(Train_Y_Predict1, color='r', label='Predicted Temp')
    plt.plot(Train_Y_TF, color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_PredictionPlot_Training' + '.png'))
    #plt.show()
    plt.close()

    # Prediction Plot without Simulation
    plt.figure()
    # plt.plot(Test_Y_Predict1[0:], color='g', label='Predicted Temp')
    # plt.plot(Test_Y[0:], color='b', label='Actual Temp', linestyle='dashed')
    plt.plot(Test_Y_Predict1[0:Testing_Points-Lag_Number], color='r', label='Predicted Temp')
    plt.plot(list(range(0,Test_Y.shape[0],1)), Test_Y[0:Testing_Points-Lag_Number], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_PredictionPlot_withoutSim' + '.png'))
    #plt.show()
    plt.close()

    # Prediction Plot with Simulation
    plt.figure()
    # plt.plot(Test_Y_Predict[0:], color='r', label='Predicted Temp')
    # plt.plot(Test_Y[0:], color='b', label='Actual Temp', linestyle='dashed')
    plt.plot(Test_Y_Predict[0:Testing_Points-Lag_Number], color='r', label='Predicted Temp')
    plt.plot(list(range(0,Test_Y.shape[0],1)), Test_Y[0:Testing_Points-Lag_Number], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath,
                             ANNModel_Key +  '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(
                                 kk) + '_PredictionPlot_withSim' + '.png'))
    #plt.show()
    plt.close()

    # =============================================================================
    # Creating ANN Model Output Data in Sim_ANNModelData Folder
    # =============================================================================

    # Saving Output Data
    Predict_Y_List = np.reshape(Test_Y_Predict, (np.shape(Test_Y_Predict)[0])).tolist()
    Actual_Y_List = Test_Y.tolist()

    Predict_Actual_Y_Dict = {'DateTime': DateTime_Test.tolist()[Lag_Number:], 'Predict_Y': Predict_Y_List, 'Actual_Y': Actual_Y_List}
    Predict_Actual_Y_DF = pd.DataFrame(Predict_Actual_Y_Dict)

    Predict_Actual_Y_DF_File_Name = ANNModel_Key + '_Predict_Actual_Y_DF' + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_withSim' + '.pickle'

    # Saving Output Data without Simulation
    Predict_Y_List1 = np.transpose(np.reshape(Test_Y_Predict1, (np.shape(Test_Y_Predict1)[0]))).tolist()
    Actual_Y_List1 = Test_Y.tolist()

    Predict_Actual_Y_Dict1 = {'DateTime': DateTime_Test.tolist()[Lag_Number:], 'Predict_Y': Predict_Y_List1, 'Actual_Y': Actual_Y_List1}
    Predict_Actual_Y_DF1 = pd.DataFrame(Predict_Actual_Y_Dict1)

    Predict_Actual_Y_DF_File_Name1 = ANNModel_Key + '_Predict_Actual_Y_DF' + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_withoutSim' + '.pickle'



    # =============================================================================
    # Storing ANN Model Data in Sim_ANNModelData Folder
    # =============================================================================

    # Saving the Accuracy Table
    ANN_Model_Accuracy_File_Name = ANNModel_Key + '_ANN_Model_Accuracy' + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.csv'
    ANN_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, ANN_Model_Accuracy_File_Name), index=False)

    # Saving ANN Model Output Data as a .pickle File in Results Folder
    pickle.dump(Predict_Actual_Y_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_Y_DF_File_Name), "wb"))
    pickle.dump(Predict_Actual_Y_DF1,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_Y_DF_File_Name1), "wb"))

    # Saving Trained ANN Model
    ANNModel_FileName = ANNModel_Key + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(Aggregation_UnitNumber)
    Current_ANNModel.save(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_FileName))
    
    if ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):

        # Saving Data Scaler
        TrainDataX_Scaler_FileName = ANNModel_Key + '_TrainDataX_Scaler' + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(Aggregation_UnitNumber) + '.save'
        joblib.dump(Scaler_Train_X , os.path.join(Sim_ANNModelData_FolderPath, TrainDataX_Scaler_FileName)) 
        
        TrainDataY_Scaler_FileName = ANNModel_Key + '_TrainDataY_Scaler' + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(Aggregation_UnitNumber) + '.save'
        joblib.dump(Scaler_Train_Y , os.path.join(Sim_ANNModelData_FolderPath, TrainDataY_Scaler_FileName)) 
        


# =============================================================================
# PHVAC without Simulation: PHVAC1
# =============================================================================


# PHVAC Plot
plt.figure()
plt.plot(PHVAC1[0:Testing_Points-Lag_Number], label='PHVAC Computed without Sim', color='r')
plt.plot(list(range(0,AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Lag_Number-1:Testing_Points-1].shape[0],1)), AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Lag_Number-1:Testing_Points-1], label='PHVAC Actual', color='b', linestyle='dashed')
plt.title('PHVAC Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Time')
plt.ylabel('PHVAC', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + 'PHVACPlot_withoutSim' + '.png'))
# plt.show()
plt.close()


# Computing Percentage Accuracy of the PHVAC
PHVAC1_PercentageAccuracy = np.mean(np.absolute((PHVAC1-np.reshape(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Lag_Number-1:-1].to_numpy(),(PHVAC1.shape[0],1)))/(np.reshape(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Lag_Number-1:-1].to_numpy(),(PHVAC1.shape[0],1)))) *100)
PHVAC1_PercentageAccuracy = PHVAC1_PercentageAccuracy.tolist()

# Appending Percentage Accuracy into Table
PHVAC1_PercentageAccuracy_DF = pd.DataFrame([[ANNModel_Key, PHVAC1_PercentageAccuracy]],columns=['ANN Model Name', 'PHVAC1_Percentage_Mean_Error'])

Predict_Y_List = np.reshape(PHVAC1, (np.shape(PHVAC1)[0])).tolist()
Actual_Y_List = AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].tolist()
Actual_Y_List = Actual_Y_List[Lag_Number-1:-1]

Predict_Actual_PHVAC1_Dict = {'DateTime': DateTime_Test.tolist()[Lag_Number-1:-1], 'Predict_Y': Predict_Y_List, 'Actual_Y': Actual_Y_List}
Predict_Actual_PHVAC1_DF = pd.DataFrame(Predict_Actual_PHVAC1_Dict)

Predict_Actual_PHVAC1_DF_File_Name = ANNModel_Key + '_Predict_Actual_PHVAC_withoutSim' + '_Ind_'+ str(Total_Aggregation_Zone_Number) + 'Zone' + '.pickle'

# Saving PHVAC Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_PHVAC1_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_PHVAC1_DF_File_Name), "wb"))

# Saving the Accuracy Table as a .csv File in Results Folder
PHVAC1_Accuracy_File_Name = ANNModel_Key + '_PHVAC_Model_Mean_Error_withoutSim'+ '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
PHVAC1_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, PHVAC1_Accuracy_File_Name), index=False)


# =============================================================================
# PHVAC from Simulation
# =============================================================================

# PHVAC Plot
plt.figure()
plt.plot(PHVAC[0:Testing_Points-Lag_Number], label='PHVAC Computed with Sim', color='r')
plt.plot(list(range(0,AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Lag_Number-1:Testing_Points-1].shape[0],1)), AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Lag_Number-1:Testing_Points-1], label='PHVAC Actual', color='b', linestyle='dashed')
plt.title('PHVAC Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Time')
plt.ylabel('PHVAC', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + 'PHVACPlot_withSim' + '.png'))
# plt.show()
plt.close()


# Computing Percentage Accuracy of the PHVAC
PHVAC_PercentageAccuracy = np.mean(np.absolute((PHVAC-np.reshape(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Lag_Number-1:-1].to_numpy(),(PHVAC.shape[0],1)))/(np.reshape(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Lag_Number-1:-1].to_numpy(),(PHVAC.shape[0],1)))) *100)
PHVAC_PercentageAccuracy = PHVAC_PercentageAccuracy.tolist()

# Appending Percentage Accuracy into Table
PHVAC_PercentageAccuracy_DF = pd.DataFrame([[ANNModel_Key, PHVAC_PercentageAccuracy]],columns=['ANN Model Name', 'PHVAC_Percentage_Mean_Error'])

Predict_Y_List = np.reshape(PHVAC, (np.shape(PHVAC)[0])).tolist()
Actual_Y_List = AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].tolist()
Actual_Y_List = Actual_Y_List[Lag_Number-1:-1]

Predict_Actual_PHVAC_Dict = {'DateTime': DateTime_Test.tolist()[Lag_Number-1:-1], 'Predict_Y': Predict_Y_List, 'Actual_Y': Actual_Y_List}
Predict_Actual_PHVAC_DF = pd.DataFrame(Predict_Actual_PHVAC_Dict)

Predict_Actual_PHVAC_DF_File_Name = ANNModel_Key + '_Predict_Actual_PHVAC_withSim'+ '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.pickle'

# Saving PHVAC Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_PHVAC_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_PHVAC_DF_File_Name), "wb"))

# Saving the Accuracy Table as a .csv File in Results Folder
PHVAC_Accuracy_File_Name = ANNModel_Key + '_PHVAC_Model_Mean_Error_withSim'+ '_Ind_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
PHVAC_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, PHVAC_Accuracy_File_Name), index=False) 

