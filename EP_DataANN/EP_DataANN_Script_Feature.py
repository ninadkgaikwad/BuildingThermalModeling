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

# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Learning_Rate = 0.001

Loss_Function = 'mean_squared_error'

Epochs = 10

Validation_Split = 0.2

## User Input: Aggregation Unit Number ##
Aggregation_UnitNumber = 1

Total_Aggregation_Zone_Number = 1

# Aggregation Zone NameStem Input
Aggregation_Zone_NameStem = 'Aggregation_Zone'

ANNModel_Key = 'ANN_Model'

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

    # Creating Required File Names

    Aggregation_DF_Test_File_Name = 'Aggregation_DF_Test_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    Aggregation_DF_Train_File_Name = 'Aggregation_DF_Train_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    ANN_HeatInput_Test_DF_File_Name = 'ANN_HeatInput_Test_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    ANN_HeatInput_Train_DF_File_Name = 'ANN_HeatInput_Train_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'


    # Get Required Files from Sim_AggregatedTestTrainData_FolderPath
    AggregatedTest_Dict_File = open(
        os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Test_File_Name), "rb")
    AggregatedTest_DF = pickle.load(AggregatedTest_Dict_File)

    AggregatedTrain_Dict_File = open(
        os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Train_File_Name), "rb")
    AggregatedTrain_DF = pickle.load(AggregatedTrain_Dict_File)

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
        QAC_Train_1 = ANN_HeatInput_Train_DF['QAC'][ii][0]

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
        QAC_Test_1 = ANN_HeatInput_Test_DF['QAC'][ii][0]

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
        [ANN_HeatInput_Train_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QAC_Corrected']].iloc[:-1, :],
         AggregatedTrain_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]], axis=1)
    Train_Y = AggregatedTrain_DF['Zone_Air_Temperature_'].iloc[1:]

    AggregatedTest_DF.reset_index(drop=True, inplace=True)
    ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
    Test_X = pd.concat(
        [ANN_HeatInput_Test_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QAC_Corrected']].iloc[:-1, :],
         AggregatedTest_DF[['Site_Outdoor_Air_Drybulb_Temperature_', 'Zone_Air_Temperature_']].iloc[:-1, :]], axis=1)
    Test_Y = AggregatedTest_DF['Zone_Air_Temperature_'].iloc[1:]

    # =============================================================================
    # ANN Modelling
    # =============================================================================

    # Initializing Dataframe for Percentage Accuracy
    ANN_PercentageAccuracy_DF = pd.DataFrame(columns=['ANN Model Name', 'Training Accuracy', 'Testing Accuracy'])

    # Creating Normalization Layer
    Train_X_Array = np.array(Train_X)

    # Normalization_Layer = layers.Normalization(input_shape=[1, ], axis=None)
    Normalization_Layer = tf.keras.layers.Normalization(axis=-1)
    Normalization_Layer.adapt(Train_X_Array)

    # Creating ANN Model
    Current_ANNModel = tf.keras.Sequential([
        Normalization_Layer,
        layers.Dense(1, activation='relu'),
        layers.Dense(1)
    ])

    # Printing out the Summary of ANN Model
    Current_ANNModel.summary()

    # Setting up the Optimizer for ANN Model
    Current_ANNModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate),
        loss=Loss_Function)

    # Training Current ANN Model
    history = Current_ANNModel.fit(
        Train_X,
        Train_Y,
        epochs=Epochs,
        # Suppress Logging
        verbose=0,
        # Calculating Validation Results of the Training Dataset
        validation_split=Validation_Split)

    # Saving Current Trained ANN Model
    # Current_ANNModel.save(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_TrainedModel'))


    # Predicting on Training and Testing Set Using Trained Model
    Test_Y_Predict = Current_ANNModel.predict(Test_X)
    Train_Y_Predict = Current_ANNModel.predict(Train_X)

    # Computing Percentage Accuracy of the Model
    Train_PercentageAccuracy = (np.absolute((np.mean(Train_Y_Predict) - np.mean(Train_Y.to_numpy()))) / np.mean(
        Train_Y.to_numpy())) * 100
    Test_PercentageAccuracy = (np.absolute((np.mean(Test_Y_Predict) - np.mean(Test_Y.to_numpy()))) / np.mean(
        Test_Y.to_numpy())) * 100

    Train_PercentageAccuracy = Train_PercentageAccuracy.tolist()
    Test_PercentageAccuracy = Test_PercentageAccuracy.tolist()

    # Appending Percentage Accuracy into Table
    ANN_PercentageAccuracy_Current_DF = pd.DataFrame([[ANNModel_Key, Train_PercentageAccuracy, Test_PercentageAccuracy]],
                                                     columns=['ANN Model Name', 'Training Accuracy', 'Testing Accuracy'])
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
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val_Loss')
    plt.title('Training Plot: ' + ANNModel_Key + ' Model')
    plt.xlabel('Epoch')
    plt.ylabel('Error ' + 'Zone Temperature', labelpad=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_TrainingPlot' + '.png'))
    # plt.show()
    plt.close()


    # Prediction Plot
    plt.figure()
    plt.plot(Test_Y_Predict[0:2016], color='r', label='Predicted Temp')
    plt.plot(Test_Y[0:2016], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '_PredictionPlot' + '.png'))
    #plt.show()
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