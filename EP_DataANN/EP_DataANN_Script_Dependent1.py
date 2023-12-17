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
import copy

# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Learning_Rate = 0.001

Loss_Function = 'mean_squared_error'

Epochs = 2

Validation_Split = 0.2

## User Input: Aggregation Unit Number ##
Aggregation_UnitNumber = 1

Total_Aggregation_Zone_Number = 2

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

# Initializing Train_X, Train_Y, Test_X, Test_Y
Train_X = copy.deepcopy(pd.DataFrame())
Train_Y = copy.deepcopy(pd.DataFrame())
Test_X = copy.deepcopy(pd.DataFrame())
Test_Y = copy.deepcopy(pd.DataFrame())

Predict_Actual_Y_DF = copy.deepcopy(pd.DataFrame())

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

    ANN_HeatInput_Train_DF.insert(2, 'QZic'+str(kk), QZic_Train)
    ANN_HeatInput_Train_DF.insert(2, 'QZir'+str(kk), QZir_Train)
    ANN_HeatInput_Train_DF.insert(2, 'QSol1_Corrected'+str(kk), QSol1_Train)
    ANN_HeatInput_Train_DF.insert(2, 'QSol2_Corrected'+str(kk), QSol2_Train)
    ANN_HeatInput_Train_DF.insert(2, 'QAC_Corrected'+str(kk), QAC_Train)

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

    ANN_HeatInput_Test_DF.insert(2, 'QZic'+str(kk), QZic_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QZir'+str(kk), QZir_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QSol1_Corrected'+str(kk), QSol1_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QSol2_Corrected'+str(kk), QSol2_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QAC_Corrected'+str(kk), QAC_Test)

    AggregatedTrain_DF.rename({'Site_Outdoor_Air_Drybulb_Temperature_':'Site_Outdoor_Air_Drybulb_Temperature_'+str(kk), 'Zone_Air_Temperature_':'Zone_Air_Temperature_'+str(kk)}, axis=1, inplace=True)
    AggregatedTest_DF.rename({'Site_Outdoor_Air_Drybulb_Temperature_': 'Site_Outdoor_Air_Drybulb_Temperature_' + str(kk), 'Zone_Air_Temperature_': 'Zone_Air_Temperature_' + str(kk)}, axis=1, inplace=True)

    # Training and Testing X and Y
    ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
    AggregatedTrain_DF.reset_index(drop=True, inplace=True)
    Train_X = pd.concat(
        [Train_X, ANN_HeatInput_Train_DF[['QSol1_Corrected'+str(kk), 'QSol2_Corrected'+str(kk), 'QZic'+str(kk), 'QZir'+str(kk), 'QAC_Corrected'+str(kk)]].iloc[:-1, :],
         AggregatedTrain_DF[['Site_Outdoor_Air_Drybulb_Temperature_'+str(kk), 'Zone_Air_Temperature_'+str(kk)]].iloc[:-1, :]], axis=1)
    Train_Y = pd.concat([Train_Y, AggregatedTrain_DF['Zone_Air_Temperature_'+str(kk)].iloc[1:]], axis=1)

    AggregatedTest_DF.reset_index(drop=True, inplace=True)
    ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
    Test_X = pd.concat(
        [Test_X, ANN_HeatInput_Test_DF[['QSol1_Corrected'+str(kk), 'QSol2_Corrected'+str(kk), 'QZic'+str(kk), 'QZir'+str(kk), 'QAC_Corrected'+str(kk)]].iloc[:-1, :],
         AggregatedTest_DF[['Site_Outdoor_Air_Drybulb_Temperature_'+str(kk), 'Zone_Air_Temperature_'+str(kk)]].iloc[:-1, :]], axis=1)
    Test_Y = pd.concat([Test_Y, AggregatedTest_DF['Zone_Air_Temperature_'+str(kk)].iloc[1:]], axis=1)

# =============================================================================
# ANN Modelling
# =============================================================================

# Initializing Dataframe for Percentage Error
ANN_PercentageError_DF = copy.deepcopy(pd.DataFrame(columns=['ANN Model Name', 'Training Error', 'Testing Error']))

# Creating Normalization Layer
Train_X_Array = np.array(Train_X)

# Normalization_Layer = layers.Normalization(input_shape=[1, ], axis=None)
Normalization_Layer = tf.keras.layers.Normalization(axis=-1)
Normalization_Layer.adapt(Train_X_Array)

# Creating ANN Model
Current_ANNModel = tf.keras.Sequential([
    Normalization_Layer,
    layers.Dense(units=Total_Aggregation_Zone_Number)
    # layers.Dense(5, activation='relu'),
    # layers.Dense(5, activation='relu'),
    # layers.Dense(5, activation='relu'),
    # layers.Dense(Total_Aggregation_Zone_Number)
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

# Predicting on Training and Testing Set Using Trained Model
Test_Y_Predict = Current_ANNModel.predict(Test_X)
Train_Y_Predict = Current_ANNModel.predict(Train_X)

# Saving Current Trained ANN Model
# Current_ANNModel.save(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_TrainedModel'))

# LOOP: For Loop for Each Zone Type to Compute Results
for kk in range(Total_Aggregation_Zone_Number):

    kk1 = kk + 1

    # Computing Percentage Error of the Model
    Train_PercentageError = (np.absolute((np.mean(Train_Y_Predict[:,kk]) - np.mean(Train_Y.to_numpy()[:,kk]))) / np.mean(
        Train_Y.to_numpy()[:,kk])) * 100
    Test_PercentageError = (np.absolute((np.mean(Test_Y_Predict[:,kk]) - np.mean(Test_Y.to_numpy()[:,kk]))) / np.mean(
        Test_Y.to_numpy()[:,kk])) * 100

    Train_PercentageError = Train_PercentageError.tolist()
    Test_PercentageError = Test_PercentageError.tolist()

    # Appending Percentage Error into Table
    ANN_PercentageError_Current_DF = pd.DataFrame([[ANNModel_Key + '_Zone_' + str(kk1), Train_PercentageError, Test_PercentageError]],
                                                     columns=['ANN Model Name', 'Training Error', 'Testing Error'])
    ANN_PercentageError_DF = pd.concat([ANN_PercentageError_DF, ANN_PercentageError_Current_DF])


    ## Plotting ##

    # Prediction Plot
    plt.figure()
    plt.plot(Test_Y_Predict[0:2016,kk], color='r', label='Predicted Temp')
    plt.plot(Test_Y.iloc[0:2016,kk], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk1) + '_PredictionPlot' + '.png'))
    #plt.show()
    plt.close()

'''
# Pair Plots for Training Data
sns.pairplot(Train_X, diag_kind='kde')
plt.gcf().set_size_inches(10, 10)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + '_PairPlot' + '.png'), dpi=300)
# plt.show()
plt.close()
'''

# Training Plot
plt.figure()
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.title('Training Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Epoch')
plt.ylabel('Loss', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + '_TrainingPlot' + '.png'))
# plt.show()
plt.close()

# =============================================================================
# Creating ANN Model Output Data in Sim_ANNModelData Folder
# =============================================================================

# Saving Output Data
Predict_Y_List = np.reshape(Test_Y_Predict, (np.shape(Test_Y_Predict)[0],Total_Aggregation_Zone_Number)).tolist()
Actual_Y_DF = copy.deepcopy(Test_Y)

for kk in range(Total_Aggregation_Zone_Number):

    Key_Now = 'Predict_Y'+str(kk+1)

    Predict_Y_Dict1 = {Key_Now : Predict_Y_List[kk]}

    Predict_Actual_Y_DF = pd.concat([Predict_Actual_Y_DF,pd.DataFrame(Predict_Y_Dict1)])


Actual_Y_DF.reset_index(drop=True, inplace=True)
Predict_Actual_Y_DF.reset_index(drop=True, inplace=True)

Predict_Actual_Y_DF = pd.concat([Actual_Y_DF, Predict_Actual_Y_DF],axis=1)

Predict_Actual_Y_DF_File_Name = 'Predict_Actual_Y_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.pickle'

# =============================================================================
# Storing ANN Model Data in Sim_ANNModelData Folder
# =============================================================================

# Saving the Error Table
ANN_Model_Error_File_Name = 'ANN_Model_Error_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
ANN_PercentageError_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, ANN_Model_Error_File_Name),
                                 index=False)

# Saving ANN Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_Y_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_Y_DF_File_Name), "wb"))

'''
# Saving Sim_ANNModelData as a .pickle File in Results Folder
pickle.dump(ANN_HeatInput_Train_DF, open(os.path.join(Sim_ANNModelData_FolderPath, "ANN_HeatInput_Train_DF.pickle"), "wb"))
pickle.dump(ANN_HeatInput_Test_DF, open(os.path.join(Sim_ANNModelData_FolderPath, "ANN_HeatInput_Test_DF.pickle"), "wb"))
'''