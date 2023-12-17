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
import math
import time

# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Learning_Rate = 0.001

Loss_Function = 'mean_squared_error'

Epochs = 100

Batch_Size = 25

Buffer_Input = 1000

Validation_Split = 0.2

## User Input: Aggregation Unit Number ##
Aggregation_UnitNumber = 1

Total_Aggregation_Zone_Number = 1

# FeatureType = 0 # 0 - Remove no features, 1 - Remove Internal Heat, 2 - Remove Solar Heat, 3 - Remove Ambient Temp, 4 - Remove HVAC Heat, 5 - Remove Zone Temperature, 6 - Remove all but Zone Temperature, internal heat

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

PHVAC_NoBias = np.zeros((1,1))

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

# Initializing Train_X, Train_Y, Test_X, Test_Y
Train_X = copy.deepcopy(pd.DataFrame())
Train_Y = copy.deepcopy(pd.DataFrame())
Test_X = copy.deepcopy(pd.DataFrame())
Test_Y = copy.deepcopy(pd.DataFrame())

Train_X_QHVAC = copy.deepcopy(pd.DataFrame())
Test_X_QHVAC = copy.deepcopy(pd.DataFrame())

Predict_Actual_Y_DF = copy.deepcopy(pd.DataFrame())
Predict_Actual_Y_NoBias_DF = copy.deepcopy(pd.DataFrame())
Predict_Actual_Y_DF1 = copy.deepcopy(pd.DataFrame())

PHVAC_Regression_Model_List = []

# LOOP: Output Generation for Each Aggregated Zone

for kk in range(Total_Aggregation_Zone_Number):

    kk = kk + 1

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

        PHVAC_Regression_Model_List.append(PHVAC_Regression_Model)

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
        PHVAC_Regression_Model_List.append(PHVAC_Regression_Model)

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
        # QAC_Test_1 = ANN_HeatInput_Test_DF['QAC'][ii][0]
        QAC_Test_1 = AggregatedTest_DF['QHVAC_X'].iloc[ii]

        QSol1_Test.append(QSol1_Test_1)
        QSol2_Test.append(QSol2_Test_1)
        QAC_Test.append(QAC_Test_1)

    ANN_HeatInput_Test_DF.insert(2, 'QZic'+str(kk), QZic_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QZir'+str(kk), QZir_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QSol1_Corrected'+str(kk), QSol1_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QSol2_Corrected'+str(kk), QSol2_Test)
    ANN_HeatInput_Test_DF.insert(2, 'QAC_Corrected'+str(kk), QAC_Test)

    AggregatedTrain_DF.rename({'Zone_Air_Temperature_':'Zone_Air_Temperature_'+str(kk), 'System_Node_Temperature_':'System_Node_Temperature_'+str(kk), 'System_Node_Mass_Flow_Rate_':'System_Node_Mass_Flow_Rate_'+str(kk)}, axis=1, inplace=True)
    AggregatedTest_DF.rename({'Zone_Air_Temperature_': 'Zone_Air_Temperature_' + str(kk), 'System_Node_Temperature_':'System_Node_Temperature_'+str(kk), 'System_Node_Mass_Flow_Rate_':'System_Node_Mass_Flow_Rate_'+str(kk)}, axis=1, inplace=True)

    # Training and Testing X and Y

    ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
    AggregatedTrain_DF.reset_index(drop=True, inplace=True)
    Train_X = pd.concat([Train_X, ANN_HeatInput_Train_DF[['QSol1_Corrected'+str(kk), 'QSol2_Corrected'+str(kk), 'QZic'+str(kk), 'QZir'+str(kk), 'QAC_Corrected'+str(kk)]].iloc[:-1, :],AggregatedTrain_DF[['Zone_Air_Temperature_'+str(kk)]].iloc[:-1, :]], axis=1)
    Train_Y = pd.concat([Train_Y, AggregatedTrain_DF['Zone_Air_Temperature_'+str(kk)].iloc[1:]], axis=1)

    AggregatedTest_DF.reset_index(drop=True, inplace=True)
    ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
    Test_X = pd.concat([Test_X, ANN_HeatInput_Test_DF[['QSol1_Corrected'+str(kk), 'QSol2_Corrected'+str(kk), 'QZic'+str(kk), 'QZir'+str(kk), 'QAC_Corrected'+str(kk)]].iloc[:-1, :],AggregatedTest_DF[['Zone_Air_Temperature_'+str(kk)]].iloc[:-1, :]], axis=1)
    Test_Y = pd.concat([Test_Y, AggregatedTest_DF['Zone_Air_Temperature_'+str(kk)].iloc[1:]], axis=1)

    Train_X_QHVAC = pd.concat([Train_X_QHVAC,AggregatedTrain_DF[['System_Node_Temperature_'+str(kk),'System_Node_Mass_Flow_Rate_'+str(kk)]].iloc[1:]], axis=1)
    Test_X_QHVAC = pd.concat([Test_X_QHVAC,AggregatedTest_DF[['System_Node_Temperature_'+str(kk),'System_Node_Mass_Flow_Rate_'+str(kk)]].iloc[1:]], axis=1)

Train_X = pd.concat([Train_X, AggregatedTrain_DF['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[:-1]], axis=1)
Test_X = pd.concat([Test_X, AggregatedTest_DF['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[:-1]], axis=1)


# =============================================================================
# ANN Modelling
# =============================================================================

# Initializing Dataframe for Percentage Accuracy
ANN_PercentageAccuracy_DF = copy.deepcopy(pd.DataFrame(columns=['ANN Model Name', 'Training Accuracy', 'Testing Accuracy']))

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
    layers.Dense(100, activation='relu', kernel_regularizer='l2'),
    # layers.Dense(5 0, activation='relu', kernel_regularizer='l2'),
    # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
    # layers.Dense(30, activation='relu'),
    # layers.Dense(30, activation='relu'),
    # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
    # layers.LSTM(10, return_sequences=True, return_state=True),
    layers.Dense(Total_Aggregation_Zone_Number)
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

Mid_Index = math.floor(Test_X_TF_Array.shape[0] / 2)
Test_X_Rest = Test_X[Mid_Index:]
Test_X_TF_Array_Bias = Test_X_TF_Array[0:Mid_Index]
Test_X_TF_Array_Rest = Test_X_TF_Array[Mid_Index:]
Test_X_Rest.reset_index(drop=True, inplace=True)


Test_Y_Bias = Test_Y[0:Mid_Index]
Test_Y_Rest = Test_Y[Mid_Index:]
Test_Y_Rest.reset_index(drop=True, inplace=True)

Test_Y_Predict_Bias = np.zeros((Test_X_TF_Array_Bias.shape[0],Total_Aggregation_Zone_Number))
Test_Y_Predict_NoBias = np.zeros((Test_X_TF_Array_Rest.shape[0],Total_Aggregation_Zone_Number))
Test_Y_Predict = np.zeros((Test_X_TF_Array_Rest.shape[0],Total_Aggregation_Zone_Number))
# Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0]))

# Initializing the Simulation
PHVAC_Current_Bias = np.zeros((Test_X_TF_Array_Bias.shape[0], Total_Aggregation_Zone_Number))
PHVAC_Current = np.zeros((Test_X_TF_Array_Rest.shape[0], Total_Aggregation_Zone_Number))
PHVAC_Current_NoBias = np.zeros((Test_X_TF_Array_Rest.shape[0], Total_Aggregation_Zone_Number))
# PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0],1))

Tz_Current = np.zeros((1, Total_Aggregation_Zone_Number))
Ts = np.zeros((len(Test_X), Total_Aggregation_Zone_Number))
M_Dot = np.zeros((len(Test_X), Total_Aggregation_Zone_Number))

QHVAC_X_Index_Set = [4]
Tz_X_Index_Set = [6]

# LOOP: For Loop to get the Parameter for Simulation
for kk in range(Total_Aggregation_Zone_Number):

    kk1 = kk + 1

    Tz_Current[0, kk] = Test_X['Zone_Air_Temperature_'+str(kk1)].iloc[0]
    Ts[:, kk] = Test_X_QHVAC['System_Node_Temperature_'+str(kk1)].to_numpy()
    M_Dot[:, kk] = Test_X_QHVAC['System_Node_Mass_Flow_Rate_'+str(kk1)].to_numpy()

    if (kk != 0):
        QHVAC_X_Index_Set.append(QHVAC_X_Index_Set[kk-2]+6)
        Tz_X_Index_Set.append(Tz_X_Index_Set[kk-2]+6)


Ca = 1.004


# LOOP: For Loop Simulation
for jj in range(Test_X_TF_Array_Bias.shape[0]):

    # Computing QHVAC from Predicted Temperature
    QHVAC_Computed = Ca * M_Dot[jj,:] * (Ts[jj,:] - Tz_Current[0,:])

    # Creating X for Current Timestep
    Test_X_TF_Array_Bias[jj,QHVAC_X_Index_Set] = QHVAC_Computed
    Test_X_TF_Array_Bias[jj,Tz_X_Index_Set] = Tz_Current

    # Predicting Next Timestep Temperature
    Test_Y_Predict_Bias[jj,:] = Current_ANNModel(Test_X_TF_Array_Bias[jj,:])

    # Computing PHVAC Current
    for kk in range(Total_Aggregation_Zone_Number):

        QHVAC_Computed_Array = np.abs(np.reshape(QHVAC_Computed[kk],(1,1)))
        PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
        PHVAC_Current_Bias[jj,kk] = PHVAC_Regression_Model(QHVAC_Computed_Array)

    # Feedback Step
    Tz_Current = Test_Y_Predict_Bias[jj,:]
    Tz_Current = tf.reshape(Tz_Current,[1,Total_Aggregation_Zone_Number])


# Computing the Difference between Test_Y_Predict and Test_Y_Actual to Get the Bias
Bias = Test_Y_Predict_Bias - Test_Y_Bias.to_numpy()

# Computing the Mean of the Bias
Bias_Mean = Bias.mean(axis=0)
Bias_Mean = np.reshape(Bias_Mean,(1,Total_Aggregation_Zone_Number))

Tz_Current = np.zeros((1, Total_Aggregation_Zone_Number))
Tz_Current_NoBias = np.zeros((1, Total_Aggregation_Zone_Number))
Ts = np.zeros((Mid_Index+1, Total_Aggregation_Zone_Number))
M_Dot = np.zeros((Mid_Index+1, Total_Aggregation_Zone_Number))

QHVAC_X_Index_Set = [4]
Tz_X_Index_Set = [6]

# LOOP: For Loop to get the Parameter for Simulation
for kk in range(Total_Aggregation_Zone_Number):

    kk1 = kk + 1

    Tz_Current[0, kk] = Test_X['Zone_Air_Temperature_'+str(kk1)].iloc[Mid_Index]
    Tz_Current_NoBias[0, kk] = Test_X['Zone_Air_Temperature_' + str(kk1)].iloc[Mid_Index]
    Ts[:, kk] = Test_X_QHVAC['System_Node_Temperature_'+str(kk1)].to_numpy()[Mid_Index:]
    M_Dot[:, kk] = Test_X_QHVAC['System_Node_Mass_Flow_Rate_'+str(kk1)].to_numpy()[Mid_Index:]

    if (kk != 0):
        QHVAC_X_Index_Set.append(QHVAC_X_Index_Set[kk-2]+6)
        Tz_X_Index_Set.append(Tz_X_Index_Set[kk-2]+6)

# LOOP: Simulation for Loop with Considering Bias into Account
for jj in range(Test_X_TF_Array_Rest.shape[0]):
    # Computing QHVAC from Predicted Temperature
    QHVAC_Computed_NoBias = Ca * M_Dot[jj,:] * (Ts[jj,:] - Tz_Current_NoBias[0,:])
    QHVAC_Computed = Ca * M_Dot[jj,:] * (Ts[jj,:] - Tz_Current[0,:])

    # Creating X for Current Timestep
    Test_X_TF_Array_Rest[jj, QHVAC_X_Index_Set] = QHVAC_Computed_NoBias
    Test_X_TF_Array_Rest[jj, Tz_X_Index_Set] = Tz_Current_NoBias

    # Predicting Next Timestep Temperature
    Test_Y_Predict[jj,:] = Current_ANNModel(Test_X_TF_Array_Rest[jj, :]) - Bias_Mean[0,:]
    Test_Y_Predict_NoBias[jj,:] = Test_Y_Predict[jj] + Bias_Mean[0,:]

    # Computing PHVAC Current
    for kk in range(Total_Aggregation_Zone_Number):

        QHVAC_Computed_NoBias_Array = np.abs(np.reshape(QHVAC_Computed_NoBias[kk],(1,1)))
        PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
        PHVAC_Current_NoBias[jj,kk] = PHVAC_Regression_Model(QHVAC_Computed_NoBias_Array)

        QHVAC_Computed_Array = np.abs(np.reshape(QHVAC_Computed[kk], (1, 1)))
        PHVAC_Current[jj, kk] = PHVAC_Regression_Model(QHVAC_Computed_Array)

    # Feedback Step
    Tz_Current_NoBias = Test_Y_Predict_NoBias[jj,:]
    Tz_Current = Test_Y_Predict[jj,:]

    Tz_Current_NoBias = tf.reshape(Tz_Current_NoBias, [1, Total_Aggregation_Zone_Number])
    Tz_Current = tf.reshape(Tz_Current, [1, Total_Aggregation_Zone_Number])


# Computing PHVAC
PHVAC_NoBias = PHVAC_Current_NoBias.sum(axis=1)
PHVAC = PHVAC_Current.sum(axis=1)

# Predicting on Training and Testing Set Using Trained Model without Simulation
Test_Y_Predict1 = Current_ANNModel.predict(Test_X_Rest)
Train_Y_Predict1 = Current_ANNModel.predict(Train_X)

for kk in range(Total_Aggregation_Zone_Number):

    kk1 = kk + 1
    PHVAC1_Current = PHVAC_Regression_Model.predict(Test_X_Rest['QAC_Corrected'+ str(kk1)].abs())

    PHVAC1 = PHVAC1 + PHVAC1_Current


# LOOP: For Loop for Each Zone Type to Compute Results
for Zone_Number in range(Total_Aggregation_Zone_Number):

    Zone_Number1 = Zone_Number + 1

    # Computing Percentage Accuracy of the Model without Simulation
    Train_PercentageAccuracy = (np.absolute((np.mean(Train_Y_Predict1[:, Zone_Number]) - np.mean(Train_Y.to_numpy()[:, Zone_Number]))) / np.mean(Train_Y.to_numpy()[:, Zone_Number])) * 100
    Test_PercentageAccuracy = (np.absolute((np.mean(Test_Y_Predict1[:, Zone_Number]) - np.mean(Test_Y_Rest.to_numpy()[:, Zone_Number]))) / np.mean(Test_Y_Rest.to_numpy()[:, Zone_Number])) * 100

    # Computing Percentage Accuracy of the Model with Simulation
    # Test_Y_Predict = np.reshape(Test_Y_Predict,(Test_Y_Predict.shape[0],1))
    Test_PercentageAccuracy_Sim = (np.absolute((np.mean(Test_Y_Predict[:, Zone_Number]) - np.mean(Test_Y_Rest.to_numpy()[:, Zone_Number]))) / np.mean(Test_Y_Rest.to_numpy()[:, Zone_Number])) * 100
    Test_PercentageAccuracy_Sim_NoBias = (np.absolute((np.mean(Test_Y_Predict_NoBias[:, Zone_Number]) - np.mean(Test_Y_Rest.to_numpy()[:, Zone_Number]))) / np.mean(Test_Y_Rest.to_numpy()[:, Zone_Number])) * 100

    Train_PercentageAccuracy = Train_PercentageAccuracy.tolist()
    Test_PercentageAccuracy = Test_PercentageAccuracy.tolist()
    Test_PercentageAccuracy_Sim = Test_PercentageAccuracy_Sim.tolist()
    Test_PercentageAccuracy_Sim_NoBias = Test_PercentageAccuracy_Sim_NoBias.tolist()

    # Appending Percentage Accuracy into Table
    ANN_PercentageAccuracy_Current_DF = pd.DataFrame([[ANNModel_Key + '_Zone_' + str(Zone_Number1), Train_PercentageAccuracy, Test_PercentageAccuracy, Test_PercentageAccuracy_Sim_NoBias, Test_PercentageAccuracy_Sim]],columns=['ANN Model Name', 'Training Accuracy', 'Testing Accuracy Without Simulation','Testing Accuracy With Simulation - Bias not accounted', 'Testing Accuracy With Simulation- Bias accounted'])
    ANN_PercentageAccuracy_DF = pd.concat([ANN_PercentageAccuracy_DF, ANN_PercentageAccuracy_Current_DF],ignore_index=True)


    ## Plotting ##

    # Prediction Plot without Simulation
    plt.figure()
    plt.plot(Test_Y_Predict1[0:2016,Zone_Number], color='g', label='Predicted Temp')
    plt.plot(Test_Y_Rest.iloc[0:2016,Zone_Number], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath,ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(Zone_Number1) + '_PredictionPlot_withoutSim' + '.png'))
    # plt.show()
    plt.close()

    # Prediction Plot with Simulation
    plt.figure()
    plt.plot(Test_Y_Predict[0:2016,Zone_Number], color='g', label='Predicted Temp accounting Bias')
    plt.plot(Test_Y_Predict_NoBias[0:2016,Zone_Number], color='r', label='Predicted Temp without accounting Bias', linestyle='dashed')
    plt.plot(Test_Y_Rest.iloc[0:2016,Zone_Number], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath,ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(Zone_Number1) + '_PredictionPlot_withSim' + '.png'))
    # plt.show()
    plt.close()

# Pair Plots for Training Data
sns.pairplot(Train_X, diag_kind='kde')
plt.gcf().set_size_inches(10, 10)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_PairPlot' + '.png'), dpi=300)
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
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_LossPlot' + '.png'))
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
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_ErrorPlot' + '.png'))
# plt.show()
plt.close()

# =============================================================================
# Creating ANN Model Output Data in Sim_ANNModelData Folder
# =============================================================================

# Saving Output Data with Simulation
Predict_Y_List = np.transpose(np.reshape(Test_Y_Predict, (np.shape(Test_Y_Predict)[0],Total_Aggregation_Zone_Number))).tolist()
Predict_Y_List_NoBias = np.transpose(np.reshape(Test_Y_Predict_NoBias, (np.shape(Test_Y_Predict_NoBias)[0],Total_Aggregation_Zone_Number))).tolist()
Actual_Y_DF = copy.deepcopy(Test_Y_Rest)

for Number in range(Total_Aggregation_Zone_Number):

    Key_Now = 'Predict_Y' + str(Number+1)
    Key_Now_NoBias = 'Predict_Y_NoBias' + str(Number + 1)

    Predict_Y_Dict_Current = {Key_Now : Predict_Y_List[Number]}
    Predict_Y_NoBias_Dict_Current = {Key_Now_NoBias: Predict_Y_List_NoBias[Number]}

    Predict_Actual_Y_DF = pd.concat([Predict_Actual_Y_DF,pd.DataFrame(Predict_Y_Dict_Current)])
    Predict_Actual_Y_NoBias_DF = pd.concat([Predict_Actual_Y_NoBias_DF, pd.DataFrame(Predict_Y_NoBias_Dict_Current)])


Actual_Y_DF.reset_index(drop=True, inplace=True)
Predict_Actual_Y_DF.reset_index(drop=True, inplace=True)
Predict_Actual_Y_NoBias_DF.reset_index(drop=True, inplace=True)

Predict_Actual_Y_DF = pd.concat([Actual_Y_DF, Predict_Actual_Y_DF, Predict_Actual_Y_NoBias_DF],axis=1)

Predict_Actual_Y_DF_File_Name = 'Predict_Actual_Y_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_withSim' + '.pickle'

# Saving Output Data without Simulation
Predict_Y_List1 = np.transpose(np.reshape(Test_Y_Predict1, (np.shape(Test_Y_Predict1)[0], Total_Aggregation_Zone_Number))).tolist()
Actual_Y_DF1 = copy.deepcopy(Test_Y_Rest)

for Number in range(Total_Aggregation_Zone_Number):
    Key_Now = 'Predict_Y' + str(Number + 1)

    Predict_Y_Dict_Current = {Key_Now: Predict_Y_List1[Number]}

    Predict_Actual_Y_DF1 = pd.concat([Predict_Actual_Y_DF1, pd.DataFrame(Predict_Y_Dict_Current)])

Actual_Y_DF1.reset_index(drop=True, inplace=True)
Predict_Actual_Y_DF1.reset_index(drop=True, inplace=True)

Predict_Actual_Y_DF1 = pd.concat([Actual_Y_DF1, Predict_Actual_Y_DF1], axis=1)

Predict_Actual_Y_DF_File_Name1 = 'Predict_Actual_Y_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_withoutSim' + '.pickle'

# =============================================================================
# Storing ANN Model Data in Sim_ANNModelData Folder
# =============================================================================

# Saving the Accuracy Table
ANN_Model_Accuracy_File_Name = 'ANN_Model_Accuracy_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
ANN_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, ANN_Model_Accuracy_File_Name), index=False)

# Saving ANN Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_Y_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_Y_DF_File_Name), "wb"))
pickle.dump(Predict_Actual_Y_DF1,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_Y_DF_File_Name1), "wb"))

# Saving Trained ANN Model
ANNModel_FileName = ANNModel_Key + '_Dep1_' + str(Total_Aggregation_Zone_Number) 
Current_ANNModel.save(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_FileName))

# =============================================================================
# PHVAC without Simulation: PHVAC1
# =============================================================================

AggregatedTest_DF_Rest = AggregatedTest_DF.iloc[Mid_Index:-1]
AggregatedTest_DF_Rest.reset_index(drop=True, inplace=True)

# PHVAC Plot
plt.figure()
plt.plot(PHVAC1[0:2016], label='PHVAC Computed without Sim')
plt.plot(AggregatedTest_DF_Rest['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:2016], label='PHVAC Actual')
plt.title('PHVAC Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Time')
plt.ylabel('PHVAC', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + 'PHVACPlot_withoutSim' + '.png'))
# plt.show()
plt.close()


# Computing Percentage Accuracy of the PHVAC1 (without Simulation)
PHVAC1_PercentageAccuracy = (np.absolute((np.mean(PHVAC1) - np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Mid_Index:-1].to_numpy()))) / np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Mid_Index:-1].to_numpy())) * 100
PHVAC1_PercentageAccuracy = PHVAC1_PercentageAccuracy.tolist()

# Appending Percentage Accuracy into Table
PHVAC1_PercentageAccuracy_DF = pd.DataFrame([[ANNModel_Key, PHVAC1_PercentageAccuracy]],columns=['ANN Model Name', 'PHVAC1_PercentageAccuracy'])


Predict_Y_List = np.reshape(PHVAC1, (np.shape(PHVAC1)[0])).tolist()
Actual_Y_List = AggregatedTest_DF_Rest['Facility_Total_HVAC_Electric_Demand_Power_'].tolist()

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
plt.plot(PHVAC[0:2016], label='PHVAC Computed with Sim - Bias accounted')
plt.plot(PHVAC_NoBias[0:2016], label='PHVAC Computed with Sim - No Bias accounted')
plt.plot(AggregatedTest_DF_Rest['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:2016], label='PHVAC Actual')
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
PHVAC_PercentageAccuracy = (np.absolute((np.mean(PHVAC) - np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Mid_Index:-1].to_numpy()))) / np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Mid_Index:-1].to_numpy())) * 100
PHVAC_PercentageAccuracy = PHVAC_PercentageAccuracy.tolist()

PHVAC_PercentageAccuracy_NoBias = (np.absolute((np.mean(PHVAC_NoBias) - np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Mid_Index:-1].to_numpy()))) / np.mean(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[Mid_Index:-1].to_numpy())) * 100
PHVAC_PercentageAccuracy_NoBias = PHVAC_PercentageAccuracy_NoBias.tolist()

# Appending Percentage Accuracy into Table
PHVAC_PercentageAccuracy_DF = pd.DataFrame([[ANNModel_Key, PHVAC_PercentageAccuracy_NoBias, PHVAC_PercentageAccuracy]],columns=['ANN Model Name', 'PHVAC_PercentageAccuracy - Bias not accounted', 'PHVAC_PercentageAccuracy - Bias accounted'])

Predict_Y_List = np.reshape(PHVAC, (np.shape(PHVAC)[0])).tolist()
Predict_Y_List_NoBias = np.reshape(PHVAC_NoBias, (np.shape(PHVAC_NoBias)[0])).tolist()
Actual_Y_List = AggregatedTest_DF_Rest['Facility_Total_HVAC_Electric_Demand_Power_'].tolist()

Predict_Actual_PHVAC_Dict = {'Predict_Y_BiasNotAccounted': Predict_Y_List_NoBias, 'Predict_Y_BiasAccounted': Predict_Y_List, 'Actual_Y': Actual_Y_List}
Predict_Actual_PHVAC_DF = pd.DataFrame(Predict_Actual_PHVAC_Dict)

Predict_Actual_PHVAC_DF_File_Name = 'Predict_Actual_PHVAC_withSim_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.pickle'

# Saving PHVAC Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_PHVAC_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_PHVAC_DF_File_Name), "wb"))

# Saving the Accuracy Table as a .csv File in Results Folder
PHVAC_Accuracy_File_Name = 'PHVAC_Model_Accuracy_withSim_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
PHVAC_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, PHVAC_Accuracy_File_Name), index=False)

