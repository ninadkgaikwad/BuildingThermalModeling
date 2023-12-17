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
import copy
import math
import time

# =============================================================================
# User Inputs
# =============================================================================

# Simulation Folder and Files
Simulation_Name = "test1"

Training_RegressionModel_File_Name1 = 'TrainingData_RegressionModel_Dict_Aggregation_Dict_2Zone_1.pickle'

Testing_RegressionModel_File_Name1 = 'TestingData_RegressionModel_Dict_Aggregation_Dict_2Zone_1.pickle'

## User Input: Aggregation Unit Number ##
Aggregation_UnitNumber_Total = 1

# Learning Parameters for Linear Regression
Learning_Rate = 0.001

Loss_Function = 'mean_squared_error'

Epochs = 10

Validation_Split = 0.2

Batch_Size = 100

Buffer_Input = 1000

# Result File Names
Regression_Model_Accuracy_File_Name1 = 'Regression_Model_Accuracy'

ANN_HeatInput_Train_File_Name1 = 'ANN_HeatInput_Train_DF'

ANN_HeatInput_Test_File_Name1 = 'ANN_HeatInput_Test_DF'

for ii in range(Aggregation_UnitNumber_Total):

    Aggregation_UnitNumber = ii + 1

    # =============================================================================
    # Creating Input File Names
    # =============================================================================
    Training_RegressionModel_File_Name = "_".join(
        Training_RegressionModel_File_Name1.split('.')[0].split('_')[:-1]) + '_' + str(
        Aggregation_UnitNumber) + '.pickle'

    Testing_RegressionModel_File_Name = "_".join(
        Testing_RegressionModel_File_Name1.split('.')[0].split('_')[:-1]) + '_' + str(
        Aggregation_UnitNumber) + '.pickle'

    # =============================================================================
    # Creating Result File Names
    # =============================================================================
    Regression_Model_Accuracy_File_Name = Regression_Model_Accuracy_File_Name1 + '_' + \
                                          Training_RegressionModel_File_Name.split('_')[5] + '_' + str(
        Aggregation_UnitNumber) + '.csv'

    ANN_HeatInput_Train_File_Name = ANN_HeatInput_Train_File_Name1 + '_' + \
                                    Training_RegressionModel_File_Name.split('_')[5] + '_' + str(
        Aggregation_UnitNumber) + '.pickle'

    ANN_HeatInput_Test_File_Name = ANN_HeatInput_Test_File_Name1 + '_' + Testing_RegressionModel_File_Name.split('_')[
        5] + '_' + str(Aggregation_UnitNumber) + '.pickle'

    PHVAC_RegressionModel_File_Name = 'PHVAC_RegressionModel_' + str(Aggregation_UnitNumber_Total) + 'Zone_' + str(
        Aggregation_UnitNumber)

    # =============================================================================
    # Getting Required Data from Sim_ProcessedData
    # =============================================================================

    # Getting Current File Directory Path
    Current_FilePath = os.path.dirname(__file__)

    # Getting Sim_TrainingTestingData Folder Path
    Sim_ProcessedData_FolderPath = os.path.join(Current_FilePath, '..', '..', 'Results', 'Processed_BuildingSim_Data',
                                                Simulation_Name, 'Sim_TrainingTestingData')

    # Get Required Files from Sim_TrainingTestingData_FolderPath
    TestingData_RegressionModel_Dict_File = open(
        os.path.join(Sim_ProcessedData_FolderPath, Testing_RegressionModel_File_Name), "rb")

    TestingData_RegressionModel_Dict = pickle.load(TestingData_RegressionModel_Dict_File)

    TrainingData_RegressionModel_Dict_File = open(
        os.path.join(Sim_ProcessedData_FolderPath, Training_RegressionModel_File_Name), "rb")

    TrainingData_RegressionModel_Dict = pickle.load(TrainingData_RegressionModel_Dict_File)

    # =============================================================================
    # Creating Sim_RegressionModelData Folder
    # =============================================================================

    # Making Additional Folders for storing Aggregated Files
    Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..', 'Results',
                                                         'Processed_BuildingSim_Data')

    Sim_RegressionModelData_FolderName = 'Sim_RegressionModelData'

    # Checking if Folders Exist if not create Folders
    if (
            os.path.isdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name,
                                       Sim_RegressionModelData_FolderName))):

        # Folders Exist
        z = None

    else:

        os.mkdir(
            os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_RegressionModelData_FolderName))

    # Creating Sim_RegressionModelData Folder Path
    Sim_RegressionModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name,
                                                      Sim_RegressionModelData_FolderName)

    # =============================================================================
    # Regression Modelling
    # =============================================================================

    # Initializating Dataframe to Store Training and Testing Data for ANN
    ANN_HeatInput_Train_DF = copy.deepcopy(pd.DataFrame())
    ANN_HeatInput_Test_DF = copy.deepcopy(pd.DataFrame())

    # Initializing Dataframe for Percentage Accuracy
    Regression_PercentageAccuracy_DF = copy.deepcopy(
        pd.DataFrame(columns=['Regression Model Name', 'Training Accuracy', 'Testing Accuracy']))

    # FOR LOOP: Creating Regression Models
    for RegressionModel_Key in TrainingData_RegressionModel_Dict:

        # Getting Training and Testing X and Y for Current Model

        # IF: For QAC
        if (RegressionModel_Key == 'QAC'):
            Train_X = TrainingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 0].abs()
            Test_X = TestingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 0].abs()
        else:
            Train_X = TrainingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 0]
            Test_X = TestingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 0]

        Train_Y = TrainingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 1]
        Test_Y = TestingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 1]

        # Creating Training and Validation Sets
        Train_X1 = Train_X[0:math.floor(Train_X.shape[0]*(1-Validation_Split))]
        Train_Y1 = Train_Y[0:math.floor(Train_Y.shape[0] * (1 - Validation_Split))]

        Train_Index = Train_X1.shape[0]

        Val_X1 = Train_X[Train_Index:Train_Index+math.floor(Train_X.shape[0] * (Validation_Split))]
        Val_Y1 = Train_Y[Train_Index:Train_Index+math.floor(Train_Y.shape[0] * (Validation_Split))]

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

        Normalization_Layer = layers.Normalization(input_shape=[1, ], axis=None)
        Normalization_Layer.adapt(Train_X_Array)

        # Creating Regression Model
        Current_RegressionModel = tf.keras.Sequential([
            Normalization_Layer,
            layers.Dense(units=1)
        ])

        # Instantiate an Optimizer
        optimizer = keras.optimizers.SGD(learning_rate=Learning_Rate)

        # Instantiate a Loss Function.
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Printing out the Summary of Regression Model
        Current_RegressionModel.summary()

        # Prepare the Metrics.
        train_acc_metric = keras.metrics.MeanSquaredError()
        val_acc_metric = keras.metrics.MeanSquaredError()


        Training_Loss_Value_Set = []
        Val_Loss_Value_Set = []
        Training_Error_Value_Set = []
        Val_Error_Value_Set = []


        for epoch in range(Epochs):

            #print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = Current_RegressionModel(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, Current_RegressionModel.trainable_weights)
                optimizer.apply_gradients(zip(grads, Current_RegressionModel.trainable_weights))

                # Update training metric.
                train_acc_metric.update_state(y_batch_train, logits)

                # Log every 200 batches.
                #if step % 200 == 0:
                    #print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))
                    #print("Seen so far: %d samples" % ((step + 1) * Batch_Size))

            # training_loss_value[counter] = loss_value
            Training_Loss_Value_Set = tf.experimental.numpy.append(Training_Loss_Value_Set, loss_value)

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            #print("Training acc over epoch: %.4f" % (float(train_acc),))

            # training_error_value[counter] = train_acc
            Training_Error_Value_Set = tf.experimental.numpy.append(Training_Error_Value_Set, train_acc)

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in val_dataset:
                val_logits = Current_RegressionModel(x_batch_val, training=False)
                val_loss_value = loss_fn(y_batch_val, val_logits)
                # Update val metrics
                val_acc_metric.update_state(y_batch_val, val_logits)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            #print("Validation acc: %.4f" % (float(val_acc),))
            #print("Time taken: %.2fs" % (time.time() - start_time))

            Val_Loss_Value_Set = tf.experimental.numpy.append(Val_Loss_Value_Set, val_loss_value)
            Val_Error_Value_Set = tf.experimental.numpy.append(Val_Error_Value_Set, val_acc)

        Test_X_TF_Array = Test_X_TF.numpy()
        Test_X_TF_Array = np.reshape(Test_X_TF_Array, (Test_X_TF_Array.shape[0],1))
        Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0]))

        # LOOP: For Loop Simulation
        for jj in range(Test_X_TF_Array.shape[0]):
            Test_Y_Predict[jj] = Current_RegressionModel(Test_X_TF_Array[jj])


        # Predicting on Training and Testing Set Using Trained Model
        Test_Y_Predict1 = Current_RegressionModel.predict(Test_X)
        Train_Y_Predict1 = Current_RegressionModel.predict(Train_X)

        # Storing Output from Trained Regression Model as an Input for ANN
        ANN_HeatInput_Train_DF[RegressionModel_Key] = Train_Y_Predict1.tolist()
        ANN_HeatInput_Test_DF[RegressionModel_Key] = Test_Y_Predict1.tolist()

        # Computing Percentage Accuracy of the Model
        Train_PercentageAccuracy = (np.absolute((np.mean(Train_Y_Predict1) - np.mean(Train_Y.to_numpy()))) / np.mean(
            Train_Y.to_numpy())) * 100
        Test_PercentageAccuracy = (np.absolute((np.mean(Test_Y_Predict1) - np.mean(Test_Y.to_numpy()))) / np.mean(
            Test_Y.to_numpy())) * 100

        Train_PercentageAccuracy = Train_PercentageAccuracy.tolist()
        Test_PercentageAccuracy = Test_PercentageAccuracy.tolist()

        # Appending Percentage Accuracy into Table
        Regression_PercentageAccuracy_Current_DF = pd.DataFrame(
            [[RegressionModel_Key, Train_PercentageAccuracy, Test_PercentageAccuracy]],
            columns=['Regression Model Name', 'Training Accuracy', 'Testing Accuracy'])
        Regression_PercentageAccuracy_DF = pd.concat(
            [Regression_PercentageAccuracy_DF, Regression_PercentageAccuracy_Current_DF], ignore_index=True)

        RegressionModel_File_Name = RegressionModel_Key + str(Aggregation_UnitNumber_Total) + 'Zone_' + str(
            Aggregation_UnitNumber)

        # Saving Regression Model
        Current_RegressionModel.save(os.path.join(Sim_RegressionModelData_FolderPath, RegressionModel_File_Name))

        # Saving PHVAC Regression Model
        # if (RegressionModel_Key == 'QAC'):
            # Current_RegressionModel.save(
                # os.path.join(Sim_RegressionModelData_FolderPath, PHVAC_RegressionModel_File_Name))
            

        ## Plotting ##
        # Pair Plots for Training Data
        sns.pairplot(TrainingData_RegressionModel_Dict[RegressionModel_Key], diag_kind='kde')
        plt.gcf().set_size_inches(10, 10)
        plt.tight_layout()
        plt.savefig(os.path.join(Sim_RegressionModelData_FolderPath, RegressionModel_Key + '_PairPlot' + '_' +
                                 Training_RegressionModel_File_Name.split('_')[5] + '_' + str(
            Aggregation_UnitNumber) + '.png'), dpi=300)
        # plt.show()
        plt.close()

        # Training Plot
        plt.figure()
        plt.plot(Training_Loss_Value_Set, label='Loss')
        plt.plot(Val_Loss_Value_Set, label='Val_Loss')
        plt.title('Training Plot: ' + RegressionModel_Key + ' Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss ' + TrainingData_RegressionModel_Dict[RegressionModel_Key].columns[1], labelpad=15)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Sim_RegressionModelData_FolderPath, RegressionModel_Key + '_TrainingPlot' + '_' +
                                 Training_RegressionModel_File_Name.split('_')[5] + '_' + str(
            Aggregation_UnitNumber) + '.png'))
        # plt.show()
        plt.close()

        # Error Plot
        # Training Plot
        plt.figure()
        plt.plot(Training_Error_Value_Set, label='Loss')
        plt.plot(Val_Error_Value_Set, label='Val_Loss')
        plt.title('Error Plot: ' + RegressionModel_Key + ' Model')
        plt.xlabel('Epoch')
        plt.ylabel('Error ' + TrainingData_RegressionModel_Dict[RegressionModel_Key].columns[1], labelpad=15)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Sim_RegressionModelData_FolderPath, RegressionModel_Key + '_ErrorPlot' + '_' +
                                 Training_RegressionModel_File_Name.split('_')[5] + '_' + str(
            Aggregation_UnitNumber) + '.png'))
        # plt.show()
        plt.close()

        # Prediction Plot
        plt.figure()
        plt.scatter(Test_X, Test_Y, label='Data')
        plt.plot(Test_X, Test_Y_Predict, color='k', label='Predictions')
        plt.xlabel(TrainingData_RegressionModel_Dict[RegressionModel_Key].columns[0])
        plt.ylabel(TrainingData_RegressionModel_Dict[RegressionModel_Key].columns[1], labelpad=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(Sim_RegressionModelData_FolderPath, RegressionModel_Key + '_PredictionPlot' + '_' +
                                 Training_RegressionModel_File_Name.split('_')[5] + '_' + str(
            Aggregation_UnitNumber) + '.png'))
        # plt.show()
        plt.close()

    # =============================================================================
    # Storing Regression Model Data in Sim_RegressionModelData Folder
    # =============================================================================

    # Saving the Accuracy Table
    Regression_PercentageAccuracy_DF.to_csv(
        os.path.join(Sim_RegressionModelData_FolderPath, Regression_Model_Accuracy_File_Name), index=False)

    # Saving Sim_RegressionModelData as a .pickle File in Results Folder
    pickle.dump(ANN_HeatInput_Train_DF,
                open(os.path.join(Sim_RegressionModelData_FolderPath, ANN_HeatInput_Train_File_Name), "wb"))
    pickle.dump(ANN_HeatInput_Test_DF,
                open(os.path.join(Sim_RegressionModelData_FolderPath, ANN_HeatInput_Test_File_Name), "wb"))
