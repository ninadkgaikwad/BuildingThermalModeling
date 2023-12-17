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

# =============================================================================
# User Inputs
# =============================================================================

# Simulation Folder and Files
Simulation_Name = "test1"

Training_RegressionModel_File_Name1 = 'TrainingData_RegressionModel_Dict_Aggregation_Dict_1Zone_1.pickle'

Testing_RegressionModel_File_Name1 = 'TestingData_RegressionModel_Dict_Aggregation_Dict_1Zone_1.pickle'

## User Input: Aggregation Unit Number ##
Aggregation_UnitNumber_Total = 1

# Learning Parameters for Linear Regression
Learning_Rate = 0.1

Loss_Function = 'mean_squared_error'

Epochs = 10

Validation_Split = 0.2

# Result File Names
Regression_Model_Accuracy_File_Name1 = 'Regression_Model_Accuracy'

ANN_HeatInput_Train_File_Name1 = 'ANN_HeatInput_Train_DF'

ANN_HeatInput_Test_File_Name1 = 'ANN_HeatInput_Test_DF'


for ii in range(Aggregation_UnitNumber_Total):

    Aggregation_UnitNumber = ii + 1

    # =============================================================================
    # Creating Input File Names
    # =============================================================================
    Training_RegressionModel_File_Name = "_".join(Training_RegressionModel_File_Name1.split('.')[0].split('_')[:-1]) + '_' + str(Aggregation_UnitNumber) + '.pickle'

    Testing_RegressionModel_File_Name = "_".join(Testing_RegressionModel_File_Name1.split('.')[0].split('_')[:-1]) + '_' + str(Aggregation_UnitNumber) + '.pickle'

    # =============================================================================
    # Creating Result File Names
    # =============================================================================
    Regression_Model_Accuracy_File_Name = Regression_Model_Accuracy_File_Name1 + '_' + Training_RegressionModel_File_Name.split('_')[5] + '_' +  str(Aggregation_UnitNumber) + '.csv'

    ANN_HeatInput_Train_File_Name = ANN_HeatInput_Train_File_Name1 + '_' + Training_RegressionModel_File_Name.split('_')[5] + '_' + str(Aggregation_UnitNumber) + '.pickle'

    ANN_HeatInput_Test_File_Name = ANN_HeatInput_Test_File_Name1 + '_' + Testing_RegressionModel_File_Name.split('_')[5] + '_' + str(Aggregation_UnitNumber) + '.pickle'

    PHVAC_RegressionModel_File_Name = 'PHVAC_RegressionModel_' + str(Aggregation_UnitNumber_Total) + 'Zone_' + str(Aggregation_UnitNumber)

    # =============================================================================
    # Getting Required Data from Sim_ProcessedData
    # =============================================================================

    # Getting Current File Directory Path
    Current_FilePath = os.path.dirname(__file__)

    # Getting Sim_TrainingTestingData Folder Path
    Sim_ProcessedData_FolderPath = os.path.join(Current_FilePath,  '..',  '..', 'Results', 'Processed_BuildingSim_Data', Simulation_Name, 'Sim_TrainingTestingData')

    # Get Required Files from Sim_TrainingTestingData_FolderPath
    TestingData_RegressionModel_Dict_File = open(os.path.join(Sim_ProcessedData_FolderPath,Testing_RegressionModel_File_Name),"rb")

    TestingData_RegressionModel_Dict = pickle.load(TestingData_RegressionModel_Dict_File)

    TrainingData_RegressionModel_Dict_File = open(os.path.join(Sim_ProcessedData_FolderPath,Training_RegressionModel_File_Name),"rb")

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
    os.path.isdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_RegressionModelData_FolderName))):

        # Folders Exist
        z = None

    else:

        os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_RegressionModelData_FolderName))

    # Creating Sim_RegressionModelData Folder Path
    Sim_RegressionModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name,
                                                 Sim_RegressionModelData_FolderName)


    # =============================================================================
    # Regression Modelling
    # =============================================================================

    # Initializating Dataframe to Store Training and Testing Data for ANN
    ANN_HeatInput_Train_DF = copy.deepcopy(pd.DataFrame())
    ANN_HeatInput_Test_DF = copy.deepcopy(pd.DataFrame())

    # Adding DateTime to the Dataframes
    ANN_HeatInput_Train_DF['DateTime'] = TrainingData_RegressionModel_Dict['DateTime'] 
    ANN_HeatInput_Test_DF['DateTime'] = TestingData_RegressionModel_Dict['DateTime'] 

    # Initializing Dataframe for Percentage Accuracy
    Regression_PercentageAccuracy_DF = copy.deepcopy(pd.DataFrame(columns=['Regression Model Name', 'Training Accuracy', 'Testing Accuracy']))

    # FOR LOOP: Creating Regression Models
    for RegressionModel_Key in TrainingData_RegressionModel_Dict:

        # Getting Training and Testing X and Y for Current Model

        #IF: For DateTime
        if (RegressionModel_Key == 'DateTime'):

            continue

        #IF: For QAC
        if (RegressionModel_Key == 'QAC'):
            Train_X = TrainingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 0].abs()
            Test_X = TestingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 0].abs()
        else:
            Train_X = TrainingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 0]
            Test_X = TestingData_RegressionModel_Dict[RegressionModel_Key].iloc[:, 0]

        Train_Y = TrainingData_RegressionModel_Dict[RegressionModel_Key].iloc[:,1]
        Test_Y = TestingData_RegressionModel_Dict[RegressionModel_Key].iloc[:,1]

        # Creating Normalization Layer
        Train_X_Array = np.array(Train_X)

        Normalization_Layer = layers.Normalization(input_shape=[1, ], axis=None)
        Normalization_Layer.adapt(Train_X_Array)

        # Creating Regression Model
        Current_RegressionModel = tf.keras.Sequential([
            Normalization_Layer,
            layers.Dense(units=1)
        ])

        # Printing out the Summary of Regression Model
        Current_RegressionModel.summary()

        # Setting up the Optimizer for Regression Model
        Current_RegressionModel.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate),
            loss=Loss_Function)

        # Training Current Regression Model
        history = Current_RegressionModel.fit(
            Train_X,
            Train_Y,
            epochs=Epochs,
            # Suppress Logging
            verbose=0,
            # Calculating Validation Results of the Training Dataset
            validation_split=Validation_Split)

        # Saving Current Trained Regression Model
        # Current_RegressionModel.save(os.path.join(Sim_RegressionModelData_FolderPath, RegressionModel_Key + '_TrainedModel'))

        '''
        test_results = {}
    
        test_results['horsepower_model'] = horsepower_model.evaluate(
            test_features['Horsepower'],
            test_labels, verbose=0)
    
        '''

        # Predicting on Training and Testing Set Using Trained Model
        Test_Y_Predict = Current_RegressionModel.predict(Test_X)
        Train_Y_Predict = Current_RegressionModel.predict(Train_X)

        # Storing Output from Trained Regression Model as an Input for ANN
        ANN_HeatInput_Train_DF[RegressionModel_Key] = Train_Y_Predict.tolist()
        ANN_HeatInput_Test_DF[RegressionModel_Key] = Test_Y_Predict.tolist()

        # Computing Percentage Accuracy of the Model
        Train_PercentageAccuracy = (np.absolute((np.mean(Train_Y_Predict) - np.mean(Train_Y.to_numpy()))) / np.mean(Train_Y.to_numpy())) * 100
        Test_PercentageAccuracy = (np.absolute((np.mean(Test_Y_Predict) - np.mean(Test_Y.to_numpy()))) / np.mean(Test_Y.to_numpy())) * 100

        Train_PercentageAccuracy = Train_PercentageAccuracy.tolist()
        Test_PercentageAccuracy = Test_PercentageAccuracy.tolist()

        # Appending Percentage Accuracy into Table
        Regression_PercentageAccuracy_Current_DF = pd.DataFrame([[RegressionModel_Key, Train_PercentageAccuracy, Test_PercentageAccuracy]], columns=['Regression Model Name','Training Accuracy','Testing Accuracy'])
        Regression_PercentageAccuracy_DF = pd.concat([Regression_PercentageAccuracy_DF,Regression_PercentageAccuracy_Current_DF], ignore_index=True)

        # Saving PHVAC Regression Model
        if (RegressionModel_Key == 'QAC'):
            Current_RegressionModel.save(os.path.join(Sim_RegressionModelData_FolderPath,PHVAC_RegressionModel_File_Name))

        ## Plotting ##
        # Pair Plots for Training Data
        sns.pairplot(TrainingData_RegressionModel_Dict[RegressionModel_Key], diag_kind='kde')
        plt.gcf().set_size_inches(10, 10)
        plt.tight_layout()
        plt.savefig(os.path.join(Sim_RegressionModelData_FolderPath, RegressionModel_Key + '_PairPlot' + '_' + Training_RegressionModel_File_Name.split('_')[5] + '_' + str(Aggregation_UnitNumber) + '.png'), dpi=300)
        # plt.show()
        plt.close()

        # Training Plot
        plt.figure()
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Val_Loss')
        plt.title('Training Plot: ' + RegressionModel_Key + ' Model')
        plt.xlabel('Epoch')
        plt.ylabel('Error ' + TrainingData_RegressionModel_Dict[RegressionModel_Key].columns[1], labelpad=15)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Sim_RegressionModelData_FolderPath, RegressionModel_Key + '_TrainingPlot' + '_' + Training_RegressionModel_File_Name.split('_')[5] + '_' + str(Aggregation_UnitNumber) + '.png'))
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
        plt.savefig(os.path.join(Sim_RegressionModelData_FolderPath, RegressionModel_Key + '_PredictionPlot' + '_' + Training_RegressionModel_File_Name.split('_')[5] + '_' + str(Aggregation_UnitNumber) + '.png'))
        # plt.show()
        plt.close()




    # =============================================================================
    # Storing Regression Model Data in Sim_RegressionModelData Folder
    # =============================================================================

    # Saving the Accuracy Table
    Regression_PercentageAccuracy_DF.to_csv(os.path.join(Sim_RegressionModelData_FolderPath, Regression_Model_Accuracy_File_Name), index= False)

    # Saving Sim_RegressionModelData as a .pickle File in Results Folder
    pickle.dump(ANN_HeatInput_Train_DF, open(os.path.join(Sim_RegressionModelData_FolderPath, ANN_HeatInput_Train_File_Name), "wb"))
    pickle.dump(ANN_HeatInput_Test_DF, open(os.path.join(Sim_RegressionModelData_FolderPath, ANN_HeatInput_Test_File_Name), "wb"))
