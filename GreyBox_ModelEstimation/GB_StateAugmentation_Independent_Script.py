# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import math
import time
import tracemalloc
import copy

# Getting path for custom modules 
module_path = os.path.abspath(os.path.join('.','bayesian-filters-smoothers'))

# Adding custom module path to Python path
if module_path not in sys.path:
    sys.path.append(module_path)
    
# Printing the custom module paths
print('Custom Module Path:' + '\n' + module_path)

## Importing custom modules
from bayesian_filters_smoothers import Extended_Kalman_Filter_Smoother as EKFS
from bayesian_filters_smoothers import Unscented_Kalman_Filter_Smoother as UKFS
from bayesian_filters_smoothers import Gaussian_Filter_Smoother as GFS

# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Shoter_ResultsPath = True  # True - Results are stored in shorter path , False - Results are stored in the appropriate directory

DateTime_Tuple = ('01/01/2013', '12/31/2013')

StateAug_Filter = 2  # 1 - EKF, 2 - UKF , 3 - GF-Type1 Gauss-Hermite , 4 - GF-Type2 Spherical Cubature

## User Input: For the Filter : For ALL
P_model = 1

Theta_Initial = [ 0.05, 0.05, 10000000, 10000000, 0.5, 0.5]  # [R_zw, R_wa, C_z, C_w, A_1, A_2]

# Filter process/measurement noise covariances : For ALL
Q_model = 1
Q_params = 1
R_model = 0.01

# Filter constant parameters : Both UKF and GF
n = 8  # Dimension of states of the system
m = 1  # Dimension of measurements of the system

p = 3  # Order of Hermite Polynomial :  For GF

alpha = 0.9 # Controls spread of Sigma Points about mean :  For UKF
k = 0.9  # Controls spread of Sigma Points about mean :  For UKF
beta = 5  # Helps to incorporate prior information abou thwe non-Gaussian :  For UKF

# Plot Size parameters
Plot_Width = 15
Plot_Height = 10

## User Input: Aggregation Unit Number ##
# Aggregation_UnitNumber = 1

Total_Aggregation_Zone_Number = 5

Aggregation_UnitNumber = 2

# Aggregation Zone NameStem Input
Aggregation_Zone_NameStem = 'Aggregation_Zone'

if (StateAug_Filter == 1):
    GBModel_Key = 'StateAug_EKFS'
elif (StateAug_Filter == 2):
    GBModel_Key = 'StateAug_UKFS'
elif (StateAug_Filter == 3):
    GBModel_Key = 'StateAug_GFS1'
elif (StateAug_Filter == 4):
    GBModel_Key = 'StateAug_GFS2'

#--------------------------------------------------------------Defining System Model---------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
# Defining the Simple Pendulum Discrete-Time Nonlinear System Function
def SingleZone_f1(x_k_1, u_k_1):
    """Provides discrete time dynamics for a nonlinear Single-Zone system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        u_k_1 (numpy.array): Previous system measurement
        
    Returns:
        x_k_true (numpy.array): Current true system state      
    """
    
    # Simulation parameters
    ts = 300    
       
    # Getting individual state components
    Tz_k_1 = x_k_1[0,0]
    Tw_k_1 = x_k_1[1,0]

    # Getting the parameter components : [R_zw, R_wa, C_z, C_w, A_1, A_2] 
    Rzw_k_1 = x_k_1[2,0]
    Rwa_k_1 = x_k_1[3,0]
    Cz_k_1 = x_k_1[4,0]
    Cw_k_1 = x_k_1[5,0]
    A1_k_1 = x_k_1[6,0]
    A2_k_1 = x_k_1[7,0]

    # Getting Control Components : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
    Ta = u_k_1[0,0]
    Q_sol1 = u_k_1[1,0]
    Q_sol2 = u_k_1[2,0]
    Q_Zic = u_k_1[3,0]
    Q_Zir = u_k_1[4,0]
    Q_ac = u_k_1[5,0]
    
    # Computing current true system state
    Tz_k = Tz_k_1 + ts*(((1/(Cz_k_1 * Rzw_k_1)) * (Tw_k_1 - Tz_k_1)) + 
                        ((1/Cz_k_1) * (Q_Zic + Q_ac)) +
                        ((A1_k_1/Cz_k_1) * (Q_sol1)))
    Tw_k = Tw_k_1 + ts*(((1/(Cw_k_1 * Rzw_k_1)) * (Tz_k_1 - Tw_k_1)) +
                        ((1/(Cw_k_1 * Rwa_k_1)) * (Ta - Tw_k_1)) +
                        ((1/Cw_k_1) * (Q_Zir)) +
                        ((A2_k_1/Cw_k_1) * (Q_sol2))) 

    # Computing current true system parameters
    Rzw_k = Rzw_k_1
    Rwa_k = Rwa_k_1
    Cz_k = Cz_k_1
    Cw_k = Cw_k_1
    A1_k = A1_k_1
    A2_k = A2_k_1
                                        
    x_k_true = np.reshape(np.array([Tz_k, Tw_k, Rzw_k, Rwa_k, Cz_k, Cw_k, A1_k, A2_k]),(8,1))
                                        
    # Return statement
    return x_k_true 

# Defining the Simple Pendulum Discrete-Time Nonlinear Measurement Function
def SingleZone_h1(x_k_1):
    """Provides discrete time Measurement for a nonlinear Single-Zone system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        
    Returns:
        y_k_true (numpy.array): Current true system measurement       
    """
    
    # Simulation parameters
    
    # Getting individual state components
    Tz_k_1 = x_k_1[0,0]
    
    # Computing current true system measurement
    y_k_true = Tz_k_1 
    y_k_true = np.reshape(y_k_true,(1,1))
                                        
    # Return statement
    return y_k_true 


#----------------------------------------------------------Defining System Model Jacobian----------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
# Defining the Single-Zone Discrete-Time Linear System Function
def SingleZone_F(x_k_1, u_k_1):
    """Provides discrete time nonlinear Single-Zone system Jacobian
    
    Args:
        x_k_1 (numpy.array): Previous system state
        u_k_1 (numpy.array): Previous system measurement
        
    Returns:
        F (numpy.array): System linearized Dynamics Jacobian      
    """
        
    # Getting individual state components
    Tz_k_1 = x_k_1[0,0]
    Tw_k_1 = x_k_1[1,0]

    # Getting the parameter components
    Rzw_k_1 = x_k_1[2,0]
    Rwa_k_1 = x_k_1[3,0]
    Cz_k_1 = x_k_1[4,0]
    Cw_k_1 = x_k_1[5,0]
    A1_k_1 = x_k_1[6,0]
    A2_k_1 = x_k_1[7,0]

    # Getting Control Components : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
    Ta = u_k_1[0,0]
    Q_sol1 = u_k_1[1,0]
    Q_sol2 = u_k_1[2,0]
    Q_Zic = u_k_1[3,0]
    Q_Zir = u_k_1[4,0]
    Q_ac = u_k_1[5,0]
    
    # Computing System State Jacobian
    F = np.reshape(np.array([[-(1/(Cz_k_1 * Rzw_k_1)), (1/(Cz_k_1 * Rzw_k_1)), (-(1/(Cz_k_1 * Rzw_k_1**2)) * (Tw_k_1 - Tz_k_1)), 0, (-(1/Cz_k_1**2) * (((Tw_k_1 - Tz_k_1)/(Rzw_k_1)) + (Q_Zic + Q_ac) + (A1_k_1 * Q_sol1))), 0, (Q_sol1/Cz_k_1), 0], 
                             [(1/(Cw_k_1 * Rzw_k_1)), -((1/(Cw_k_1 * Rzw_k_1)) +(1/(Cw_k_1 * Rwa_k_1))), (-(1/(Cw_k_1 * Rzw_k_1**2)) * (Tz_k_1 - Tw_k_1)), (-(1/(Cw_k_1 * Rwa_k_1**2)) * (Ta - Tw_k_1)), 0, (-(1/Cw_k_1**2) * (((Tz_k_1 - Tw_k_1)/(Rzw_k_1)) + ((Ta - Tw_k_1)/(Rwa_k_1)) + (Q_Zir) + (A2_k_1 * Q_sol2))), 0, (Q_sol2/Cw_k_1)], 
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 1]]),(8,8))
    
                                        
    # Return statement
    return F 

# Defining the Simple Pendulum Discrete-Time Linearized Measurement Function
def SingleZone_H(x_k_1):
    """Provides discrete time nonlinear Single-Zone system Measurement Jacobian
    
    Args:
        x_k_1 (numpy.array): Previous system state
        
    Returns:
        H (numpy.array): System linearized Measurement Jacobian      
    """
        
    # Getting individual state components
    theta_k_1 = x_k_1[0,0]
    omega_k_1 = x_k_1[1,0]
    
    # Computing System State Jacobian
    H = np.reshape(np.array([1, 0, 0, 0, 0, 0, 0, 0]),(1,8))
    
                                        
    # Return statement
    return H 

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

# kk = kk + 1

kk = Aggregation_UnitNumber

print("Current Unit Number: " + str(kk))

# Creating Required File Names

Aggregation_DF_Test_File_Name = 'Aggregation_DF_Test_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

Aggregation_DF_Train_File_Name = 'Aggregation_DF_Train_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

ANN_HeatInput_Test_DF_File_Name = 'ANN_HeatInput_Test_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

ANN_HeatInput_Train_DF_File_Name = 'ANN_HeatInput_Train_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

PHVAC_Regression_Model_File_Name = 'QAC_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk)

# Get Required Files from Sim_AggregatedTestTrainData_FolderPath    
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

    Sim_ANNModelData_FolderName = 'ANN_Model_Data'

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

# Joining Train/Test Aggregated_DF
Aggregated_DF = pd.concat([AggregatedTrain_DF.reset_index(drop=True), AggregatedTest_DF.reset_index(drop=True)])
Aggregated_DF.sort_values("DateTime", inplace=True)
Aggregated_DF.reset_index(drop=True, inplace=True)

# Joining Train/Test ANN_HeatInput_DF
ANN_HeatInput_Train_DF.insert(0, 'DateTime', DateTime_Train)
ANN_HeatInput_Test_DF.insert(0, 'DateTime', DateTime_Test)
ANN_HeatInput_DF = pd.concat([ANN_HeatInput_Train_DF.reset_index(drop=True), ANN_HeatInput_Test_DF.reset_index(drop=True)])
ANN_HeatInput_DF.sort_values("DateTime", inplace=True)
ANN_HeatInput_DF.reset_index(drop=True, inplace=True)

# Getting Data within Start and End Dates provided by user
# Add DateTime to Aggregation_DF
DateTime_List = Aggregated_DF['DateTime'].tolist() 

# Getting Start and End Dates for the Dataset
StartDate_Dataset = DateTime_List[0]
EndDate_Dataset = DateTime_List[-1]

# Getting the File Resolution from DateTime_List
DateTime_Delta = DateTime_List[1] - DateTime_List[0]

FileResolution_Minutes = DateTime_Delta.seconds/60

    # Getting Start and End Date
StartDate = datetime.datetime.strptime(DateTime_Tuple[0],'%m/%d/%Y')
EndDate = datetime.datetime.strptime(DateTime_Tuple[1],'%m/%d/%Y')

# User Dates Corrected
StartDate_Corrected = datetime.datetime(StartDate.year,StartDate.month,StartDate.day,0,int(FileResolution_Minutes),0)
EndDate_Corrected = datetime.datetime(EndDate.year,EndDate.month,EndDate.day,23,60-int(FileResolution_Minutes),0)

Counter_DateTime = -1

# Initializing DateRange List
DateRange_Index = []

# FOR LOOP:
for Element in DateTime_List:
    Counter_DateTime = Counter_DateTime + 1

    if (Element >= StartDate_Corrected and Element <= EndDate_Corrected):
        DateRange_Index.append(Counter_DateTime)

# Getting Train and Test Dataset
Aggregated_DF = copy.deepcopy(Aggregated_DF.iloc[DateRange_Index,:])
ANN_HeatInput_DF = copy.deepcopy(ANN_HeatInput_DF.iloc[DateRange_Index,:])

Aggregated_DF.reset_index(drop=True, inplace=True)
ANN_HeatInput_DF.reset_index(drop=True, inplace=True)


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
for ii in range(ANN_HeatInput_DF.shape[0]):
    # print(ii)
    QZic_Train_1 = ANN_HeatInput_DF['QZic_P'][ii][0] + ANN_HeatInput_DF['QZic_L'][ii][0] + \
                    ANN_HeatInput_DF['QZic_EE'][ii][0]
    QZir_Train_1 = ANN_HeatInput_DF['QZir_P'][ii][0] + ANN_HeatInput_DF['QZir_L'][ii][0] + \
                    ANN_HeatInput_DF['QZir_EE'][ii][0] + ANN_HeatInput_DF['QZivr_L'][ii][0]
    QZic_Train.append(QZic_Train_1)
    QZir_Train.append(QZir_Train_1)

    QSol1_Train_1 = ANN_HeatInput_DF['QSol1'][ii][0]
    QSol2_Train_1 = ANN_HeatInput_DF['QSol2'][ii][0]
    # QAC_Train_1 = ANN_HeatInput_DF['QAC'][ii][0]
    QAC_Train_1 = Aggregated_DF['QHVAC_X'].iloc[ii]

    QSol1_Train.append(QSol1_Train_1)
    QSol2_Train.append(QSol2_Train_1)
    QAC_Train.append(QAC_Train_1)

ANN_HeatInput_DF.insert(2, 'QZic', QZic_Train)
ANN_HeatInput_DF.insert(2, 'QZir', QZir_Train)
ANN_HeatInput_DF.insert(2, 'QSol1_Corrected', QSol1_Train)
ANN_HeatInput_DF.insert(2, 'QSol2_Corrected', QSol2_Train)
ANN_HeatInput_DF.insert(2, 'QAC_Corrected', QAC_Train)

ANN_HeatInput_DF.reset_index(drop=True, inplace=True)
Aggregated_DF.reset_index(drop=True, inplace=True)

# Creating Common DF
Common_Data_DF = pd.concat([Aggregated_DF[[ 'DateTime', 'Zone_Air_Temperature_', 'Site_Outdoor_Air_Drybulb_Temperature_']], ANN_HeatInput_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']]], axis=1)

# Debugger
plt.figure()
plt.plot(Common_Data_DF['QAC_Corrected'])
plt.show()

# Creating Y_Measurement and U_Measurement
Y_Measurement = np.zeros((1,1))
U_Measurement = np.zeros((6,1))

for ii in range(Aggregated_DF.shape[0]):

    # Getting State Measurement
    T_z = Common_Data_DF['Zone_Air_Temperature_'].iloc[ii]

    # Getting Disturbances 
    T_a = Common_Data_DF['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[ii]
    Q_sol1 = Common_Data_DF['QSol1_Corrected'].iloc[ii]
    Q_sol2 = Common_Data_DF['QSol2_Corrected'].iloc[ii]
    Q_Zic = Common_Data_DF['QZic'].iloc[ii]
    Q_Zir = Common_Data_DF['QZir'].iloc[ii]

    # Getting Control
    Q_ac = Common_Data_DF['QAC_Corrected'].iloc[ii]

    # Updating Y_Measurement
    Y_Measurement = np.hstack((Y_Measurement, np.reshape(np.array([T_z]), (1,1))))

    # Updating U_Measurement
    U_Measurement = np.hstack((U_Measurement, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))

# Removing first elements from Y_Measurement and U_Measurement
Y_Measurement = Y_Measurement[:,1:]
U_Measurement = U_Measurement[:,1:]


# =============================================================================
# Filter Creation
# =============================================================================

# Creating initial Filter mean vector
m_ini_model = np.reshape(np.array([Y_Measurement[:,0][0], U_Measurement[1,0], Theta_Initial[0], Theta_Initial[1], Theta_Initial[2], Theta_Initial[3], Theta_Initial[4], Theta_Initial[5],]), (n,1))

# Creating initial Filter state covariance matrix
P_ini_model = P_model*np.eye(n)

# Create the model Q and R matrices
Q_d = np.eye(n) * np.reshape(np.array([Q_model, Q_model, Q_params,Q_params,Q_params,Q_params,Q_params,Q_params]), (n,1))
R_d = np.reshape(np.array([R_model]), (1,1))

# If Elseif Loop: For type of Filter chosen
if (StateAug_Filter == 1):

    SingleZone_Filter = EKFS(SingleZone_f1, SingleZone_F, SingleZone_h1, SingleZone_H, m_ini_model, P_ini_model, Q_d, R_d)

elif (StateAug_Filter == 2):

    SingleZone_Filter = UKFS(SingleZone_f1, SingleZone_h1, n, m, alpha, k, beta,  m_ini_model, P_ini_model, Q_d, R_d)

elif (StateAug_Filter == 3):

    SingleZone_Filter = GFS(1, SingleZone_f1, SingleZone_h1, n, m,  m_ini_model, P_ini_model, Q_d, R_d, p)

elif (StateAug_Filter == 4):

    SingleZone_Filter = GFS(2, SingleZone_f1, SingleZone_h1, n, m,  m_ini_model, P_ini_model, Q_d, R_d, p)


# =============================================================================
# Filter Time Evolution
# =============================================================================

# Initializing model filter state array to store time evolution
x_model_nonlinear_filter = m_ini_model

# If Elseif Loop: For type of Filter chosen
if (StateAug_Filter == 1):

    # FOR LOOP: For each discrete time-step
    for ii in range(Y_Measurement.shape[1]-1):
        
        ## For measurements coming from Nonlinear System
        
        # Extended Kalman Filter: Predict Step    
        m_k_, P_k_ = SingleZone_Filter.Extended_Kalman_Predict(np.reshape(U_Measurement[:,ii],(6,1)))
        
        # Extended Kalman Filter: Update Step
        v_k, S_k, K_k = SingleZone_Filter.Extended_Kalman_Update(np.reshape(Y_Measurement[:,ii+1][0], (1,1)), m_k_, P_k_)    
        
        # Storing the Filtered states
        x_k_filter = SingleZone_Filter.m_k
        x_model_nonlinear_filter = np.hstack((x_model_nonlinear_filter, x_k_filter))

elif (StateAug_Filter == 2):

    # FOR LOOP: For each discrete time-step
    for ii in range(Y_Measurement.shape[1]-1):
    
        ## For measurements coming from Nonlinear System
        
        # Extended Kalman Filter: Predict Step    
        m_k_, P_k_, D_k = SingleZone_Filter.Unscented_Kalman_Predict(np.reshape(U_Measurement[:,ii],(6,1)))
        
        # Extended Kalman Filter: Update Step
        mu_k, S_k, C_k, K_k = SingleZone_Filter.Unscented_Kalman_Update(np.reshape(Y_Measurement[:,ii+1][0], (1,1)), m_k_, P_k_)    
        
        # Storing the Filtered states
        x_k_filter = SingleZone_Filter.m_k
        x_model_nonlinear_filter = np.hstack((x_model_nonlinear_filter, x_k_filter))

elif ((StateAug_Filter == 3) or (StateAug_Filter == 4)):

    # FOR LOOP: For each discrete time-step
    for ii in range(Y_Measurement.shape[1]-1):
    
        ## For measurements coming from Nonlinear System
        
        # Extended Kalman Filter: Predict Step    
        m_k_, P_k_, D_k = SingleZone_Filter.Gaussian_Predict(np.reshape(U_Measurement[:,ii],(6,1)))
        
        # Extended Kalman Filter: Update Step
        mu_k, S_k, C_k, K_k = SingleZone_Filter.Gaussian_Update(np.reshape(Y_Measurement[:,ii+1][0], (1,1)), m_k_, P_k_)    
        
        # Storing the Filtered states
        x_k_filter = SingleZone_Filter.m_k
        x_model_nonlinear_filter = np.hstack((x_model_nonlinear_filter, x_k_filter))

# =============================================================================
# Plotting : Filter
# =============================================================================
# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting True vs. Filtered States of Nonlinear System
plt.subplot(411)
plt.plot(Y_Measurement[:,:].transpose(), linestyle='-', linewidth=3, label=r'$T_{z_{meas}}$ $(^{ \circ }C)$')
plt.plot(U_Measurement[0,:].transpose(), linestyle='-', linewidth=3, label=r'$T_{a_{meas}}$ $(^{ \circ }C)$')
plt.plot(x_model_nonlinear_filter[0:2,:].transpose(), linestyle='--', linewidth=1, label=[r'$T_{z_{filter}}$ $(^{ \circ }C)$', r'$T_{w_{filter}}$ $(^{ \circ }C)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Single Zone - Measured vs. Filtered States' + GBModel_Key, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System  $(^{ \circ }C / W)$  $(J / ^{ \circ }C)$
plt.subplot(412)
plt.plot(x_model_nonlinear_filter[2:4,:].transpose(), linestyle='--', linewidth=1, label=[r'$R_{zw_{filter}}$ $(^{ \circ }C / W)$', r'$R_{wa_{filter}}$ $(^{ \circ }C / W)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(R_{zw},R_{wa})$', fontsize=12)
plt.title('Single Zone - Filtered Parameters ' + r'$R_{zw},R_{wa}$ ' + GBModel_Key, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.subplot(413)
plt.plot(x_model_nonlinear_filter[4:6,:].transpose(), linestyle='--', linewidth=1, label=[r'$C_{z_{filter}}$ $(J / ^{ \circ }C)$', r'$C_{w_{filter}}$ $(J / ^{ \circ }C)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(C_{z}, C_{w})$', fontsize=12)
plt.title('Single Zone - Filtered Parameters ' + r'$C_{z}, C_{w}$ ' + GBModel_Key, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.subplot(414)
plt.plot(x_model_nonlinear_filter[6:8,:].transpose(), linestyle='--', linewidth=1, label=[r'$A_{sol1_{filter}}$ ', r'$A_{sol2_{filter}}$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(A_{sol1}, A_{sol2})$', fontsize=12)
plt.title('Single Zone - Filtered Parameters  ' + r'$A_{sol1}, A_{sol2}$ ' + GBModel_Key, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


#plt.show()

# =============================================================================
# Smoother Time Evolution
# =============================================================================

## Create the input vector list
U_Measurement_list = [] 

for ii in range(U_Measurement.shape[1]):
    
    U_Measurement_list.append(np.reshape(U_Measurement[:,ii], (6,1)))
    
## Create the measurement vector list
Y_Measurement_list = []

for ii in range(Y_Measurement.shape[1]):
    
    Y_Measurement_list.append(np.reshape(Y_Measurement[:,ii], (1,1)))



# If Elseif Loop: For type of Filter chosen
if (StateAug_Filter == 1):

    # Running Smoother
    G_k_list, m_k_s_list, P_k_s_list = SingleZone_Filter.Extended_Kalman_Smoother(U_Measurement_list, Y_Measurement_list[1:])     
        

elif (StateAug_Filter == 2):

    # Running Smoother
    G_k_list, m_k_s_list, P_k_s_list = SingleZone_Filter.Unscented_Kalman_Smoother(U_Measurement_list, Y_Measurement_list[1:]) 
        

elif ((StateAug_Filter == 3) or (StateAug_Filter == 4)):

    # Running Smoother
    G_k_list, m_k_s_list, P_k_s_list = SingleZone_Filter.Gaussian_Smoother(U_Measurement_list, Y_Measurement_list[1:]) 

# Storing the Filtered states
for ii in range(len(m_k_s_list)):
    
    if (ii == 0):
        
        x_model_nonlinear_smoother = m_k_s_list[ii]
        
    else:
        
        x_model_nonlinear_smoother = np.hstack((x_model_nonlinear_smoother, m_k_s_list[ii]))

# =============================================================================
# Plotting : Filter
# =============================================================================
# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting True vs. Smoothed States of Nonlinear System
plt.subplot(411)
plt.plot(Y_Measurement[:,:].transpose(), linestyle='-', linewidth=3, label=r'$T_{z_{meas}}$ $(^{ \circ }C)$')
plt.plot(U_Measurement[0,:].transpose(), linestyle='-', linewidth=3, label=r'$T_{a_{meas}}$ $(^{ \circ }C)$')
plt.plot(x_model_nonlinear_smoother[0:2,:].transpose(), linestyle='--', linewidth=1, label=[r'$T_{z_{smooth}}$ $(^{ \circ }C)$', r'$T_{w_{smooth}}$ $(^{ \circ }C)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Single Zone - Measured vs. Smoothed States' + GBModel_Key, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Smoothed Parameters of Nonlinear System  $(^{ \circ }C / W)$  $(J / ^{ \circ }C)$
plt.subplot(412)
plt.plot(x_model_nonlinear_smoother[2:4,:].transpose(), linestyle='--', linewidth=1, label=[r'$R_{zw_{smooth}}$ $(^{ \circ }C / W)$', r'$R_{wa_{smooth}}$ $(^{ \circ }C / W)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(R_{zw},R_{wa})$', fontsize=12)
plt.title('Single Zone - Smoothed Parameters ' + r'$R_{zw},R_{wa}$ ' + GBModel_Key, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.subplot(413)
plt.plot(x_model_nonlinear_smoother[4:6,:].transpose(), linestyle='--', linewidth=1, label=[r'$C_{z_{smooth}}$ $(J / ^{ \circ }C)$', r'$C_{w_{smooth}}$ $(J / ^{ \circ }C)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(C_{z}, C_{w})$', fontsize=12)
plt.title('Single Zone - Smoothed Parameters ' + r'$C_{z}, C_{w}$ ' + GBModel_Key, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.subplot(414)
plt.plot(x_model_nonlinear_smoother[6:8,:].transpose(), linestyle='--', linewidth=1, label=[r'$A_{sol1_{smooth}}$ ', r'$A_{sol2_{smooth}}$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(A_{sol1}, A_{sol2})$', fontsize=12)
plt.title('Single Zone - Smoothed Parameters  ' + r'$A_{sol1}, A_{sol2}$ ' + GBModel_Key, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


plt.show()