# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:47:14 2022

@author: ninad
"""

# Importing Desired Modules
import numpy as np
import pandas as pd

# My App Module

def CreateTimeVector(TimeDuration, TimeStep):
    
    TimeVector = np.arange(0, TimeDuration, TimeStep)
    
    TimeVector = np.reshape(TimeVector,(TimeVector.shape[0],1))
    
    return TimeVector

def CreateSine(TimeVector, A, F, P):
    
    Sine = A*np.sin(2*np.pi*F*TimeVector+np.radians(P))
    
    return Sine
    
def Compute_with_Sines(TimeVector, Sine1, Sine2, Computation_Option):
    
    # omputing new Sine Wave
    
    if (Computation_Option == 1): # Addition
    
        Sine_New = Sine1 + Sine2
    
    elif (Computation_Option == 2): # Subtraction
    
        Sine_New = Sine1 - Sine2
    
    elif (Computation_Option == 3): # Multiplication
    
        Sine_New = Sine1 * Sine2
        
    # Creating a Combined Table for Graphing purposes
    Combined_Array = np.hstack((TimeVector, Sine1, Sine2, Sine_New))
    
    Sines_DF = pd.DataFrame(Combined_Array, columns = ['Time','Sine_1','Sine_2','Sine_New'])
    
    return Sines_DF