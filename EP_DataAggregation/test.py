# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import numpy as np
import pandas as pd
import scipy.io
import opyplus as op
import re
import shutil
import datetime
import pickle


# Process .eio Output File and save in Results Folder
# =============================================================================

# Reading .eio Output File
Eio_OutputFile_Path = 'eplusout.eio'

# Initializing Eio_OutputFile_Dict
Eio_OutputFile_Dict = {}

with open(Eio_OutputFile_Path) as f:
    Eio_OutputFile_Lines = f.readlines()

# Removing Intro Lines
Eio_OutputFile_Lines = Eio_OutputFile_Lines[1:]

# FOR LOOP: For each category in .eio File
for Line_1 in Eio_OutputFile_Lines:

    # IF ELSE LOOP: To check category
    if (Line_1.find('!') >= 0):

        print(Line_1 + '\n')

        # Get the Key for the .eio File category
        Pattern_1 = "<(.*?)>"

        Category_Key = re.search(Pattern_1, Line_1).group(1)

        # Get the Column Names for the .eio File category
        DF_ColumnName_List = Line_1.split(',')[1:]

        # Removing the '\n From the Last Name
        DF_ColumnName_List[-1] = DF_ColumnName_List[-1].split('\n')[0]

        # Removing Empty Element
        if DF_ColumnName_List[-1] == ' ':
            DF_ColumnName_List = DF_ColumnName_List[:-1]

        # Initializing DF_Index_List
        DF_Index_List = []

        # Initializing DF_Data_List
        DF_Data_List = []

        # FOR LOOP: For all elements of current .eio File category
        for Line_2 in Eio_OutputFile_Lines:

            # IF ELSE LOOP: To check data row belongs to current Category
            if ((Line_2.find('!') == -1) and (Line_2.find(Category_Key) >= 0)):

                print(Line_2 + '\n')

                DF_ColumnName_List_Length = len(DF_ColumnName_List)

                # Split Line_2
                Line_2_Split = Line_2.split(',')

                # Removing the '\n From the Last Data
                Line_2_Split[-1] = Line_2_Split[-1].split('\n')[0]

                # Removing Empty Element
                if Line_2_Split[-1] == ' ':
                    Line_2_Split = Line_2_Split[:-1]

                # Getting DF_Index_List element
                DF_Index_List.append(Line_2_Split[0])

                Length_Line2 = len(Line_2_Split[1:])

                Line_2_Split_1 = Line_2_Split[1:]

                # Filling up Empty Column
                if Length_Line2 < DF_ColumnName_List_Length:
                    Len_Difference = DF_ColumnName_List_Length - Length_Line2

                    for ii in range(Len_Difference):
                        Line_2_Split_1.append('NA')

                    # Getting DF_Data_List element
                    DF_Data_List.append(Line_2_Split_1)

                else:
                    # Getting DF_Data_List element
                    DF_Data_List.append(Line_2_Split[1:])

            else:

                continue

        # Creating DF_Table
        DF_Table = pd.DataFrame(DF_Data_List, index=DF_Index_List, columns=DF_ColumnName_List)

        # Adding DF_Table to the Eio_OutputFile_Dict
        Eio_OutputFile_Dict[Category_Key] = DF_Table

    else:

        continue

# Saving Eio_OutputFile_Dict as a .pickle File in Results Folder
pickle.dump(Eio_OutputFile_Dict, open("Eio_OutputFile.pickle", "wb"))