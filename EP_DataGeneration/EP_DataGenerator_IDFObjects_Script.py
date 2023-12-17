# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 21:11:17 2022

@author: ninad gaikwad
"""

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

# Custom Modules


# =============================================================================
# Process and Save IDF File Object Information in Results Folder
# =============================================================================

# Material Records
IDF_Material = epm_Edited_IDFFile.Material

IDF_Material_Records_Dict = IDF_Material._records

Material_List = []

MaterialRecord_Dict = {}

for key in IDF_Material_Records_Dict:

    MaterialRecord_Dict['name'] = IDF_Material_Records_Dict[key].name

    MaterialRecord_Dict['roughness'] = IDF_Material_Records_Dict[key].roughness

    MaterialRecord_Dict['thickness'] = IDF_Material_Records_Dict[key].thickness

    MaterialRecord_Dict['conductivity'] = IDF_Material_Records_Dict[key].conductivity

    MaterialRecord_Dict['density'] = IDF_Material_Records_Dict[key].density

    MaterialRecord_Dict['specific_heat'] = IDF_Material_Records_Dict[key].specific_heat

    MaterialRecord_Dict['thermal_absorptance'] = IDF_Material_Records_Dict[key].thermal_absorptance

    MaterialRecord_Dict['solar_absorptance'] = IDF_Material_Records_Dict[key].solar_absorptance

    MaterialRecord_Dict['visible_absorptance'] = IDF_Material_Records_Dict[key].visible_absorptance

    Material_List.append(MaterialRecord_Dict)

    MaterialRecord_Dict = {}


# Material_NoMass Records
IDF_MaterialNoMass = epm_Edited_IDFFile.Material_NoMass

IDF_MaterialNoMass_Records_Dict = IDF_MaterialNoMass._records

MaterialNoMass_List = []

MaterialNoMassRecord_Dict = {}

for key in IDF_MaterialNoMass_Records_Dict:

    MaterialNoMassRecord_Dict['name'] = IDF_MaterialNoMass_Records_Dict[key].name

    MaterialNoMassRecord_Dict['roughness'] = IDF_MaterialNoMass_Records_Dict[key].roughness

    MaterialNoMassRecord_Dict['thermal_resistance'] = IDF_MaterialNoMass_Records_Dict[key].thermal_resistance

    MaterialNoMassRecord_Dict['thermal_absorptance'] = IDF_MaterialNoMass_Records_Dict[key].thermal_absorptance

    MaterialNoMassRecord_Dict['solar_absorptance'] = IDF_MaterialNoMass_Records_Dict[key].solar_absorptance

    MaterialNoMassRecord_Dict['visible_absorptance'] = IDF_MaterialNoMass_Records_Dict[key].visible_absorptance

    MaterialNoMass_List.append(MaterialNoMassRecord_Dict)

    MaterialNoMassRecord_Dict = {}

# Construction Records
IDF_Construction = epm_Edited_IDFFile.Construction

IDF_Construction_Records_Dict = IDF_Construction._records

Construction_List = []

ConstructionRecord_Dict = {}

for key in IDF_Construction_Records_Dict:

    ConstructionRecord_Dict['name'] = IDF_Construction_Records_Dict[key].name

    ConstructionRecord_Dict['outside_layer'] = IDF_Construction_Records_Dict[key].outside_layer

    ConstructionRecord_Dict['layer_2'] = IDF_Construction_Records_Dict[key].layer_2

    ConstructionRecord_Dict['layer_3'] = IDF_Construction_Records_Dict[key].layer_3

    ConstructionRecord_Dict['layer_4'] = IDF_Construction_Records_Dict[key].layer_4

    ConstructionRecord_Dict['layer_5'] = IDF_Construction_Records_Dict[key].layer_5

    ConstructionRecord_Dict['layer_6'] = IDF_Construction_Records_Dict[key].layer_6

    ConstructionRecord_Dict['layer_7'] = IDF_Construction_Records_Dict[key].layer_7

    ConstructionRecord_Dict['layer_8'] = IDF_Construction_Records_Dict[key].layer_8

    ConstructionRecord_Dict['layer_9'] = IDF_Construction_Records_Dict[key].layer_9

    ConstructionRecord_Dict['layer_10'] = IDF_Construction_Records_Dict[key].layer_10

    Construction_List.append(ConstructionRecord_Dict)

    ConstructionRecord_Dict = {}

# Zone Records
IDF_Zone = epm_Edited_IDFFile.Zone

IDF_Zone_Records_Dict = IDF_Zone._records

Zone_List = []

ZoneRecord_Dict = {}

for key in IDF_Zone_Records_Dict:

    ZoneRecord_Dict['name'] = IDF_Zone_Records_Dict[key].name

    ZoneRecord_Dict['part_of_total_floor_area'] = IDF_Zone_Records_Dict[key].part_of_total_floor_area

    Zone_List.append(ZoneRecord_Dict)

    ZoneRecord_Dict = {}

# BuildingSurface_Detailed Records
IDF_BuildingSurface = epm_Edited_IDFFile.BuildingSurface_Detailed

IDF_BuildingSurface_Records_Dict = IDF_BuildingSurface._records

BuildingSurface_List = []

BuildingSurfaceRecord_Dict = {}

for key in IDF_BuildingSurface_Records_Dict:

    BuildingSurfaceRecord_Dict['name'] = IDF_BuildingSurface_Records_Dict[key].name

    BuildingSurfaceRecord_Dict['surface_type'] = IDF_BuildingSurface_Records_Dict[key].surface_type

    BuildingSurfaceRecord_Dict['zone_name'] = IDF_BuildingSurface_Records_Dict[key].zone_name

    BuildingSurfaceRecord_Dict['construction_name'] = IDF_BuildingSurface_Records_Dict[key].construction_name

    BuildingSurfaceRecord_Dict['number_of_vertices'] = IDF_BuildingSurface_Records_Dict[key].number_of_vertices

    Number_Vertices = BuildingSurfaceRecord_Dict['number_of_vertices']

    VertexVector_List = []

    ii = 0

    for i in range(int(Number_Vertices)):

        ii = i i +1

        VertexName_X = 'vertex_' + str(ii) + '_x_coordinate'

        VertexName_Y = 'vertex_' + str(ii) + '_y_coordinate'

        VertexName_Z = 'vertex_' + str(ii) + '_z_coordinate'

        VertexVector = []

        VertexVector.append(IDF_BuildingSurface_Records_Dict[key][VertexName_X])

        VertexVector.append(IDF_BuildingSurface_Records_Dict[key][VertexName_Y])

        VertexVector.append(IDF_BuildingSurface_Records_Dict[key][VertexName_Z])

        VertexVector_List.append(np.array(VertexVector))

    ii = 0

    Current_SurfaceArea_Sum = 0

    for i in range(int(Number_Vertices)):

        ii_1 = ii + 1

        if (ii_1 > int(Number_Vertices ) -1):

            ii_1 = 0

        Current_SurfaceArea_Sum = Current_SurfaceArea_Sum + \
                    ((1 / 2) * (np.cross(VertexVector_List[ii], VertexVector_List[ii_1])))

        ii = ii + 1

    BuildingSurfaceRecord_Dict['area'] = np.linalg.norm(Current_SurfaceArea_Sum)

    BuildingSurface_List.append(BuildingSurfaceRecord_Dict)

    BuildingSurfaceRecord_Dict = {}

# FenestrationSurface_Detailed Records
IDF_FenestrationSurface = epm_Edited_IDFFile.FenestrationSurface_Detailed

IDF_FenestrationSurface_Records_Dict = IDF_FenestrationSurface._records

FenestrationSurface_List = []

FenestrationSurfaceRecord_Dict = {}

for key in IDF_FenestrationSurface_Records_Dict:

    FenestrationSurfaceRecord_Dict['name'] = IDF_FenestrationSurface_Records_Dict[key].name

    FenestrationSurfaceRecord_Dict['surface_type'] = IDF_FenestrationSurface_Records_Dict[key].surface_type

    FenestrationSurfaceRecord_Dict['building_surface_name'] = IDF_FenestrationSurface_Records_Dict[
        key].building_surface_name

    FenestrationSurfaceRecord_Dict['construction_name'] = IDF_FenestrationSurface_Records_Dict[key].construction_name

    FenestrationSurfaceRecord_Dict['number_of_vertices'] = IDF_FenestrationSurface_Records_Dict[key].number_of_vertices

    Number_Vertices = FenestrationSurfaceRecord_Dict['number_of_vertices']

    VertexVector_List = []

    ii = 0

    for i in range(int(Number_Vertices)):
        ii = ii + 1

        VertexName_X = 'vertex_' + str(ii) + '_x_coordinate'

        VertexName_Y = 'vertex_' + str(ii) + '_y_coordinate'

        VertexName_Z = 'vertex_' + str(ii) + '_z_coordinate'

        VertexVector = []

        VertexVector.append(IDF_FenestrationSurface_Records_Dict[key][VertexName_X])

        VertexVector.append(IDF_FenestrationSurface_Records_Dict[key][VertexName_Y])

        VertexVector.append(IDF_FenestrationSurface_Records_Dict[key][VertexName_Z])

        VertexVector_List.append(np.array(VertexVector))

    ii = 0

    Current_SurfaceArea_Sum = 0

    for i in range(int(Number_Vertices)):

        ii_1 = ii + 1

        if (ii_1 > int(Number_Vertices) - 1):
            ii_1 = 0

        Current_SurfaceArea_Sum = Current_SurfaceArea_Sum + (
                    (1 / 2) * (np.cross(VertexVector_List[ii], VertexVector_List[ii_1])))

        ii = ii + 1

    FenestrationSurfaceRecord_Dict['area'] = np.linalg.norm(Current_SurfaceArea_Sum)

    FenestrationSurface_List.append(FenestrationSurfaceRecord_Dict)

    FenestrationSurfaceRecord_Dict = {}

# Sizing_System Records
IDF_SizingSystem = epm_Edited_IDFFile.Sizing_System

IDF_SizingSystem_Records_Dict = IDF_SizingSystem._records

SizingSystem_List = []

SizingSystemRecord_Dict = {}

for key in IDF_SizingSystem_Records_Dict:
    SizingSystemRecord_Dict['airloop_name'] = IDF_SizingSystem_Records_Dict[key].airloop_name

    SizingSystem_List.append(SizingSystemRecord_Dict)

    SizingSystemRecord_Dict = {}

IDFObjectRecords_Dict = {'Material_List': Material_List, 'MaterialNoMass_List': MaterialNoMass_List,
                         'Construction_List': Construction_List, 'Zone_List': Zone_List,
                         'BuildingSurface_List': BuildingSurface_List,
                         'FenestrationSurface_List': FenestrationSurface_List, 'SizingSystem_List': SizingSystem_List}

pickle.dump(IDFObjectRecords_Dict,
            open(os.path.join(Sim_IDFProcessedData_FolderPath, "IDF_ObjectRecords_DictListDict.pickle"), "wb"))

# scipy.io.savemat(os.path.join(Sim_IDFProcessedData_FolderPath,"IDF_ObjectRecords_DictListDict.mat"), IDFObjectRecords_Dict)