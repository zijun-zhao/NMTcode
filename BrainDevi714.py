#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 20:58:15 2019

@author: penglab
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import neuro_morpho_toolbox as nmt

# %%Initialization

ccftable = pd.read_excel('/home/penglab/Documents/data/CCFv3 Summary Structures.xlsx', usecols=[1, 2, 3, 5, 6, 7], index_col=0,names=[ 'fullname', 'Abbrevation', 'depth in tree', 'structure_id_path','total_voxel_counts (10 um)'])
anofile = nmt.image('/home/penglab/Documents/data/annotation_10.nrrd')
bs = nmt.brain_structure('/home/penglab/Documents/data/Mouse.csv')
axonJaneliaccf = pd.read_excel('/home/penglab/Documents/dataSource/JaneliaAxonccf.xlsx', index_col=0)
SOMAJaneliaccf = pd.read_excel('/home/penglab/Documents/dataSource/JaneliaSomaccf.xlsx', index_col=0)
mouseDF = pd.read_excel('/home/penglab/Documents/dataSource/mouseDF.xlsx', index_col=0)
ccfDF = pd.read_excel('/home/penglab/Documents/dataSource/ccfDF.xlsx', index_col=0)
ccfVolume= pd.read_excel('/home/penglab/Documents/dataSource/ccfVolume.xlsx', index_col=0)
#%%  Read the nrrd file
arr3D = anofile.array
IDrange, IDcounts = np.unique(arr3D, return_counts=True)
#%%
VoxelResult = ccfVolume.copy()
VoxelResult['Child ID']=np.nan

for iterIdex in VoxelResult.index:
    child_temp=bs.get_all_child_id(iterIdex)
    child_T_store=[]
    for iiterIdx in child_temp:
        if iiterIdx in IDrange:
            child_T_store.append(iiterIdx)
    #store all the ID as a whole string
    list2=[str(i) for i in child_T_store]  
    list3=' '.join(list2) 
    VoxelResult.loc[iterIdex,'Child ID']=list3
# Add another column of information indicating the child number    
Childnum_store=[]
for IDchild in VoxelResult.index:
    lenchild=len(VoxelResult.loc[IDchild,'Child ID'].split())
    VoxelResult.loc[IDchild,'Child num']=lenchild  


#%% Look into the ~50micron problem 
#Here all the deviation is set to 50 micron


DeviDF=VoxelResult.copy()
    
dimx = arr3D.shape[0]
dimy = arr3D.shape[1]
dimz = arr3D.shape[2]

mu = 5
sigma = mu /10
sampleNo = 1
ranX = np.random.normal(mu, sigma, sampleNo)
# make it to be integers
#ranX = round(ranX[0])
ranX=mu
ranY = np.random.normal(mu, sigma, sampleNo)
ranY = round(ranY[0])
ranY=mu
ranZ = np.random.normal(mu, sigma, sampleNo)
ranZ=mu

CropArr = np.zeros(arr3D.shape)
CropArr[0:int(dimx-ranX), 0:int(dimy-ranY), 0:int(dimz-ranZ)]= arr3D[int(ranX):int(dimx), int(ranY):int(dimy), int(ranZ): int(dimz)]

copyarr = anofile.array

# First record the regions only have one subregions(may be only itself)
childonly1 = []
child1DF = DeviDF.copy()
child1DF = child1DF[child1DF['Child num'] == 1]
ii = 0
for iterID in child1DF.loc[:]['Child ID']:
    itemindex = np.where(copyarr == int(iterID))
    temA = []
    temA = CropArr[itemindex]
    temp = np.sum(temA == int(iterID))
    childonly1.append(temp)
    ii = ii + 1
    print('For 1-child brain regions, have finished ', ii / len(child1DF.loc[:]['ID']))
child1DF.loc[:, 'Voxel_after'] = childonly1
child1DF['True Ratio'] = child1DF['Voxel_after'] / child1DF['Voxel']

# Then record the regions that have more than one

chilmore1 = []
childm1DF = DeviDF.copy()
childm1DF = childm1DF[childm1DF['Child num'] > 1]

ii = 0
for iterID in childm1DF.loc[:]['ID']:
    temp = 0
    for iiterID in childm1DF.loc[iterID, 'Child ID'].split():
        # For all the possible child ID
        # By previous observations, all the child ID shows up here must in the nrrd file
        itemindex = np.where(copyarr == int(iiterID))
        temA = []
        temA = CropArr[itemindex]
        temp = temp + np.sum(temA == int(iiterID))
    chilmore1.append(temp)
    ii = ii + 1
    print('For more than 1-child brain regions, have finished ', ii / len(childm1DF.loc[:]['ID']))
childm1DF.loc[:, 'Voxel_after'] = chilmore1
childm1DF['True Ratio'] = childm1DF['Voxel_after'] / childm1DF['Voxel']

# Concat the previous result
DeviDF = pd.concat([child1DF, childm1DF])

# Reindex it

DeviDF = DeviDF.reindex(index=ccftable.index)

DeviDF.to_excel('/home/penglab/Documents/dataSource/DeviDF.xlsx')
