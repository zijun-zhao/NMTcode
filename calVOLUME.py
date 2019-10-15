# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:41:20 2019

@author: dell
"""
#%% Initialization
import numpy as np
import pandas as pd
import neuro_morpho_toolbox as nmt



#%%  Read the Mouse file
ccftable = pd.read_excel('/home/penglab/Documents/data/CCFv3 Summary Structures.xlsx', usecols=[1, 2, 3, 5, 6, 7], index_col=0,names=[ 'fullname', 'Abbrevation', 'depth in tree', 'structure_id_path','total_voxel_counts (10 um)'])
annofile = nmt.image('/home/penglab/Documents/data/annotation_10.nrrd')
bs = nmt.brain_structure('/home/penglab/Documents/data/Mouse.csv')

#%%  Read the nrrd file
arr3D = annofile.array
IDrange, IDcounts = np.unique(arr3D, return_counts=True)

#%%  Generate DataVoxel_ccf

'''
Dataframe **DataVoxel_ccf** stores the ID, Voxel calculated by zzj, and the difference between the groundtruth in the CCF. xsxl

Here the difference is calculated by the nrrd V- the CCF V


'''


my_cols = [i for i in range(20)]

# show the CCF table
DataVoxel2 = pd.DataFrame(index=IDrange,
                          columns=['ID', 'Full Structure Name', 'Abbreviation', 'depth in tree', 'Structure ID Path',
                                   'Voxel', 'Ref','diff(Voxel-Ref)'])
DataVoxel2['Voxel'] = IDcounts
# DataVoxel2

"""Rank the obtained information from the nrrd file according to the index in the CCF document"""
DataVoxel_ccf = DataVoxel2.reindex(index=ccftable.index)
DataVoxel_ccf['ID'] = ccftable.index
DataVoxel_ccf['Ref']= ccftable['total_voxel_counts (10 um)']
DataVoxel_ccf['diff(Voxel-Ref)'] = DataVoxel_ccf['Voxel'] - ccftable['total_voxel_counts (10 um)']
DataVoxel_ccf['Full Structure Name'] = ccftable['fullname']
DataVoxel_ccf['Abbreviation'] = ccftable['Abbrevation']
DataVoxel_ccf['Structure ID Path'] = ccftable['structure_id_path']
DataVoxel_ccf['depth in tree'] = ccftable['depth in tree']

print(DataVoxel_ccf['diff(Voxel-Ref)'])
# Delete regions that are not in the nrrd list
# Delete the NaN row





print("There are " + str(len(bs.df.index)) + " brain regions in the mouse file ")
print("There are " + str(len(IDrange)) + " brain regions in the mouse file ")
print("There are " + str(len(ccftable.index)) + " brain regions in the CCF file ")
if len(IDrange)!=len(ccftable.index):
    print("The number of ID in the nrrd file is not equal to the index in the CCF table, further merging is needed.")
#%% Merge the subregions of brain ID 
#Above does not consider the case of child regions
# Now need to analyze the number of child regions of each idx in CCF file
ccfIDX=list(ccftable.index)    
#Diffnot0DF is the dataframe where the difference is not 0.
Diffnot0DF=DataVoxel_ccf[DataVoxel_ccf['diff(Voxel-Ref)']!=0] 
#checkIDx is the corresponding Brain regions that needs to be checked.
checkIDx=list(Diffnot0DF.index)
for idd in checkIDx:
    child=(bs.get_all_child_id(idd))
    a=0;
    for childid in child:       
        if childid in IDrange:
            #print(a)
            loc=np.where(IDrange==childid)               
            a=a+IDcounts[loc]
        Diffnot0DF.loc[idd,'Voxel']=a
        #print(a)
# Delete the wrong column
Diffnot0DF.drop('diff(Voxel-Ref)', axis=1, inplace=True)
# Insert the correct new column
Diffnot0DF.insert((DataVoxel_ccf.shape[1]-1),'diff(Voxel-Ref)',Diffnot0DF['Voxel'] - ccftable['total_voxel_counts (10 um)']   )

OriginV = DataVoxel_ccf.copy()

# Only keep the dataframe where the diff is 0

OriginV= OriginV[OriginV ['diff(Voxel-Ref)']==0]

#Concate with the corrected dataframe
OriginV=pd.concat([OriginV,Diffnot0DF])
#Change the sequence according to the CCF file, name it as VoxelResult
VoxelResult=OriginV.reindex(index=ccftable.index)


# Add another column of information indicating the child ID, using ' ' to separate
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


#%% Speed up the code
#Look into the ~50micron problem 
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
arr3D = annofile.array
CropArr = np.zeros(arr3D.shape)
CropArr[0:int(dimx-ranX), 0:int(dimy-ranY), 0:int(dimz-ranZ)]= arr3D[int(ranX):int(dimx), int(ranY):int(dimy), int(ranZ): int(dimz)]

copyarr = annofile.array

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

#DeviDF.to_excel('D:\\neuro\\neuro_morhpo_toolbox-master\\neuro_morpho_toolbox\\data\\outpuuuaaat.xlsx')
