#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:52:04 2019

@author: penglab
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:53:15 2019

@author: dell
"""



#%% Initialization
import numpy as np
import pandas as pd
import os
import neuro_morpho_toolbox as nmt



#ccftable = pd.read_excel('D:\\2019Spring\\Intern\\neuro_morhpo_toolbox-master\\neuro_morpho_toolbox\\data\\CCFv3 Summary Structures.xlsx', usecols=[1, 2, 3, 5, 6, 7], index_col=0,names=['', 'fullname', 'Abbrevation', 'depth in tree', 'structure_id_path','total_voxel_counts (10 um)'])
ccftable = pd.read_excel('/home/penglab/Documents/data/CCFv3 Summary Structures.xlsx', usecols=[1, 2, 3, 5, 6, 7], index_col=0,names=[ 'fullname', 'Abbrevation', 'depth in tree', 'structure_id_path','total_voxel_counts (10 um)'])
#Read the corresponding annotation file.
anofile = nmt.image('/home/penglab/Documents/data/annotation_10.nrrd')

#Read the corresponding brain structure file. Obtain the information of child regions

bs = nmt.brain_structure('/home/penglab/Documents/data/Mouse.csv')


'''
#%%Use a dataframe to store the childregion information of Mouse.csv
mouseINFO = pd.DataFrame(index = bs.df.index,columns=['idx','Child ID','Child num'])
mouseINFO['idx'] = mouseINFO.index
for iterIdx in mouseINFO.index:
    child_temp=bs.get_all_child_id(iterIdx)
    child_T_store=[]
    for iiterIdx in child_temp:
        child_T_store.append(iiterIdx)
            #store all the ID as a whole string
    list2=[str(i) for i in child_T_store]  
    list3=' '.join(list2) 
    mouseINFO.loc[iterIdx,'Child ID']=list3
# Add another column of information indicating the child number    
Childnum_store=[]
#iiiter=0
for IDchild in mouseINFO.index:
    lenchild=len(mouseINFO.loc[IDchild,'Child ID'].split())
    mouseINFO.loc[IDchild,'Child num'] = lenchild

#mouseINFO.to_excel('/home/Zijun/Desktop/dataSource/res2.xlsx')   
mouseINFO.to_excel('/home/penglab/Documents/dataSource/mouseDF.xlsx')
'''
mouseDF = pd.read_excel('/home/penglab/Documents/dataSource/mouseDF.xlsx', index_col=0)
mouseINFO = mouseDF.copy()

#%%
#%% Define the specific ID
#np.mean(mouseDF['Child num'])
childRestri = np.mean(mouseDF['Child num'])
childRestri = 1
selectedID = mouseINFO[mouseINFO['Child num']<=childRestri].index

#%% Define the Feature Table dataframe 
iterrow=0
#%%
#Obtain all the brain regiosn
colname = selectedID
colname = colname.tolist()
#transfer int to list
#colname.extend(['soma_Region','soma_Range','soma_Count','soma_X','soma_Y','soma_Z'])

FeatureTb = pd.DataFrame(index=os.listdir('/home/penglab/Documents/Janelia_1000'),columns=colname)
#for info in os.listdir('/home/Zijun/Desktop/CLA_swc'):
#    domain = os.path.abspath('/home/Zijun/Desktop/CLA_sw') #Obtain the path of the file
for info in os.listdir('/home/penglab/Documents/Janelia_1000'):
    domain = os.path.abspath('/home/penglab/Documents/Janelia_1000') #Obtain the path of the file
    infofull = os.path.join(domain,info) #Obtain the thorough path
    testNeu = nmt.neuron(infofull,zyx=False)      
    #First read the file
    #DF containing child and soma's type,rho, theta, phi, x, y, z
    testSEG = testNeu.get_segments()   
    #in case there is nan in the soma coordinates
    testSEG [testSEG .isnull()]=0
    #%% Obtain the axon information
    #First record the specific row corresponding to the axon
    cur_hemi = testNeu.get_region_matrix(anofile, bs, region_used = selectedID)
    FeatureTb.iloc[iterrow,0:len(selectedID)] = cur_hemi['axon']       
    iterrow=iterrow+1    
    print('For all swc files, have finished ', iterrow/ len(os.listdir('/home/penglab/Documents/Janelia_1000')))
##After iteration, set NaN value to 0
FeatureTb=FeatureTb.fillna(0)
FeatureTb.to_excel('/home/penglab/Documents/dataSource/1000mean79.xlsx')

#%% Define a function return the information of axon version 1.o
            
def axonINFO(testSEG,hemidx,childRestri): 
    axon_row=testSEG[testSEG['type']==2]
    #Return the dataframe, storing the index,brain region ID and rho,theta,phi, x,y,z
        #axon_Sphere
        #Note that the coordinates needed to be modified according to nrrd file's space
        #Note that we need to consider the sampling ratio
    BR_axonDF = pd.DataFrame({"x": np.array(np.array(axon_row.x) / anofile.space['x'], dtype="int32"),
                              "y": np.array(np.array(axon_row.y) / anofile.space['y'], dtype="int32"),
                              "z": np.array(np.array(axon_row.z) / anofile.space['z'], dtype="int32"),
                              "type":axon_row.type,
                              "rho": axon_row.rho,
                              "theta":axon_row.theta,
                              "phi": axon_row.phi})
        
    BR_axonDF_whole = BR_axonDF[((BR_axonDF.x >= 0) & (BR_axonDF.x < anofile.size['x']) &
                        (BR_axonDF.y >= 0) & (BR_axonDF.y < anofile.size['y']) &
                        (BR_axonDF.z >= 0) & (BR_axonDF.z < anofile.size['z'])
                    )]
    BR_axonDF_whole["region_id"] = anofile.array[BR_axonDF.x,BR_axonDF.y, BR_axonDF.z]
    IDrange, IDcounts = np.unique(BR_axonDF_whole["region_id"], return_counts = True)
    selectedID = mouseINFO[mouseINFO['Child num']<=childRestri] 
    axonDIStri = pd.DataFrame(index = selectedID,columns=['Child num', 'Child ID', 'Region_len'])
    axonDIStri['Child ID']=np.nan
    for iterIdx in axonDIStri.index:
        child_temp=bs.get_all_child_id(iterIdx)
        child_T_store=[]
        for iiterIdx in child_temp:
            child_T_store.append(iiterIdx)
                #store all the ID as a whole string
    list2=[str(i) for i in child_T_store]  
    list3=' '.join(list2) 
    axonDIStri.loc[iterIdx,'Child ID']=list3
    # Add another column of information indicating the child number    

    for IDchild in axonDIStri.index:
        lenchild=len(axonDIStri.loc[IDchild,'Child ID'].split())
        axonDIStri.loc[IDchild,'Child num']=lenchild  
    
    # First record the regions only have one subregions(may be only itself)
    #childonly1 store the corresponding list of counting numbers
    #childonly1 = []
    child1DF = axonDIStri.copy()
    child1DF = child1DF[child1DF['Child num'] == 1]
    ii = 0
    for iterID in child1DF.index:
        iiterID = child1DF.loc[iterID]['Child ID']
        temp = np.sum(BR_axonDF["region_id"] == int(iiterID))
        #idxrow=BR_axonDF[BR_axonDF['region_id']==iiterID]
        len1 = np.sum(BR_axonDF['rho'])
        ii = ii + 1
        print('For 1-child brain regions in axon, have finished ', ii / len(child1DF.loc[:]['Child ID']))
        child1DF.loc[iterID, 'Region_count'] = temp
        child1DF.loc[iterID, 'Region_len'] = len1
        temp=0
    
    
    # Then record the regions that have more than one
    chilmore1 = []
    chilmorelen = []
    childm1DF = axonDIStri.copy()
    childm1DF = childm1DF[childm1DF['Child num'] > 1]
    ii = 0
    for iterID in childm1DF.index:
        temp = 0
        len2 = 0
        for iiterID in childm1DF.loc[iterID, 'Child ID'].split():
        # For all the possible child ID
        # By previous observations, all the child ID shows up here must in the nrrd file
            temp = temp + np.sum(BR_axonDF["region_id"] == int(iiterID))
            len2 = len2 + np.sum(BR_axonDF['rho'])
        chilmore1.append(temp)
        chilmorelen.append(len2)
        ii = ii + 1
        print('For more than 1-child brain regions in axon, have finished ', ii / len(childm1DF.loc[:]['Child ID']))
    childm1DF.loc[:, 'Region_count'] = chilmore1  
    childm1DF.loc[:, 'Region_len'] = chilmorelen 
    
    
    
    # Concat the previous result
    conCHILD = pd.concat([child1DF, childm1DF])
    conCHILD = conCHILD.fillna(0)
     # Reindex it   
    conCHILD =conCHILD.reindex(index=axonDIStri.index)   
    conCHILD=conCHILD.fillna(0)
    #set the value to axonDIStri 
    axonDIStri['Region_len']=conCHILD['Region_len']
    axonDIStri['Region_len'] = np.log(axonDIStri['Region_len'])
    #axonDIStri[axonDIStri['Region_len']<0 ]=0
    axonDIStri[axonDIStri['Region_len']<-10000000 ]=0
    #Reorder it
    axonDIStri = axonDIStri.reindex(index=ccftable.index)   
    axoL1 = np.sum(np.array(conCHILD['Region_len']))
    #axonDIStri['Region_len'] = axonDIStri['Region_len'] /np.sum(np.array(conCHILD['Region_len']))*100000
    regL1= axonDIStri['Region_len'] 
    if hemiID!=0:
        return axoL1,regL1
    else:
        if hemiID==1:
            hemiID=2
        else:
            hemiID=1
        if hemiID==1:
            BR_axonDF = BR_axonDF[((BR_axonDF.x >= 0) & (BR_axonDF.x < anofile.size['x']) &
                            (BR_axonDF.y >= 0) & (BR_axonDF.y < anofile.size['y']) &
                            (BR_axonDF.z >= 0) & (BR_axonDF.z < anofile.size['z']*0.5)
                        )]
        if hemiID==2:
            BR_axonDF = BR_axonDF[((BR_axonDF.x >= 0) & (BR_axonDF.x < anofile.size['x']) &
                            (BR_axonDF.y >= 0) & (BR_axonDF.y < anofile.size['y']) &
                            (BR_axonDF.z >= anofile.size['z']*0.5) & (BR_axonDF.z < anofile.size['z'])
                        )]
        BR_axonDF["region_id"] = anofile.array[BR_axonDF.x,BR_axonDF.y, BR_axonDF.z]
        IDrange, IDcounts = np.unique(BR_axonDF["region_id"], return_counts = True)
        
        axonDIStri = pd.DataFrame(index=ccftable.index,columns=['Child num', 'Child ID', 'Region_len'])
        axonDIStri['Child ID']=np.nan
        for iterIdx in axonDIStri.index:
            '''Here note that the get_all_child_id cannot input value 0'''
            if iterIdx==0:
                somaDIStri.loc[iterIdx,'Child ID']=0
            else:       
                child_temp=bs.get_all_child_id(iterIdx)
                child_T_store=[]
                for iiterIdx in child_temp:
                     if iiterIdx in IDrange:
                        child_T_store.append(iiterIdx)
                    #store all the ID as a whole string
                list2=[str(i) for i in child_T_store]  
                list3=' '.join(list2) 
                axonDIStri.loc[iterIdx,'Child ID']=list3
        # Add another column of information indicating the child number    

        for IDchild in axonDIStri.index:
            lenchild=len(axonDIStri.loc[IDchild,'Child ID'].split())
            axonDIStri.loc[IDchild,'Child num']=lenchild  
        
        # First record the regions only have one subregions(may be only itself)
        #childonly1 store the corresponding list of counting numbers
        #childonly1 = []
        child1DF = axonDIStri.copy()
        child1DF = child1DF[child1DF['Child num'] == 1]
        ii = 0
        for iterID in child1DF.index:
            iiterID = child1DF.loc[iterID]['Child ID']
            temp = np.sum(BR_axonDF["region_id"] == int(iiterID))
            
            len1 = np.sum(BR_axonDF['rho'])
            ii = ii + 1
            print('For 1-child brain regions in axon, have finished ', ii / len(child1DF.loc[:]['Child ID']))
            child1DF.loc[iterID, 'Region_count'] = temp
            child1DF.loc[iterID, 'Region_len'] = len1
            temp=0
        















#%% Define a function return the information of axon
            
def axonINFO(testSEG,somax,somay,somaz,hemiID): 
    axon_row=testSEG[testSEG['type']==2]
    #Return the dataframe, storing the index,brain region ID and rho,theta,phi, x,y,z
        #axon_Sphere
        #Note that the coordinates needed to be modified according to nrrd file's space
        #Note that we need to consider the sampling ratio
    BR_axonDF = pd.DataFrame({"x": np.array(np.array(axon_row.x) / anofile.space['x'], dtype="int32"),
                              "y": np.array(np.array(axon_row.y) / anofile.space['y'], dtype="int32"),
                              "z": np.array(np.array(axon_row.z) / anofile.space['z'], dtype="int32"),
                              "type":axon_row.type,
                              "rho": axon_row.rho,
                              "theta":axon_row.theta,
                              "phi": axon_row.phi})
    #Delete the swc information whose coordinates is outside the nrrd file
    FeatureTb.loc[info,'hemiINFO'] = hemiID
    if hemiID==0:
        #see if it is on the left side
        if somaz<anofile.size['z']*0.5:
            hemiID==1
        else:
            hemiID=2
        
    if hemiID==1:
        BR_axonDF = BR_axonDF[((BR_axonDF.x >= 0) & (BR_axonDF.x < anofile.size['x']) &
                        (BR_axonDF.y >= 0) & (BR_axonDF.y < anofile.size['y']) &
                        (BR_axonDF.z >= 0) & (BR_axonDF.z < anofile.size['z']*0.5)
                    )]
    if hemiID==2:
        BR_axonDF = BR_axonDF[((BR_axonDF.x >= 0) & (BR_axonDF.x < anofile.size['x']) &
                        (BR_axonDF.y >= 0) & (BR_axonDF.y < anofile.size['y']) &
                        (BR_axonDF.z >= anofile.size['z']*0.5) & (BR_axonDF.z < anofile.size['z'])
                    )]
    BR_axonDF["region_id"] = anofile.array[BR_axonDF.x,BR_axonDF.y, BR_axonDF.z]
    IDrange, IDcounts = np.unique(BR_axonDF["region_id"], return_counts = True)
    
    axonDIStri = pd.DataFrame(index=ccftable.index,columns=['Child num', 'Child ID', 'Region_len'])
    axonDIStri['Child ID']=np.nan
    for iterIdx in axonDIStri.index:
        '''Here note that the get_all_child_id cannot input value 0'''
        if iterIdx==0:
            somaDIStri.loc[iterIdx,'Child ID']=0
        else:       
            child_temp=bs.get_all_child_id(iterIdx)
            child_T_store=[]
            for iiterIdx in child_temp:
                 if iiterIdx in IDrange:
                    child_T_store.append(iiterIdx)
                #store all the ID as a whole string
            list2=[str(i) for i in child_T_store]  
            list3=' '.join(list2) 
            axonDIStri.loc[iterIdx,'Child ID']=list3
    # Add another column of information indicating the child number    
    Childnum_store=[]
    iiiter=0
    for IDchild in axonDIStri.index:
        lenchild=len(axonDIStri.loc[IDchild,'Child ID'].split())
        axonDIStri.loc[IDchild,'Child num']=lenchild  
    
    # First record the regions only have one subregions(may be only itself)
    #childonly1 store the corresponding list of counting numbers
    #childonly1 = []
    child1DF = axonDIStri.copy()
    child1DF = child1DF[child1DF['Child num'] == 1]
    ii = 0
    for iterID in child1DF.index:
        iiterID = child1DF.loc[iterID]['Child ID']
        temp = np.sum(BR_axonDF["region_id"] == int(iiterID))
        idxrow=BR_axonDF[BR_axonDF['region_id']==iiterID]
        len1 = np.sum(BR_axonDF['rho'])
        ii = ii + 1
        print('For 1-child brain regions in axon, have finished ', ii / len(child1DF.loc[:]['Child ID']))
        child1DF.loc[iterID, 'Region_count'] = temp
        child1DF.loc[iterID, 'Region_len'] = len1
        temp=0
    
    
    # Then record the regions that have more than one
    chilmore1 = []
    chilmorelen = []
    childm1DF = axonDIStri.copy()
    childm1DF = childm1DF[childm1DF['Child num'] > 1]
    ii = 0
    for iterID in childm1DF.index:
        temp = 0
        len2 = 0
        for iiterID in childm1DF.loc[iterID, 'Child ID'].split():
        # For all the possible child ID
        # By previous observations, all the child ID shows up here must in the nrrd file
            temp = temp + np.sum(BR_axonDF["region_id"] == int(iiterID))
            idxrow=BR_axonDF[BR_axonDF['region_id']==iiterID]
            len2 = len2 + np.sum(BR_axonDF['rho'])
        chilmore1.append(temp)
        chilmorelen.append(len2)
        ii = ii + 1
        print('For more than 1-child brain regions in axon, have finished ', ii / len(childm1DF.loc[:]['Child ID']))
    childm1DF.loc[:, 'Region_count'] = chilmore1  
    childm1DF.loc[:, 'Region_len'] = chilmorelen 
    
    
    
    # Concat the previous result
    conCHILD = pd.concat([child1DF, childm1DF])
    conCHILD = conCHILD.fillna(0)
     # Reindex it   
    conCHILD =conCHILD.reindex(index=axonDIStri.index)   
    conCHILD=conCHILD.fillna(0)
    #set the value to axonDIStri 
    axonDIStri['Region_len']=conCHILD['Region_len']
    axonDIStri['Region_len'] = np.log(axonDIStri['Region_len'])
    #axonDIStri[axonDIStri['Region_len']<0 ]=0
    axonDIStri[axonDIStri['Region_len']<-10000000 ]=0
    #Reorder it
    axonDIStri = axonDIStri.reindex(index=ccftable.index)   
    axoL1 = np.sum(np.array(conCHILD['Region_len']))
    #axonDIStri['Region_len'] = axonDIStri['Region_len'] /np.sum(np.array(conCHILD['Region_len']))*100000
    regL1= axonDIStri['Region_len'] 
    if hemiID!=0:
        return axoL1,regL1
    else:
        if hemiID==1:
            hemiID=2
        else:
            hemiID=1
        if hemiID==1:
            BR_axonDF = BR_axonDF[((BR_axonDF.x >= 0) & (BR_axonDF.x < anofile.size['x']) &
                            (BR_axonDF.y >= 0) & (BR_axonDF.y < anofile.size['y']) &
                            (BR_axonDF.z >= 0) & (BR_axonDF.z < anofile.size['z']*0.5)
                        )]
        if hemiID==2:
            BR_axonDF = BR_axonDF[((BR_axonDF.x >= 0) & (BR_axonDF.x < anofile.size['x']) &
                            (BR_axonDF.y >= 0) & (BR_axonDF.y < anofile.size['y']) &
                            (BR_axonDF.z >= anofile.size['z']*0.5) & (BR_axonDF.z < anofile.size['z'])
                        )]
        BR_axonDF["region_id"] = anofile.array[BR_axonDF.x,BR_axonDF.y, BR_axonDF.z]
        IDrange, IDcounts = np.unique(BR_axonDF["region_id"], return_counts = True)
        
        axonDIStri = pd.DataFrame(index=ccftable.index,columns=['Child num', 'Child ID', 'Region_len'])
        axonDIStri['Child ID']=np.nan
        for iterIdx in axonDIStri.index:
            '''Here note that the get_all_child_id cannot input value 0'''
            if iterIdx==0:
                somaDIStri.loc[iterIdx,'Child ID']=0
            else:       
                child_temp=bs.get_all_child_id(iterIdx)
                child_T_store=[]
                for iiterIdx in child_temp:
                     if iiterIdx in IDrange:
                        child_T_store.append(iiterIdx)
                    #store all the ID as a whole string
                list2=[str(i) for i in child_T_store]  
                list3=' '.join(list2) 
                axonDIStri.loc[iterIdx,'Child ID']=list3
        # Add another column of information indicating the child number    

        for IDchild in axonDIStri.index:
            lenchild=len(axonDIStri.loc[IDchild,'Child ID'].split())
            axonDIStri.loc[IDchild,'Child num']=lenchild  
        
        # First record the regions only have one subregions(may be only itself)
        #childonly1 store the corresponding list of counting numbers
        #childonly1 = []
        child1DF = axonDIStri.copy()
        child1DF = child1DF[child1DF['Child num'] == 1]
        ii = 0
        for iterID in child1DF.index:
            iiterID = child1DF.loc[iterID]['Child ID']
            temp = np.sum(BR_axonDF["region_id"] == int(iiterID))
            
            len1 = np.sum(BR_axonDF['rho'])
            ii = ii + 1
            print('For 1-child brain regions in axon, have finished ', ii / len(child1DF.loc[:]['Child ID']))
            child1DF.loc[iterID, 'Region_count'] = temp
            child1DF.loc[iterID, 'Region_len'] = len1
            temp=0
        
        
        # Then record the regions that have more than one
        chilmore1 = []
        chilmorelen = []
        #store the corresponding information where child num >1
        childm1DF = axonDIStri.copy()
        childm1DF = childm1DF[childm1DF['Child num'] > 1]
        ii = 0
        for iterID in childm1DF.index:
            temp = 0
            len2 = 0
            for iiterID in childm1DF.loc[iterID, 'Child ID'].split():
            # For all the possible child ID
            # By previous observations, all the child ID shows up here must in the nrrd file
                temp = temp + np.sum(BR_axonDF["region_id"] == int(iiterID))
                
                len2 = len2 + np.sum(BR_axonDF['rho'])
            chilmore1.append(temp)
            chilmorelen.append(len2)
            ii = ii + 1
            print('For more than 1-child brain regions in axon, have finished ', ii / len(childm1DF.loc[:]['Child ID']))
        childm1DF.loc[:, 'Region_count'] = chilmore1  
        childm1DF.loc[:, 'Region_len'] = chilmorelen 
        
        
        
        # Concat the previous result
        conCHILD = pd.concat([child1DF, childm1DF])
        conCHILD = conCHILD.fillna(0)
         # Reindex it   
        conCHILD =conCHILD.reindex(index=axonDIStri.index)   
        conCHILD=conCHILD.fillna(0)
        #set the value to axonDIStri 
        axonDIStri['Region_len']=conCHILD['Region_len']
        axonDIStri['Region_len'] = np.log(axonDIStri['Region_len'])
        #axonDIStri[axonDIStri['Region_len']<0 ]=0
        axonDIStri[axonDIStri['Region_len']<-10000000 ]=0
        #Reorder it
        axonDIStri = axonDIStri.reindex(index=ccftable.index)   
        axoL = np.sum(np.array(conCHILD['Region_len']))
        #axonDIStri['Region_len'] = axonDIStri['Region_len'] /np.sum(np.array(conCHILD['Region_len']))*100000
        regL= axonDIStri['Region_len'] 
        
        return axoL1,regL1, axoL,regL         


#%% Define the Feature Table dataframe 
iterrow=0
FeatureTb = pd.DataFrame(index=os.listdir('/home/Zijun/Desktop/CLA_swc'),columns=colname)
for info in os.listdir('/home/Zijun/Desktop/CLA_swc'):
    domain = os.path.abspath('/home/Zijun/Desktop/CLA_swc') #Obtain the path of the file
    infofull = os.path.join(domain,info) #Obtain the thorough path
    testNeu = nmt.neuron(infofull,zyx=False) 
        
    #First read the file
    #DF containing child and soma's type,rho, theta, phi, x, y, z
    testSEG = testNeu.get_segments()   
    #in case there is nan in the soma coordinates
    testSEG [testSEG .isnull()]=0
    #%% Obtain the sphere near the soma
    #First record the specific row corresponding to the soma
    soma_row = testSEG[testSEG['type']==1]
    #extract the coordinates of the soma
    soma_x = int(np.mean(soma_row['x']))
    soma_y = int(np.mean(soma_row['y']))
    soma_z = int(np.mean(soma_row['z']))
    #set the radius of the sphere
    r_soma = 20
    #Record the sphere near the soma
    soma_Sphere = testSEG[((testSEG.x <= soma_x+r_soma) & (testSEG.x > soma_x-r_soma) &
                         (testSEG.y <= soma_y+r_soma) & (testSEG.y > soma_y-r_soma) &
                         (testSEG.z <= soma_z+r_soma) & (testSEG.z > soma_z-r_soma)
                        )]
        
        
    #Return the dataframe, storing the index,brain region ID and rho,theta,phi, x,y,z
    #soma_Sphere
    #Note that the coordinates needed to be modified according to nrrd file's space
    #Note that we need to consider the sampling ratio
    BR_somaDF = pd.DataFrame({"x": np.array(np.array(soma_Sphere.x) / anofile.space['x'], dtype="int32"),
                                   "y": np.array(np.array(soma_Sphere.y) / anofile.space['y'], dtype="int32"),
                                   "z": np.array(np.array(soma_Sphere.z) / anofile.space['z'], dtype="int32"),
                                   "type":soma_Sphere.type,
                                   "rho": soma_Sphere.rho,
                                   "theta": soma_Sphere.theta,
                                   "phi": soma_Sphere.phi})
    #Delete the swc information whose coordinates is outside the nrrd file
    BR_somaDF = BR_somaDF[((BR_somaDF.x >= 0) & (BR_somaDF.x < anofile.size['x']) &
                         (BR_somaDF.y >= 0) & (BR_somaDF.y < anofile.size['y']) &
                         (BR_somaDF.z >= 0) & (BR_somaDF.z < anofile.size['z'])
                        )]
    BR_somaDF["region_id"] = anofile.array[BR_somaDF.x,BR_somaDF.y, BR_somaDF.z]
    #Set the datafrane's index as the region_id shows up in the dataframe
    BR_somaDF.set_index('region_id')
    IDrange, IDcounts = np.unique(BR_somaDF["region_id"], return_counts = True)
    
    somaDIStri = pd.DataFrame(index=ccftable.index,columns=['Child num', 'Child ID', 'Soma count'])
    somaDIStri['Child ID']=np.nan
    '''Here the index in somaDIStri will be the ID showing up in the n'''
    print('Now going to analyze the child region info of the regions in CCF file\'s index')
    for iterIdx in somaDIStri.index:
        '''Here note that the get_all_child_id cannot input value 0'''
        if iterIdx not in bs.df.index:
            somaDIStri.loc[iterIdx,'Child ID']=0
            
            print('Region '+str(iterIdx)+' is not recorded in the brain structure')
        else:       
            child_temp=bs.get_all_child_id(iterIdx)
            child_T_store=[]
            for iiterIdx in child_temp:
                 if iiterIdx in IDrange:
                    child_T_store.append(iiterIdx)
                #store all the ID as a whole string
            list2=[str(i) for i in child_T_store]  
            list3=' '.join(list2) 
            somaDIStri.loc[iterIdx,'Child ID']=list3
    # Add another column of information indicating the child number    
    Childnum_store=[]
    iiiter=0
    for IDchild in somaDIStri.index:
        lenchild=len(somaDIStri.loc[IDchild,'Child ID'].split())
        somaDIStri.loc[IDchild,'Child num']=lenchild  
    
    # First record the regions only have one subregions(may be only itself)
    #childonly1 store the corresponding list of counting numbers
    '''This part is very easy to have problems, we need to know that child id may not be the index, we also need to be careful about the iteration'''
    child1DF = somaDIStri.copy()
    child1DF = child1DF[child1DF['Child num'] == 1]
    ii = 0
    for iterID in child1DF.index:
        iiterID=child1DF.loc[iterID, 'Child ID']
        temp = np.sum(BR_somaDF["region_id"] == int(iiterID))
        ii = ii + 1
        print('For 1-child brain regions in soma, have finished ', ii / len(child1DF.loc[:]['Child ID']))
            #store the sum to the dataframe directly, not using a string
        child1DF.loc[iterID, 'Soma count'] = temp
    child1DF=child1DF.fillna(0)
    
    # Then record the regions that have more than one
    #chilmore1 = []
    childm1DF = somaDIStri.copy()
    childm1DF = childm1DF[childm1DF['Child num'] > 1]
    ii = 0
    for iterID in childm1DF.index:
        temp = 0
        for iiterID in childm1DF.loc[iterID, 'Child ID'].split():
        # For all the possible child ID
        # By previous observations, all the child ID shows up here must in the nrrd file
            temp = temp + np.sum(BR_somaDF["region_id"] == int(iiterID))
        childm1DF.loc[iterID, 'Soma count'] = temp
        #chilmore1.append(temp)
        ii = ii + 1
        print('For more than 1-child brain regions in soma, have finished ', ii / len(childm1DF.loc[:]['Child ID']))
    #childm1DF.loc[:, 'Soma count'] = chilmore1
    childm1DF=childm1DF.fillna(0)
           
    # Concat the previous result
    conCHILDsoma = pd.concat([child1DF, childm1DF])
    
    #sort the value according to soma_count
    conCHILDsoma = conCHILDsoma.copy()
    conCHILDsoma = conCHILDsoma.reindex(index=ccftable.index,fill_value='0')   
    
     # Reindex it   
    somaDIStri['Soma count']=conCHILDsoma['Soma count']
    somaDIStri=somaDIStri.fillna(0)

    copysomadis=somaDIStri.copy()
    copysomadis['Soma count']= copysomadis['Soma count'].astype(int)
    copysomadis=copysomadis.sort_values(by='Soma count', ascending=False, na_position='first')
    #Record the region of the soma
    FeatureTb.loc[info,'soma_Region'] = copysomadis.index[0]
    #Record the coordinates of the soma acoording to the calculated brain region
    #First extract the child id of the specific brain region
    child_id = somaDIStri.loc[copysomadis.index[0],:]['Child ID']
    #find the corresponding rows in the sphere
    SOMAcorDF = pd.DataFrame(columns=BR_somaDF.columns)
    for iiiiterID in child_id.split():
        SOMAcorDF=SOMAcorDF.append(BR_somaDF[BR_somaDF['region_id']==int(iiiiterID)],ignore_index=False)
    cor_x = np.mean(SOMAcorDF['x'])
    cor_y = np.mean(SOMAcorDF['y'])
    cor_z = np.mean(SOMAcorDF['z'])
    FeatureTb.loc[info,'soma_X'] = cor_x
    FeatureTb.loc[info,'soma_Y'] = cor_y
    FeatureTb.loc[info,'soma_Z'] = cor_z      
    
    
    
    #If more regions are detected, record the region of possible soma areas 
    copysomadis = copysomadis.reindex(index=ccftable.index)
    copysomadis=copysomadis[copysomadis['Soma count']>0]
    list_IDrange = []
    list_IDcounts = []
    for iiterIdx in copysomadis.index:
        list_IDrange.append(iiterIdx)
        #store all the ID as a whole string
        list2 = [str(i) for i in list_IDrange]  
        list3 = ' '.join(list2) 
        list4 = [str(i) for i in list_IDcounts]  
        list5 = ' '.join(list4)
    for iiterIdx in copysomadis['Soma count']:
        list_IDcounts.append(iiterIdx)
        #store all the ID as a whole string
        list4 = [str(i) for i in list_IDcounts]  
        list5 = ' '.join(list4)
    FeatureTb.loc[info,'soma_Range'] = list3
    FeatureTb.loc[info,'soma_Count'] = list5
    
    #------------------------------------------------------------------------#
    #%% Obtain the axon information
    #First record the specific row corresponding to the axon
    cur_hemi = testNeu.get_region_matrix(anofile, bs, region_used=None)
    hemisame = judgesame(cur_hemi)
    #if the axon only appears in one hemisphere
    if hemisame!=0:
        axoL,regL = axonINFO(testSEG,cor_x,cor_y,cor_z,hemisame)
        FeatureTb.loc[info,'Axon_length'] = axoL
        FeatureTb.iloc[iterrow,0:len(ccftable.index)] = regL
        
    else:
        axoL1,regL1, axoL,regL = axonINFO(testSEG,cor_x,cor_y,cor_z,hemisame)
        FeatureTb.loc[info,'Axon_length'] = axoL1+axoL
        FeatureTb.loc[info,'Axon_sameL'] = axoL1
        FeatureTb.loc[info,'Axon_nonsameL']=axoL
        FeatureTb.iloc[iterrow,0:len(ccftable.index)] = regL1
        FeatureTb.iloc[iterrow,len(ccftable.index):2*len(ccftable.index)] = regL
        
            

       
    iterrow=iterrow+1    
    
    print(infofull)
    print('For all swc files, have finished ', iterrow/ len(os.listdir('/home/Zijun/Desktop/CLA_swc')))
#After iteration, set NaN value to 0
FeatureTb=FeatureTb.fillna(0)


FeatureTb.to_excel('/home/Zijun/Desktop/dataSource/res2.xlsx')    
    
    
print('The final dataframe has been saved as excel document')
    
