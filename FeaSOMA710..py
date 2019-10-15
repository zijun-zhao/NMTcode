#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 02:39:49 2019

@author: penglab
"""

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

mouseDF = pd.read_excel('/home/penglab/Documents/dataSource/mouseDF.xlsx', index_col=0)
mouseINFO = mouseDF.copy()

#%%
def get_SOMA_region(testNeu, annotation, brain_structure, radiusSOMA,mouseINFO):
    #First read the file
    #DF containing child and soma's type,rho, theta, phi, x, y, z
    testSEG = testNeu.get_segments()   
    #in case there is nan in the soma coordinates
    testSEG [testSEG .isnull()]=0
    #%% Obtain the sphere near the soma
    #First record the specific row corresponding to the soma
    soma_row = testSEG[testSEG['type']==1]
    if soma_row.shape[0]<1:
        somaDF = pd.DataFrame(columns=['idx', 'Child ID', 'Child num', 'Soma count', 'x', 'y', 'z','region_id'])
        somaDF.loc['nn']=-2
        print('************NOTE THAT the following file does not have soma whose type is 1: ',info)
    else:
        #extract the coordinates of the soma
        soma_x = int(np.mean(soma_row['x']))
        soma_y = int(np.mean(soma_row['y']))
        soma_z = int(np.mean(soma_row['z']))
        #set the radius of the sphere
        r_soma = radiusSOMA
        #Record the sphere near the soma
        soma_Sphere = testSEG[((testSEG.x <= soma_x+r_soma) & (testSEG.x > soma_x-r_soma) &
                             (testSEG.y <= soma_y+r_soma) & (testSEG.y > soma_y-r_soma) &
                             (testSEG.z <= soma_z+r_soma) & (testSEG.z > soma_z-r_soma)
                            )]
            
            
        #Return the dataframe, storing the index,brain region ID and rho,theta,phi, x,y,z
        #soma_Sphere
        #Note that the coordinates needed to be modified according to nrrd file's space
        #Note that we need to consider the sampling ratio
        BR_somaDF = pd.DataFrame({"x": np.array(np.array(soma_Sphere.x) /  annotation.space['x'], dtype="int32"),
                                       "y": np.array(np.array(soma_Sphere.y) /  annotation.space['y'], dtype="int32"),
                                       "z": np.array(np.array(soma_Sphere.z) /  annotation.space['z'], dtype="int32"),
                                       "type":soma_Sphere.type,
                                       "rho": soma_Sphere.rho,
                                       "theta": soma_Sphere.theta,
                                       "phi": soma_Sphere.phi})
        #Delete the swc information whose coordinates is outside the nrrd file
        BR_somaDF = BR_somaDF[((BR_somaDF.x >= 0) & (BR_somaDF.x < anofile.size['x']) &
                             (BR_somaDF.y >= 0) & (BR_somaDF.y < anofile.size['y']) &
                             (BR_somaDF.z >= 0) & (BR_somaDF.z < anofile.size['z'])
                            )]
        BR_somaDF["region_id"] =  annotation.array[BR_somaDF.x,BR_somaDF.y, BR_somaDF.z]
        #Set the datafrane's index as the region_id shows up in the dataframe
        BR_somaDF.set_index('region_id')
        IDrange, IDcounts = np.unique(BR_somaDF["region_id"], return_counts = True)
        
        somaDIStri = mouseINFO.copy()
        #print('Now going to analyze the child region info of the regions in Mouse file\'s index')
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
            #print('For 1-child brain regions in soma, have finished ', ii / len(child1DF.loc[:]['Child ID']))
                #store the sum to the dataframe directly, not using a string
            child1DF.loc[iterID, 'Soma count'] = temp
        child1DF=child1DF.fillna(0)
        
    
               
        # Concat the previous result
    #    conCHILDsoma = pd.concat([child1DF, childm1DF])
        #since soma region is composed of small regions, here only consider the small region
        conCHILDsoma = child1DF.copy()
        #sort the value according to soma_count
        
        conCHILDsoma = child1DF.copy()
        conCHILDsoma= conCHILDsoma.reindex(index=bs.df.index,fill_value='0')   
        
         # Reindex it   
        somaDIStri['Soma count']=conCHILDsoma['Soma count']
        somaDIStri=somaDIStri.fillna(0)
    
        copysomadis=somaDIStri.copy()
        copysomadis['Soma count']= copysomadis['Soma count'].astype(int)
        #reorder
        somaSORT=copysomadis.sort_values(by='Soma count', ascending=False, na_position='first')
        somaINDEX=somaSORT[somaSORT['Soma count']!=0].index
        #stores the ['idx', 'Child ID', 'Child num', 'Soma count'] for nonzero rows
        somaDF = somaSORT.loc[somaINDEX]
        #Error may occur if there is no 
        if somaDF.shape[0]>0:
            child_id = somaDF.loc[somaDF.index[0],:]['Child ID']
            somaDF['region_id'] = child_id
            if np.sum(somaDF['Soma count']==somaDF.iloc[0,3])>1:
                iddx=somaDF[somaDF['Soma count']==somaDF.iloc[0,3]].index
                BR_corDF= pd.DataFrame(columns=BR_somaDF.columns)
                for ii in iddx:
                    BR_corDF=BR_corDF.append(BR_somaDF[BR_somaDF['region_id']==int(ii)],ignore_index=False)
                #more than one region has more count
                somaDF ['x'] = np.mean(BR_corDF['x'])
                somaDF ['y'] = np.mean(BR_corDF['y'])
                somaDF ['z'] = np.mean(BR_corDF['z'])
            else:
                somaDF ['x'] = np.mean(BR_somaDF[BR_somaDF['region_id']==int(child_id)]['x'])
                somaDF ['y'] = np.mean(BR_somaDF[BR_somaDF['region_id']==int(child_id)]['y'])
                somaDF ['z'] = np.mean(BR_somaDF[BR_somaDF['region_id']==int(child_id)]['z'])        
            #BR_somaDF[BR_somaDF['region_id']==int(child_id)]
        else:
            # Then record the regions that have more than one
            #chilmore1 = []
            childm1DF = mouseINFO.copy()
            childm1DF = childm1DF[childm1DF['Child num'] > 1]
            #ii = 0
            for iterID in childm1DF.index:
                temp = 0
                for iiterID in childm1DF.loc[iterID, 'Child ID'].split():
                # For all the possible child ID
                # By previous observations, all the child ID shows up here must in the nrrd file
                    temp = temp + np.sum(BR_somaDF["region_id"] == int(iiterID))
                childm1DF.loc[iterID, 'Soma count'] = temp
                #chilmore1.append(temp)
                #ii = ii + 1
                #print('For more than 1-child brain regions in soma, have finished ', ii / len(childm1DF.loc[:]['Child ID']))
            #childm1DF.loc[:, 'Soma count'] = chilmore1
            childm1DF=childm1DF.fillna(0)
            nonezerochildm1DF=childm1DF[childm1DF['Soma count']!=0]
            if nonezerochildm1DF.shape[0]>0:
                conCHILDsoma=childm1DF.copy()
                #change the order and let the most counting region rank first
                conCHILDsoma=childm1DF.sort_values(by='Soma count', ascending=False, na_position='first')
                conCHILDsoma=conCHILDsoma[conCHILDsoma['Soma count']!=0]
                conCHILDsoma=conCHILDsoma.sort_values(by='Child num', ascending=True, na_position='first')
                somaDF=conCHILDsoma.copy()
                child_id = conCHILDsoma.index[0]
                #BR_somaDF[BR_somaDF['region_id']==childm1DF.index[0]
                BR_corDF= pd.DataFrame(columns=BR_somaDF.columns)
                for ii in mouseINFO.loc[child_id]['Child ID'].split():      
                    BR_corDF=BR_corDF.append(BR_somaDF[BR_somaDF['region_id']==int(ii)],ignore_index=False)
                somaDF ['x'] = np.mean(BR_corDF['x'])
                somaDF ['y'] = np.mean(BR_corDF['y'])
                somaDF ['z'] = np.mean(BR_corDF['z'])
                somaDF['region_id'] = child_id
            else:
                somaDF = somaDF.join(pd.DataFrame(columns=['x','y','z','region_id'], index=somaDF.index))
                somaDF.loc['nn']=-1
                
                print('************NOTE THAT the following file has gone out of range ',info)

    return somaDF 
#%% Define the Feature Table dataframe 

#%%
#Obtain all the brain regiosn

#transfer int to list
col_list=['soma_Region','soma_Range','soma_Count','soma_X','soma_Y','soma_Z']
FeatureTb = pd.DataFrame(index=os.listdir('/home/penglab/Documents/Janelia_1000'),columns=col_list)
#for info in os.listdir('/home/Zijun/Desktop/CLA_swc'):
#    domain = os.path.abspath('/home/Zijun/Desktop/CLA_sw') #Obtain the path of the file
iterrow=0
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
    somaDF = get_SOMA_region(testNeu, anofile, bs, 20,mouseINFO)
    FeatureTb.loc[info,'soma_X'] = np.mean(somaDF['x'])
    FeatureTb.loc[info,'soma_Y'] = np.mean(somaDF['y'])
    FeatureTb.loc[info,'soma_Z'] = np.mean(somaDF['z'])
    list2=[str(i) for i in somaDF['Child ID']]  
    list3=' '.join(list2) 
    FeatureTb.loc[info,'soma_Range'] = list3
    list2=[str(i) for i in somaDF['Soma count']]  
    list3=' '.join(list2) 
    FeatureTb.loc[info,'soma_Count'] = list3    
    FeatureTb.loc[info,'soma_Count'] = list3  
    FeatureTb.loc[info,'soma_Region'] = somaDF['region_id'].values[0]
    iterrow=iterrow+1    
    print('For all swc files, have finished ', iterrow/ len(os.listdir('/home/penglab/Documents/Janelia_1000')))
    #CLA_swc
##After iteration, set NaN value to 0
FeatureTb=FeatureTb.fillna(0)
FeatureTb.to_excel('/home/penglab/Documents/dataSource/JaneliaSoma.xlsx')







outRangelist=FeatureTb[FeatureTb['soma_Region']==-1].index
print('\n')
print('The following swc files\' soma goes out of range: ',outRangelist)
noSOMAlist=FeatureTb[FeatureTb['soma_Region']==-2].index
print('\n')
print('The following swc files have no annotated soma: ',noSOMAlist)




































































