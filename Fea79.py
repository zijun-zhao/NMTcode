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
def get_axon_separately(testNeu, annotation, brain_structure, region_used=None):
    segment = testNeu.get_segments()
    tp = pd.DataFrame({"x": np.array(np.array(segment.x) / annotation.space['x'], dtype="int32"),
                       "y": np.array(np.array(segment.y) / annotation.space['y'], dtype="int32"),
                       "z": np.array(np.array(segment.z) / annotation.space['z'], dtype="int32"),
                       "rho": segment.rho,
                       "type": segment.type
                       })
    tp = tp[((tp.x >= 0) & (tp.x < annotation.size['x']) &
             (tp.y >= 0) & (tp.y < annotation.size['y']) &
             (tp.z >= 0) & (tp.z < annotation.size['z'])
            )]
    # print(np.max(tp[["x", "y", "z"]]))
    # assert (all(tp.x >= 0) & all(tp.x < annotation.size[0]) &
    #         all(tp.y >= 0) & all(tp.y < annotation.size[1]) &
    #         all(tp.z >= 0) & all(tp.z < annotation.size[2])
    #         ), "Error: SWC segments out of range."
    tp["region_id"] = annotation.array[tp.x, tp.y, tp.z]
    if region_used is None:
        region_used = brain_structure.selected_regions
    assert all([(i in brain_structure.level.index.tolist()) for i in region_used]), "Given regions invalid. Please check 'region_used'."

    midline = annotation.size['z']/ 2
    # Get output dataframe
    res = pd.DataFrame({"structure_id": np.append(region_used, region_used),
                        "hemisphere_id": [1]*len(region_used) + [2]*len(region_used)
                        })
    soma = []
    axon = []
    basal_dendrite = []
    apical_dendrite = []

    hemi_1 = tp[tp.z < midline]
    hemi_2 = tp[tp.z >=midline]
    for cur_hemi in [hemi_1, hemi_2]:
        for ct, cur_region in enumerate(region_used):
            child_ids = brain_structure.get_all_child_id(cur_region)
            # print("%d/%d\t%s\t%d" % (ct + 1,
            #                          len(region_used),
            #                          brain_structure.level.loc[cur_region, "Abbrevation"],
            #                          len(child_ids)))
            idx = []
            for i in child_ids:
                idx = idx + cur_hemi[cur_hemi.region_id == i].index.tolist()
            temp = cur_hemi.loc[idx]
            soma.append(np.sum(temp[temp.type == 1]["rho"]))
            axon.append(np.sum(temp[temp.type == 2]["rho"]))
            basal_dendrite.append(np.sum(temp[temp.type == 3]["rho"]))
            apical_dendrite.append(np.sum(temp[temp.type == 4]["rho"]))
    res["soma"] = soma
    res["axon"] = axon
    res["(basal) dendrite"] = basal_dendrite
    res["apical dendrite"] = apical_dendrite

    return res
#%% Define the Feature Table dataframe 

#%%
#Obtain all the brain regiosn
colname = mouseINFO.index
colname = colname.tolist()
colsum = ['sum'+str(x) for x in colname]
colleft = ['left'+str(x) for x in colname]
colright = ['right'+str(x) for x in colname]
#transfer int to list
col_list=colleft
col_list.extend(colright)
col_list.extend(colsum)
FeatureTb = pd.DataFrame(index=os.listdir('/home/penglab/Documents/CLA_swc'),columns=col_list)
#for info in os.listdir('/home/Zijun/Desktop/CLA_swc'):
#    domain = os.path.abspath('/home/Zijun/Desktop/CLA_sw') #Obtain the path of the file
iterrow=0
for info in os.listdir('/home/penglab/Documents/CLA_swc'):
    domain = os.path.abspath('/home/penglab/Documents/CLA_swc') #Obtain the path of the file
    infofull = os.path.join(domain,info) #Obtain the thorough path
    testNeu = nmt.neuron(infofull,zyx=False)      
    #First read the file
    #DF containing child and soma's type,rho, theta, phi, x, y, z
    testSEG = testNeu.get_segments()   
    #in case there is nan in the soma coordinates
    testSEG [testSEG .isnull()]=0
    #%% Obtain the axon information
    #First record the specific row corresponding to the axon
    cur_hemi = get_axon_separately(testNeu,anofile, bs, region_used = mouseINFO.index)   
    axonlist = cur_hemi['axon']

    FeatureTb.iloc[iterrow,0:2*len(mouseINFO.index)] = axonlist.values
    axonlist1 = axonlist[0:len(mouseINFO.index)]
    axonlist2 = axonlist[len(mouseINFO.index):]
    FeatureTb.iloc[iterrow,2*len(mouseINFO.index):] = axonlist1.values+axonlist2.values
    iterrow=iterrow+1    
    print('For all swc files, have finished ', iterrow/ len(os.listdir('/home/penglab/Documents/CLA_swc')))
    #CLA_swc
##After iteration, set NaN value to 0
FeatureTb=FeatureTb.fillna(0)
FeatureTb.to_excel('/home/penglab/Documents/dataSource/1000mean79.xlsx')








































