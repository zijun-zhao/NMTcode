#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 03:40:08 2019

@author: penglab
"""

#%% Initialization
import numpy as np
import pandas as pd
import os
import neuro_morpho_toolbox as nmt
anofile = nmt.image('/home/penglab/Documents/data/annotation_25.nrrd')
bs = nmt.brain_structure('/home/penglab/Documents/data/Mouse.csv')
def id_to_name(region_id,bs):
    if region_id>0:
    # region_name can be either abbrevation (checked first) or description
        abbr=bs.level[bs.level.index ==region_id].Abbrevation.to_string()
    else:
        abbr='non'
    return abbr
#
'''
marker=pd.read_csv('/home/penglab/Documents/harris3D_corners.marker', index_col=0)
marker.columns
marker['x']=marker.index

infoDF = pd.DataFrame(columns = ['x','y','z'])
infoDF['x'] = marker['x'].copy()
infoDF['y'] = marker['y'].copy()
infoDF['z'] = marker['z'].copy()
'''
infoDF = pd.DataFrame(columns = ['x','y','z'])

infoDF['x'] = np.repeat(range(anofile.array.shape[0]), anofile.array.shape[1]*anofile.array.shape[2]) 

lst_y = np.repeat(range(anofile.array.shape[1]), anofile.array.shape[2]) 

infoDF['y'] = np.tile(np.repeat(range(anofile.array.shape[1]), anofile.array.shape[2]) ,anofile.array.shape[0])

infoDF['z'] =np.tile(range(anofile.array.shape[2]) ,anofile.array.shape[0]*anofile.array.shape[1])
idx_infoDF = ['node'+str(x) for x in range(infoDF.shape[0])]    
infoDF['idx'] = idx_infoDF
infoDF.set_index('idx',inplace=True)   
indexlist=[]         
for idx in infoDF.index:
    indexlist.append(anofile.array[infoDF.loc[idx]['x'],infoDF.loc[idx]['y'],infoDF.loc[idx]['z']])

abbrlist=[]  
infoDF['id'] = indexlist
for idx in infoDF.index:
    abbrlist.append(id_to_name(infoDF.loc[idx]['id'],bs))
modi_abbrlist=[]
for abbriter in abbrlist:
    if len(abbriter.split())>1:
        modi_abbrlist.append(abbriter.split()[-1])
    else:
        modi_abbrlist.append('non')
        
infoDF['abbr'] = modi_abbrlist