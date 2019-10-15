#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:21:45 2019

@author: penglab
"""


import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import neuro_morpho_toolbox as nmt
import seaborn as sns

# %%Initialization
ccftable = pd.read_excel('/home/penglab/Documents/data/CCFv3 Summary Structures.xlsx', usecols=[1, 2, 3, 5, 6, 7], index_col=0,names=[ 'fullname', 'Abbrevation', 'depth in tree', 'structure_id_path','total_voxel_counts (10 um)'])
anofile = nmt.image('/home/penglab/Documents/data/annotation_10.nrrd')
bs = nmt.brain_structure('/home/penglab/Documents/data/Mouse.csv')
#Feafile = pd.read_excel('/home/penglab/Documents/dataSource/REgion1.xlsx', index_col=0)
Feafile = pd.read_excel('/home/penglab/Documents/dataSource/mean79.xlsx', index_col=0)

mouseDF = pd.read_excel('/home/penglab/Documents/dataSource/mouseDF.xlsx', index_col=0)

#%%Visualizing dataset structure

#sns.jointplot(x='idx', y='Child num', data = mouseDF.values)
























# %% Compare with Dr. Xie's result
# Here in the subXie dataframe, only keep the original file's 'Soma_region', 'Brain_id', 'SWC_File', 'Celltype','Subtype'
Xiecluster = pd.read_csv('/home/penglab/Documents/data/clusters.csv',
                         names=['', 'cluster', 'Hemisphere', 'Soma_region', 'Brain_id', 'SWC_File', 'Celltype',
                                'Subtype', 'Bilateral'], engine='python',
                         skiprows=[0],
                         index_col=[1],
                         skipinitialspace=True)

subXie = Xiecluster[['Soma_region', 'Brain_id', 'SWC_File', 'Celltype', 'Subtype']].copy()
# Change Dr.Xie's Soma_region to 'Xie soma abbr'
subXie.rename(columns={'Soma_region': 'Xie Soma_Abbr'}, inplace=True)
subXie.index = Xiecluster['SWC_File']
subXie = subXie.reindex(index=os.listdir('/home/penglab/Documents/CLA_swc'), fill_value='0')

# %%
# Use umap to map data from high dimension to low dimension
import umap
import matplotlib.pyplot as plt
import seaborn as sns
reducer = umap.UMAP()
embedding = reducer.fit_transform(Feafile.values)
print('\n')
print('Shape of the Umap result are ', embedding.shape)
print('The result is an array with ' + str(embedding.shape[0]) + ' samples, but only ' + str(
    embedding.shape[1]) + ' feature columns (instead of the ' + str(Feafile.shape[1]) + ' we started with).')

#Show the original subtype
ShowXie = pd.DataFrame(index=subXie.index,columns=['ux','uy','Subtype','plotc'])
typeR, typeC = np.unique(subXie['Subtype'], return_counts = True)
ShowXie['ux'] = embedding[:,0]
ShowXie['uy'] = embedding[:,1]
ShowXie['Subtype'] = subXie['Subtype']
colorind = range(len(typeR))
i=0
for typeiter in typeR:
    inddex = ShowXie [ShowXie ['Subtype']== typeiter].index  
    for ii in inddex:
        ShowXie .loc[ii,'plotc']=colorind[i]
        #print(colorind[i])
    i=i+1
    
fig, ax = plt.subplots()  
fig.suptitle('Coloring according to subtype')  
for typeiter in typeR:
    speRow = ShowXie [ShowXie ['Subtype'] == typeiter]
    ax.scatter(speRow['ux'], speRow['uy'], c= sns.color_palette()[speRow['plotc'][0]], s=10,label = typeiter )
    
#Now plotc stores the color to plot
ax.legend()
ax.grid(True)   




# %% Try kmeans by deleting some features
import sklearn.cluster
from sklearn.cluster import KMeans
mouseDFsub = mouseDF[mouseDF['Child num']<=np.mean(mouseDF['Child num'])]
sortMouseDF = mouseDFsub .sort_values(['Child num'])

#del Feafile['Axon_length1']
#del Feafile['Axon_length2']
#delete the specific columns
#sortMouseDF.drop(sortMouseDF.index[-1], inplace=True)
#del Feafile[sortMouseDF.index[-1]]




subtypeR, subtypeC = np.unique(subXie['Subtype'], return_counts=True)
n_clusters = len(subtypeR)
kmeans_labels = sklearn.cluster.KMeans(n_clusters).fit_predict(Feafile.values)
plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans_labels, s=10, cmap='rainbow');
plt.axis([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 0.5, np.min(embedding[:, 1]) - 0.5,
          np.max(embedding[:, 1]) + 0.5])
plt.title('Result of k-means', fontsize=24);
plt.show()

subXie['kmeansRst']=  kmeans_labels
ct = pd.crosstab(subXie['Subtype'],  subXie['kmeansRst'] )
# Display ct
import seaborn as sns
sns.set()
print(ct)
sns.heatmap(ct,annot=True)



























# %% Cauculate the Euclidean distance matrix
from sklearn.metrics.pairwise import euclidean_distances

plt.matshow(euclidean_distances(Feafile.values, Feafile.values))
plt.colorbar()
plt.title('Show the Euclidean distance matrix')
plt.show()

# %%Combined usage
# The following example shows how easy coclust is to run several algorithms on the same dataset
import matplotlib.pyplot
import numpy as np, scipy.sparse as sp, scipy.io as io
from sklearn.metrics import confusion_matrix
from coclust.coclustering import (CoclustMod, CoclustSpecMod, CoclustInfo)
from coclust.visualization import plot_reorganized_matrix

X = Feafile.values
model_1 = CoclustMod(n_clusters=4, n_init=4)
model_1.fit(X)
model_2 = CoclustSpecMod(n_clusters=4, n_init=4)
model_2.fit(X)
model_3 = CoclustInfo(n_row_clusters=3, n_col_clusters=4, n_init=4)
model_3.fit(X)
plt.figure()

plt.title(' plot three reorganized matrices for the dataset')
plt.subplot(131)
plot_reorganized_matrix(X, model_1)
plt.subplot(132)
plot_reorganized_matrix(X, model_2)
plt.subplot(133)
plot_reorganized_matrix(X, model_3)
plt.show()
# plot the resulting reorganized matrices in order to have a first visual grasp of what can be expected from the different algorithms. A plot of three different reorganized matrices






































