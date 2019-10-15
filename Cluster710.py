#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:19:37 2019

@author: penglab
"""




import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import neuro_morpho_toolbox as nmt
#import seaborn as sns

# %%Initialization
ccftable = pd.read_excel('/home/penglab/Documents/data/CCFv3 Summary Structures.xlsx', usecols=[1, 2, 3, 5, 6, 7], index_col=0,names=[ 'fullname', 'Abbrevation', 'depth in tree', 'structure_id_path','total_voxel_counts (10 um)'])
anofile = nmt.image('/home/penglab/Documents/data/annotation_10.nrrd')
bs = nmt.brain_structure('/home/penglab/Documents/data/Mouse.csv')

axonJanelia = pd.read_excel('/home/penglab/Documents/dataSource/JaneliaAxon.xlsx', index_col=0)
axonCLA = pd.read_excel('/home/penglab/Documents/dataSource/ClaAxon.xlsx', index_col=0)

SOMAJanelia = pd.read_excel('/home/penglab/Documents/dataSource/JaneliaSoma.xlsx', index_col=0)
somaCLA = pd.read_excel('/home/penglab/Documents/dataSource/ClaSoma.xlsx', index_col=0)

mouseDF = pd.read_excel('/home/penglab/Documents/dataSource/mouseDF.xlsx', index_col=0)
#%%Define the comparing function
def comparRE(Feafile,subXie,mouseThre):
    import sklearn.cluster
    from sklearn.cluster import KMeans
    mouseDFsub = mouseDF[mouseDF['Child num']>mouseThre]
    #sortMouseDF = mouseDFsub .sort_values(['Child num'])
    del_list=mouseDFsub.index
    del_list = del_list.tolist()
    del_list1 = ['sum'+str(x) for x in del_list]
    del_list2 = ['left'+str(x) for x in del_list]   
    del_list3 = ['right'+str(x) for x in del_list]   
    for colidx in del_list1:
        if colidx in Feafile.columns:
            del Feafile[colidx]
    for colidx in del_list2:
        if colidx in Feafile.columns:
            del Feafile[colidx]
    for colidx in del_list3:
        if colidx in Feafile.columns:
            del Feafile[colidx]
    #%% Use umap to map data from high dimension to low dimension
    import umap
    import matplotlib.pyplot as plt
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(Feafile.values)
    print('\n')
    print('Shape of the Umap result are ', embedding.shape)
    print('The result is an array with ' + str(embedding.shape[0]) + ' samples, but only ' + str(
        embedding.shape[1]) + ' feature columns (instead of the ' + str(Feafile.shape[1]) + ' we started with).')
    #color_range=np.tile(range(10), embedding.shape[0]//10)
    import seaborn as sns    
    plt.scatter(embedding[:, 0], embedding[:, 1], s=10,c=sns.color_palette("Set3", 10));
    plt.axis([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 0.5, np.min(embedding[:, 1]) - 0.5,
          np.max(embedding[:, 1]) + 0.5])
    plt.title('PLot the whole dataset', fontsize=24);
    plt.show()

     #%%Show the SOMA type

    ShowXie = pd.DataFrame(index=subXie.index,columns=['ux','uy','Subtype','plotc'])
    typeR, typeC = np.unique(subXie['Subtype'], return_counts = True)
    ShowXie['ux'] = embedding100[:,0]
    ShowXie['uy'] = embedding100[:,1]
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
    fig.show()
    
     #Show the result of Kmeans on whole dataset
    subtypeR, subtypeC = np.unique(subXie['Subtype'], return_counts=True)
    n_clusters = len(subtypeR)
    # correct number of clusters
    estimator= KMeans(n_clusters, random_state=100)


    kmeans_labels =estimator.fit_predict(Feafile.values)
    #centerCLUST=estimator.cluster_centers_
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans_labels, s=10, cmap='rainbow');
    plt.axis([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 0.5, np.min(embedding[:, 1]) - 0.5,
              np.max(embedding[:, 1]) + 0.5])
    plt.title('Result of k-means on whole dataset', fontsize=24);
    plt.show()    
    inertia = estimator.inertia_
    print('The inertia will be ',inertia,', note that lower value will be better, 0 is optimized.')    
    

    
    
    subXie['kmeansRst']=  kmeans_labels[-100:]
    ct = pd.crosstab(subXie['Subtype'],  subXie['kmeansRst'] )
    # Display ct
    import seaborn as sns
    sns.set()
    print(ct)
    sns.heatmap(ct,annot=True)   
    #%%Show the original subtype
        #Umap of CLA files
    reducer = umap.UMAP()
    embedding100 = embedding[-100:,:]
    ShowXie = pd.DataFrame(index=subXie.index,columns=['ux','uy','Subtype','plotc'])
    typeR, typeC = np.unique(subXie['Subtype'], return_counts = True)
    ShowXie['ux'] = embedding100[:,0]
    ShowXie['uy'] = embedding100[:,1]
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
    fig.show()
    
     #Show the result of Kmeans on whole dataset
    subtypeR, subtypeC = np.unique(subXie['Subtype'], return_counts=True)
    n_clusters = len(subtypeR)
    # correct number of clusters
    estimator= KMeans(n_clusters, random_state=100)


    kmeans_labels =estimator.fit_predict(Feafile.values)
    #centerCLUST=estimator.cluster_centers_
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans_labels, s=10, cmap='rainbow');
    plt.axis([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 0.5, np.min(embedding[:, 1]) - 0.5,
              np.max(embedding[:, 1]) + 0.5])
    plt.title('Result of k-means on whole dataset', fontsize=24);
    plt.show()    
    inertia = estimator.inertia_
    print('The inertia will be ',inertia,', note that lower value will be better, 0 is optimized.')    
    

    
    
    subXie['kmeansRst']=  kmeans_labels[-100:]
    ct = pd.crosstab(subXie['Subtype'],  subXie['kmeansRst'] )
    # Display ct
    import seaborn as sns
    sns.set()
    print(ct)
    sns.heatmap(ct,annot=True)

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
#%% Only test Jalinea Dataset
if np.sum(axonJanelia.index==SOMAJanelia.index)==axonJanelia.shape[0]:
    FeaJanelia=axonJanelia.copy()
    FeaJanelia['Soma']=SOMAJanelia['soma_Region']
    




# %% Concate the Jalinea and CLA's axon feature dataframe


conFea_axon = pd.concat([axonJanelia, FeaCLA])
print('\n')
print('Shape of the concated axon feature dataframe is ', conFea.shape)

nor_conFea = conFea.copy()

nor_conFea[nor_conFea!=0]=np.log(nor_conFea[nor_conFea!=0])

#%%Obtain the soma dataframe
conFea_soma = pd.concat([SOMAJanelia, somaCLA])
print('\n')

SOMAJanelia = pd.read_excel('/home/penglab/Documents/dataSource/JaneliaSoma.xlsx', index_col=0)

somaCLA





outRangelist=SOMAJanelia[SOMAJanelia['soma_Region']==-1].index
print('\n')
print('The following Janelia swc files\' soma goes out of range: ',outRangelist)
noSOMAlist=SOMAJanelia[SOMAJanelia['soma_Region']==-2].index
print('\n')
print('The following Janelia swc files have no annotated soma: ',noSOMAlist)
#%%----------------------------------Case1
# non-normalized (3) sum of 1327 brain regions as features
mouseThre = np.max(mouseDF['Child num'])
testDF = conFea.iloc[:,2*len(bs.df.index):]
comparRE(testDF,subXie,mouseThre)


#%%----------------------------------Case2
#Use normalized (3) sum of 1327 brain regions as features
mouseThre = np.max(mouseDF['Child num'])
comparRE(nor_conFea.iloc[:,2*len(bs.df.index):],subXie,mouseThre)



#%%----------------------------------Case3
#Use normalized (1)(2) of 1327 regions as features
mouseThre = np.max(mouseDF['Child num'])
comparRE(nor_conFea.iloc[:,0:2*len(bs.df.index)],subXie,mouseThre)

#%%----------------------------------Case4
#Use normalized (1)(2) of one-child regions as features
mouseThre = 1
comparRE(nor_conFea.iloc[:,0:2*len(bs.df.index)],subXie,mouseThre)
#%%----------------------------------Case5
#%Use normalized (1)(2) of less-than-median regions as features
mouseThre = np.mean(mouseDF['Child num'])
comparRE(nor_conFea.iloc[:,0:2*len(bs.df.index)],subXie,mouseThre)


#%%----------------------------------Case6
#Use normalized (3) of one-child regions as features
mouseThre = 1
comparRE(nor_conFea.iloc[:,2*len(bs.df.index):],subXie,mouseThre)
#%%----------------------------------Case7
#Use normalized (3) of less-than-median regions as features
mouseThre = np.mean(mouseDF['Child num'])
comparRE(nor_conFea.iloc[:,2*len(bs.df.index):],subXie,mouseThre)



















#----------------------------------Case2
mouseThre = 1
#COnsider only the separated left and right,without normalization
comparRE(conFea.iloc[:,0:2*len(bs.df.index)],subXie,mouseThre)
#----------------------------------Case3
mouseThre = np.mean(mouseDF['Child num'])
comparRE(conFea,subXie,mouseThre)






















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

























