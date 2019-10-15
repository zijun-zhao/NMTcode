#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:19:37 2019

@author: penglab
"""


#cd /home/penglab/Downloads/neuro_morhpo_toolbox-master

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
def JaneliaAnalysis(Feafile,mouseThre,norF,ctname,somalist):
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
    anaDF = pd.DataFrame(index=Feafile.index,columns=['ux','uy','SOMA','plotc'])
    anaDF['SOMA']=somalist
    typeR, typeC = np.unique(anaDF['SOMA'], return_counts = True)
    
    if 'Soma' in Feafile.columns:
        del Feafile['Soma']
    if norF==1:
        Feafile[Feafile!=0]=np.log(Feafile[Feafile!=0])
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
    anaDF['ux'] = embedding[:,0]
    anaDF['uy'] = embedding[:,1]
    colorind = range(len(typeR))
    i=0
    for typeiter in typeR:
        inddex = anaDF[anaDF ['SOMA']== typeiter].index  
        for ii in inddex:
            anaDF.loc[ii,'plotc']=colorind[i]
            #print(colorind[i])
        i=i+1
        if i ==10:
            i=0
    '''
    fig, ax = plt.subplots()  
    fig.suptitle('Coloring according to SOMA with legend')  
    for typeiter in typeR:
        speRow = anaDF[anaDF['SOMA'] == typeiter]
        ax.scatter(speRow['ux'], speRow['uy'], c= sns.color_palette()[speRow['plotc'][0]], s=10,label = typeiter ) 
    #Now plotc stores the color to plot
    ax.legend()
    ax.grid(True)   
    fig.show()
    '''
    fig, ax = plt.subplots()  
    fig.suptitle('Coloring according to SOMA without legend')  
    for typeiter in typeR:
        speRow = anaDF[anaDF['SOMA'] == typeiter]
        ax.scatter(speRow['ux'], speRow['uy'], c= sns.color_palette()[speRow['plotc'][0]], s=10) 
    #Now plotc stores the color to plot

    ax.grid(True)   
    fig.show()    
     #%%Show the result of Kmeans on whole dataset
    typeR, typeC = np.unique(anaDF['SOMA'], return_counts = True)
    n_clusters = len(typeR)
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
    

    
    
    anaDF['kmeans Result']=  kmeans_labels
    ct = pd.crosstab(anaDF['SOMA'],  anaDF['kmeans Result'] )
    # Display ct
    import seaborn as sns
    sns.set()
    #print(ct)
    #sns.heatmap(ct,annot=True)   
    ctname="/home/penglab/Documents/confuM/"+ctname+".xlsx"
    ct.to_excel(ctname)

#%% Only test Jalinea Dataset
if np.sum(axonJanelia.index==SOMAJanelia.index)==axonJanelia.shape[0]:
    FeaJanelia=axonJanelia.copy()
    FeaJanelia['Soma']=SOMAJanelia['soma_Region']
    
#%% Delete the nonsoma rows
    
outRangelist=FeaJanelia[FeaJanelia['Soma']==-1].index
print('\n')
print('The following swc files\' soma goes out of range: ',outRangelist)
noSOMAlist=FeaJanelia[FeaJanelia['Soma']==-2].index
print('\n')
print('The following swc files have no annotated soma: ',noSOMAlist)

smallFEaJanelia=FeaJanelia.copy()
smallFEaJanelia.drop(outRangelist,inplace=True)
smallFEaJanelia.drop(noSOMAlist,inplace=True)

#%%Visualize the soma region distribution after deleting 
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
print('\n  Print the kernal density estimation of soma distribution') 
sns.distplot(smallFEaJanelia['Soma'],rug=True,hist=True)
   
    

somaR, somaC = np.unique(smallFEaJanelia['Soma'], return_counts = True)       
sns.heatmap(np.diag(somaR),cmap='YlGnBu')           
    
    
    
    
    
    
    
    
    
    
    
    
    
#%%----------------------------------Case1
# non-normalized (3) sum of 1327 brain regions as features
print('***********CASE1***********')
mouseThre = np.max(mouseDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(bs.df.index):],mouseThre,0,'case1',FeaJanelia['Soma'])


#%%----------------------------------Case2
#Use normalized (3) sum of 1327 brain regions as features
print('***********CASE2***********')
mouseThre = np.max(mouseDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(bs.df.index):],mouseThre,1,'case2',FeaJanelia['Soma'])

#%%----------------------------------Case3
#Use normalized (1)(2) of 1327 regions as features
print('***********CASE3***********')
mouseThre = np.max(mouseDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,0:2*len(bs.df.index)],mouseThre,1,'case3',FeaJanelia['Soma'])  

#%%----------------------------------Case4
#Use normalized (1)(2) of one-child regions as features
print('***********CASE4***********')
mouseThre = 1
JaneliaAnalysis(FeaJanelia.iloc[:,0:2*len(bs.df.index)],mouseThre,1,'case4',FeaJanelia['Soma'])  


#%%----------------------------------Case5
#%Use normalized (1)(2) of less-than-median regions as features
print('***********CASE5***********')
mouseThre =np.median(mouseDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,0:2*len(bs.df.index)],mouseThre,1,'case5',FeaJanelia['Soma'])  


#%%----------------------------------Case6
#Use normalized (3) of one-child regions as features
print('***********CASE6***********')
mouseThre =1
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(bs.df.index):],mouseThre,1,'case6',FeaJanelia['Soma'])  


#%%----------------------------------Case7
#Use normalized (3) of less-than-median regions as features    
print('***********CASE7***********')
mouseThre = np.median(mouseDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(bs.df.index):],mouseThre,1,'case7',FeaJanelia['Soma'])  

    
#%%VIsualize the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
iterrow=0
for info in os.listdir('/home/penglab/Documents/confuM'):
    domain = os.path.abspath('/home/penglab/Documents/confuM') #Obtain the path of the file
    infofull = os.path.join(domain,info) #Obtain the thorough path       
    caseDF = pd.read_excel(infofull, index_col=0)
    plt.figure()
    print('\n  Print the heatmap of confusion matrix')
    sns.heatmap(caseDF,cmap='YlGnBu')



somaR, somaC = np.unique(FeaJanelia['Soma'], return_counts = True)       
sns.heatmap(np.diag(somaR),cmap='YlGnBu')       