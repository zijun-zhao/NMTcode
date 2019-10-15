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

# %%Initialization

ccftable = pd.read_excel('/home/penglab/Documents/data/CCFv3 Summary Structures.xlsx', usecols=[1, 2, 3, 5, 6, 7], index_col=0,names=['', 'fullname', 'Abbrevation', 'depth in tree', 'structure_id_path','total_voxel_counts (10 um)'])
anofile = nmt.image('/home/penglab/Documents/data/annotation_10.nrrd')
bs = nmt.brain_structure('/home/penglab/Documents/data/Mouse.csv')
axonJaneliaccf = pd.read_excel('/home/penglab/Documents/dataSource/JaneliaAxonccf.xlsx', index_col=0)
SOMAJaneliaccf = pd.read_excel('/home/penglab/Documents/dataSource/JaneliaSomaccf.xlsx', index_col=0)
mouseDF = pd.read_excel('/home/penglab/Documents/dataSource/mouseDF.xlsx', index_col=0)
ccfDF = pd.read_excel('/home/penglab/Documents/dataSource/ccfDF.xlsx', index_col=0)
DeviDF = pd.read_excel('/home/penglab/Documents/dataSource/stable5.xlsx', index_col=0)
#%% Define function to transfer id to abbreviation
def id_to_name(region_id,bs):
    if region_id>0:
    # region_name can be either abbrevation (checked first) or description
        abbr=bs.level[bs.level.index ==region_id].Abbrevation.to_string()
    else:
        abbr='non'
    return abbr
     

#%%Define the comparing function
def JaneliaAnalysis(Feafile,ccfThre,norF,ctname,somalist,numC,somaLOC):
    #%%Store the soma location information
    import numpy as np
    import sklearn.cluster
    from sklearn.cluster import KMeans
    ccfDFsub = ccfDF[ccfDF['Child num']>ccfThre]
    #sortCcfDF = ccfDFsub .sort_values(['Child num'])
    del_list=ccfDFsub.index
    del_list = del_list.tolist()
    del_list1 = ['sum'+str(x) for x in del_list]
    del_list2 = ['ipsi'+str(x) for x in del_list]   
    del_list3 = ['contra'+str(x) for x in del_list]   
    for colidx in del_list1:
        if colidx in Feafile.columns:
            del Feafile[colidx]
    for colidx in del_list2:
        if colidx in Feafile.columns:
            del Feafile[colidx]
    for colidx in del_list3:
        if colidx in Feafile.columns:
            del Feafile[colidx]
            #the DF for analysis, storing the umap coordinates, soma region and plotting color
    
    anaDF = somaLOC.copy()
    #anaDF = pd.DataFrame(index=Feafile.index,columns=['ux','uy','SOMA','plotc'])
    anaDF['SOMA']=somalist     
    outRangelist=anaDF[anaDF['SOMA']==-1].index
    #print('\n')
    #print('The following swc files\' soma goes out of range: ',outRangelist)
    noSOMAlist=anaDF[anaDF['SOMA']==-2].index
    #print('\n')
    #print('The following swc files have no annotated soma: ',noSOMAlist)
    if 'Soma' in Feafile.columns:
        #do not use soma region as feature
        del Feafile['Soma']      
    typeR, typeC = np.unique(somalist, return_counts = True)
     #%%DEfine the soma plotting color
    
    #whether normalization or not
    if norF==1:
        Feafile[Feafile!=0]=np.log(Feafile[Feafile!=0])    
    colorind = range(len(typeR))
    i=0
    for typeiter in typeR:
        inddex = anaDF[anaDF ['SOMA']== typeiter].index  
        for ii in inddex:
            anaDF.loc[ii,'plotc']=colorind[i]  
        i=i+1
        if i ==10:
            i=0 

    #%% Use umap to map data from high dimension to low dimension
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns    
    import numpy as np
    from scipy.cluster.hierarchy import fcluster
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(Feafile.values)
    print('\n')
    print('Shape of the Umap result are ', embedding.shape)
    print('The result is an array with ' + str(embedding.shape[0]) + ' samples, but only ' + str(
        embedding.shape[1]) + ' feature columns (instead of the ' + str(Feafile.shape[1]) + ' we started with).')
    anaDF['ux'] = embedding[:,0]
    anaDF['uy'] = embedding[:,1]
    fig, ax = plt.subplots()  
    fig.suptitle('Coloring according to SOMA without legend indexed by CCF')     
    for typeiter in typeR:
        speRow = anaDF[anaDF['SOMA'] == typeiter]
        ax.scatter(speRow['ux'], speRow['uy'], c= sns.color_palette()[int(speRow['plotc'][0])], s=10) 
        #Now plotc stores the color to plot
    ax.grid(True) 
    #%%PLot the whole dataset using UMAP(CCF version)
    fig.show()    
    plt.figure()
    import seaborn as sns    
    plt.scatter(embedding[:, 0], embedding[:, 1], s=10,c=sns.color_palette("Set3", 10));
    plt.axis([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 0.5, np.min(embedding[:, 1]) - 0.5,
          np.max(embedding[:, 1]) + 0.5])
    plt.title('PLot the whole dataset using UMAP(CCF version)');
    plt.show()
    #%%for different ways of determining k.
    #Use the fcluster function.
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    #minimizes the total within-cluster variance
    Z = linkage(Feafile.values, 'ward')
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    c, coph_dists = cophenet(Z, pdist(Feafile.values))
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()
    anaDF['HierC']=  fcluster(Z, numC, criterion='maxclust')
    #%%Result of hierachchical on whole dataset indexed by CCF
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=anaDF['HierC'], s=10, cmap='rainbow');
    plt.axis([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 0.5, np.min(embedding[:, 1]) - 0.5,
              np.max(embedding[:, 1]) + 0.5])

    plt.title('Result of hierachchical on whole dataset indexed by CCF');
    plt.show()    
    #%% Add abbreviation row
    abbrlist=[]
    for iditer in anaDF['SOMA']:
        abbrlist.append(id_to_name(iditer,bs))
    anaDF['SOMA_abbr']=abbrlist
    
    #%% Delete the unclear rows
    anaDF.drop(outRangelist,inplace=True)
    anaDF.drop(noSOMAlist,inplace=True)
    Feafile.drop(outRangelist,inplace=True)
    Feafile.drop(noSOMAlist,inplace=True)
#%% Generate the confusion matrix

    ct = pd.crosstab(anaDF['SOMA_abbr'],  anaDF['HierC'] )    #print('*************************************************')
    from sklearn import metrics
    
    

    
    print('Estimated number of clusters: %d' % numC)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(anaDF['SOMA_abbr'],anaDF['HierC']))
    print("Completeness: %0.3f" % metrics.completeness_score(anaDF['SOMA_abbr'],anaDF['HierC']))
    print("V-measure: %0.3f" % metrics.v_measure_score(anaDF['SOMA_abbr'],anaDF['HierC']))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(anaDF['SOMA_abbr'],anaDF['HierC']))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(anaDF['SOMA_abbr'],anaDF['HierC']))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(Feafile.values, anaDF['HierC'], metric='sqeuclidean'))
    print("Cophenetic Correlation Coefficient: %0.3f"
          % c)
    print('j')
#Set the axis using the abbreviation
    ctindex=[]
    for i in range(len(ct.index)):
        if len(ct.index[i].split())<2:
            ctindex.append('non')
        else:
            ctindex.append(ct.index[i].split()[2])           
    ct.index=ctindex
    ctname="/home/penglab/Documents/confuMccf/"+ctname+".xlsx"
    ct.to_excel(ctname)

 



#%% For neuron whose soma is on the right part of the brain
#switch its "left" and "right" columns
#here "left" is actually isipi

axonJaneliaccf['soma_Z'] = SOMAJaneliaccf['soma_Z']
to_switch_J = axonJaneliaccf.copy()
not_switch_J = axonJaneliaccf.copy()
midline = anofile.size['z']/ 2
to_switch_J = to_switch_J[to_switch_J['soma_Z']>midline]
not_switch_J=not_switch_J[not_switch_J['soma_Z']<=midline]

del to_switch_J['soma_Z']
del not_switch_J['soma_Z']
colname = ccfDF.index
colname = colname.tolist()
colsum = ['sum'+str(x) for x in colname]
colleft = ['left'+str(x) for x in colname]
colright = ['right'+str(x) for x in colname]
#transfer int to list
col_list = colright.copy()
col_list.extend(colleft)
col_list.extend(colsum)
to_switch_J.columns=[col_list]

to_switch_J = pd.concat([to_switch_J.loc[:][colleft],to_switch_J.loc[:][colright],to_switch_J.loc[:][colsum]], axis=1)
(to_switch_J.columns) = (not_switch_J.columns)
#%%%%
#axonJaneliaccf=pd.concat([not_switch_J,to_switch_J], axis=1,sort=False)
axonJaneliaccf = not_switch_J.append(to_switch_J)
axonJaneliaccf = axonJaneliaccf.reindex(os.listdir('/home/penglab/Documents/Janelia_1000'))

colipsi = ['ipsi'+str(x) for x in colname]
colcontra = ['contra'+str(x) for x in colname]
col_list=colipsi.copy()
col_list.extend(colcontra)
col_list.extend(colsum)
axonJaneliaccf.columns=col_list




#%% Test the Jalinea Feature which is obtaind based on CCF index
if np.sum(axonJaneliaccf.index==SOMAJaneliaccf.index)==axonJaneliaccf.shape[0]:
    FeaJanelia=axonJaneliaccf.copy()
    FeaJanelia['Soma']=SOMAJaneliaccf['soma_Region']



#%%Test using the deleting table
  
somaLOC=SOMAJaneliaccf[['soma_X','soma_Y','soma_Z']].copy() 

  #%%----------------------------------Case1

# non-normalized (3) sum of 316 brain regions as features
print('***********CASE1 for 966 Features indexed by CCF***********')
ccfThre = np.max(ccfDF['Child num'])

JaneliaAnalysis(FeaJanelia.iloc[:,2*len(ccftable.index):],ccfThre,0,'case1',FeaJanelia['Soma'],20,somaLOC)

#%%----------------------------------Case2
#Use normalized (3) sum of 316 brain regions as features
print('***********CASE2 for 966 Features indexed by CCF***********')
ccfThre = np.max(ccfDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(ccftable.index):],ccfThre,1,'case2',FeaJanelia['Soma'],20,somaLOC)
#%%----------------------------------Case3
#Use normalized (1)(2) of 316 regions as features
print('***********CASE3 for 966 Features indexed by CCF***********')
ccfThre = np.max(ccfDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,0:2*len(ccftable.index)],ccfThre,1,'case3',FeaJanelia['Soma'],20,somaLOC)  
#%%----------------------------------Case4
#Use normalized (1)(2) of one-child regions as features
print('***********CASE4 for 966 Features indexed by CCF***********')
ccfThre = 1
JaneliaAnalysis(FeaJanelia.iloc[:,0:2*len(ccftable.index)],ccfThre,1,'case4',FeaJanelia['Soma'],20,somaLOC)  
#%%----------------------------------Case5
#%Use normalized (1)(2) of less-than-median regions as features
print('***********CASE5 for 966 Features indexed by CCF***********')
ccfThre =np.mean(ccfDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,0:2*len(ccftable.index)],ccfThre,1,'case5',FeaJanelia['Soma'],20,somaLOC)  
#%%----------------------------------Case6
#Use normalized (3) of one-child regions as features
print('***********CASE6 for 966 Features indexed by CCF***********')
ccfThre =1
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(ccftable.index):],ccfThre,1,'case6',FeaJanelia['Soma'],20,somaLOC)  
#%%----------------------------------Case7
#Use normalized (3) of less-than-median regions as features    
print('***********CASE7 for 966 Features indexed by CCF***********')
ccfThre = np.mean(ccfDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(ccftable.index):],ccfThre,1,'case7',FeaJanelia['Soma'],20,somaLOC)  
#%%VIsualize the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
iterrow=0
for info in os.listdir('/home/penglab/Documents/confuMccf'):
    domain = os.path.abspath('/home/penglab/Documents/confuMccf') #Obtain the path of the file
    infofull = os.path.join(domain,info) #Obtain the thorough path       
    caseDF = pd.read_excel(infofull, index_col=0)
    plt.figure()
    #print('\n  Print the heatmap of confusion matrix')
    sns.heatmap(caseDF,cmap='YlGnBu')
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    