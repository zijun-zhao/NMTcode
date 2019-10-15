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

ccftable = pd.read_excel('/home/penglab/Documents/data/CCFv3 Summary Structures.xlsx', usecols=[1, 2, 3, 5, 6, 7], index_col=0,names=[ 'fullname', 'Abbrevation', 'depth in tree', 'structure_id_path','total_voxel_counts (10 um)'])
anofile = nmt.image('/home/penglab/Documents/data/annotation_10.nrrd')
bs = nmt.brain_structure('/home/penglab/Documents/data/Mouse.csv')
axonJaneliaccf = pd.read_excel('/home/penglab/Documents/dataSource/JaneliaAxonccf.xlsx', index_col=0)
SOMAJaneliaccf = pd.read_excel('/home/penglab/Documents/dataSource/JaneliaSomaccf.xlsx', index_col=0)
mouseDF = pd.read_excel('/home/penglab/Documents/dataSource/mouseDF.xlsx', index_col=0)
ccfDF = pd.read_excel('/home/penglab/Documents/dataSource/ccfDF.xlsx', index_col=0)
#%% Define function to transfer id to abbreviation
def id_to_name(region_id,bs):
    if region_id>0:
    # region_name can be either abbrevation (checked first) or description
        abbr=bs.level[bs.level.index ==region_id].Abbrevation.to_string()
    else:
        abbr='non'
    return abbr
     

#%%Define the comparing function
def JaneliaAnalysis(Feafile,ccfThre,norF,ctname,somalist,numC):
    import sklearn.cluster
    from sklearn.cluster import KMeans
    ccfDFsub = ccfDF[ccfDF['Child num']>ccfThre]
    #sortCcfDF = ccfDFsub .sort_values(['Child num'])
    del_list=ccfDFsub.index
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
            #the DF for analysis, storing the umap coordinates, soma region and plotting color
    anaDF = pd.DataFrame(index=Feafile.index,columns=['ux','uy','SOMA','plotc'])
    anaDF['SOMA']=somalist
    typeR, typeC = np.unique(somalist, return_counts = True)
     #%%DEfine the soma plotting color

    colorind = range(len(typeR))
    i=0
    for typeiter in typeR:
        inddex = anaDF[anaDF ['SOMA']== typeiter].index  
        for ii in inddex:
            anaDF.loc[ii,'plotc']=colorind[i]  
        i=i+1
        if i ==10:
            i=0

    if 'Soma' in Feafile.columns:
        #do not use soma region as feature
        del Feafile['Soma']
    #whether normalization or not
    if norF==1:
        Feafile[Feafile!=0]=np.log(Feafile[Feafile!=0])
    #%% Use umap to map data from high dimension to low dimension
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns    
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
        ax.scatter(speRow['ux'], speRow['uy'], c= sns.color_palette()[speRow['plotc'][0]], s=10) 
        #Now plotc stores the color to plot
    ax.grid(True)   
    fig.show()    
 
    
    plt.figure()
    import seaborn as sns    
    plt.scatter(embedding[:, 0], embedding[:, 1], s=10,c=sns.color_palette("Set3", 10));
    plt.axis([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 0.5, np.min(embedding[:, 1]) - 0.5,
          np.max(embedding[:, 1]) + 0.5])
    plt.title('PLot the whole dataset using UMAP(CCF version)');
    plt.show()


     #%%Show the result of Kmeans on dataset
    typeR, typeC = np.unique(anaDF['SOMA'], return_counts = True)
    n_clusters = numC
    # correct number of clusters
    estimator= KMeans(n_clusters, random_state=100)
    kmeans_labels =estimator.fit_predict(Feafile.values)
    #centerCLUST=estimator.cluster_centers_
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans_labels, s=10, cmap='rainbow');
    plt.axis([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 0.5, np.min(embedding[:, 1]) - 0.5,
              np.max(embedding[:, 1]) + 0.5])

    plt.title('Result of k-means on whole dataset indexed by CCF');
    plt.show()    
    inertia = estimator.inertia_
    print('The inertia will be ',inertia,', note that lower value will be better, 0 is optimized.')    

    anaDF['kmeans Result']=  kmeans_labels
    abbrlist=[]

    for iditer in anaDF['SOMA']:
        abbrlist.append(id_to_name(iditer,bs))
    anaDF['SOMA_abbr']=abbrlist
    ct = pd.crosstab(anaDF['SOMA_abbr'],  anaDF['kmeans Result'] )    
    print('*************************************************')
    from sklearn import metrics
    print('Estimated number of clusters: %d' % numC)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(anaDF['SOMA_abbr'],anaDF['kmeans Result']))
    print("Completeness: %0.3f" % metrics.completeness_score(anaDF['SOMA_abbr'],anaDF['kmeans Result']))
    print("V-measure: %0.3f" % metrics.v_measure_score(anaDF['SOMA_abbr'],anaDF['kmeans Result']))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(anaDF['SOMA_abbr'],anaDF['kmeans Result']))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(anaDF['SOMA_abbr'],anaDF['kmeans Result']))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(Feafile.values, anaDF['kmeans Result'], metric='sqeuclidean'))
    print('*************************************************')
#here I want to set the axis using the abbreviation
    ctindex=[]
    for i in range(len(ct.index)):
        if len(ct.index[i].split())<2:
            ctindex.append('non')
        else:
            ctindex.append(ct.index[i].split()[2])           
    ct.index=ctindex
    ctname="/home/penglab/Documents/confuMccf/"+ctname+".xlsx"
    ct.to_excel(ctname)

 

#%% Test the Jalinea Feature which is obtaind based on CCF index
if np.sum(axonJaneliaccf.index==SOMAJaneliaccf.index)==axonJaneliaccf.shape[0]:
    FeaJanelia=axonJaneliaccf.copy()
    FeaJanelia['Soma']=SOMAJaneliaccf['soma_Region']
#%%Visualize the distribution of swc files' soma regions
import seaborn as sns
import matplotlib.pyplot as plt
typeR, typeC = np.unique(FeaJanelia['Soma'], return_counts = True)
#transfer id to uniform distributed value
'''
v_range = range(len(typeR))
i=0
id2value= pd.DataFrame(columns=['idx'])
id2value['ID']=FeaJanelia['Soma']
'''
somaDIS=FeaJanelia.copy()
abbr_list=[]
for iditer in FeaJanelia['Soma']:
    abbr_list.append(id_to_name(iditer,bs))
    
somaDIS['idx']=somaDIS.index
somaDIS['abbr']=abbr_list
somaDIS=somaDIS.iloc[:,-2:]

for i in somaDIS.index:
    if len(somaDIS.loc[i]['abbr'].split())>1:
        somaDIS.loc[i]['abbr']=somaDIS.loc[i]['abbr'].split()[2]
    else:
        somaDIS.loc[i]['abbr']='non'

fig, ax = plt.subplots(figsize = (14, 10))
somaDIS['abbr'].value_counts().plot(ax=ax, kind='bar')
ax.set_title('Visualize the distribution of Janelia\' soma regions indexed by CCF',fontsize=15)



import jieba               #分词库
from wordcloud import WordCloud   #词云库
from PIL import Image
somaDIS_repeated = pd.concat([somaDIS]*300, ignore_index=True)
cut_text= jieba.cut(' '.join([str(i) for i in somaDIS_repeated['abbr']]  ))
result= "/".join(cut_text)#必须给个符号分隔开分词结果来形成字符串,否则不能绘制词云
#print(result)

#3、生成词云图，这里需要注意的是WordCloud默认不支持中文，所以这里需已下载好的中文字库
#无自定义背景图：需要指定生成词云图的像素大小，默认背景颜色为黑色,统一文字颜色：mode='RGBA'和colormap='pink'
image = Image.open('/home/penglab/Documents/picpic.bmp')
graph = np.array(image)

wc = WordCloud(background_color='white',width=800,height=600,max_font_size=500,mask=graph,min_font_size=10)#,mode='RGBA',colormap='pink')
wc.generate(result)
wc.to_file('/home/penglab/Documents/WC.png') #按照设置的像素宽高度保存绘制好的词云图，比下面程序显示更清晰

# 4、显示图片
plt.figure("词云图") #指定所绘图名称
plt.imshow(wc)       # 以图片的形式显示词云
plt.axis("off")      #关闭图像坐标系
plt.show()







#%% Delete the nonsoma rows
outRangelist=FeaJanelia[FeaJanelia['Soma']==-1].index
print('\n')
print('The following swc files\' soma goes out of range: ',outRangelist)
noSOMAlist=FeaJanelia[FeaJanelia['Soma']==-2].index
print('\n')
print('The following swc files have no annotated soma: ',noSOMAlist)

subFeaJanelia_ccf=FeaJanelia.copy()
subFeaJanelia_ccf.drop(outRangelist,inplace=True)
subFeaJanelia_ccf.drop(noSOMAlist,inplace=True)
 

#%%Visualize the soma region distribution after deleting 
import matplotlib.pyplot as plt
typeR, typeC = np.unique(subFeaJanelia_ccf['Soma'], return_counts = True)

somaDIS=subFeaJanelia_ccf.copy()
abbr_list=[]
for iditer in subFeaJanelia_ccf['Soma']:
    abbr_list.append(id_to_name(iditer,bs))
    
somaDIS['idx']=somaDIS.index
somaDIS['abbr']=abbr_list
somaDIS=somaDIS.iloc[:,-2:]

for i in somaDIS.index:
    if len(somaDIS.loc[i]['abbr'].split())>1:
        somaDIS.loc[i]['abbr']=somaDIS.loc[i]['abbr'].split()[2]
    else:
        somaDIS.loc[i]['abbr']='non'

fig, ax = plt.subplots(figsize = (14, 10))
somaDIS['abbr'].value_counts().plot(ax=ax, kind='bar')
ax.set_title('Visualize the subset of Janelia\' soma regions indexed by CCF',fontsize=15)



import jieba               #分词库
from wordcloud import WordCloud   #词云库

cut_text= jieba.cut(' '.join([str(i) for i in somaDIS['abbr']]  ))
result= "/".join(cut_text)#必须给个符号分隔开分词结果来形成字符串,否则不能绘制词云
#print(result)

#3、生成词云图，这里需要注意的是WordCloud默认不支持中文，所以这里需已下载好的中文字库
#无自定义背景图：需要指定生成词云图的像素大小，默认背景颜色为黑色,统一文字颜色：mode='RGBA'和colormap='pink'
wc = WordCloud(background_color='white',width=800,height=600,max_font_size=50,
               max_words=1000)#,min_font_size=10)#,mode='RGBA',colormap='pink')
wc.generate(result)
wc.to_file('/home/penglab/Documents/subWC.png') #按照设置的像素宽高度保存绘制好的词云图，比下面程序显示更清晰

# 4、显示图片
plt.figure("词云图") #指定所绘图名称
plt.imshow(wc)       # 以图片的形式显示词云
plt.axis("off")      #关闭图像坐标系
plt.show()



#%%----------------------------------Case1

# non-normalized (3) sum of 1327 brain regions as features
print('***********CASE1 for 1002 Features indexed by CCF***********')
ccfThre = np.max(ccfDF['Child num'])

JaneliaAnalysis(FeaJanelia.iloc[:,2*len(ccftable.index):],ccfThre,0,'case1',FeaJanelia['Soma'],20)

#%%----------------------------------Case2
#Use normalized (3) sum of 1327 brain regions as features
print('***********CASE2 for 1002 Features indexed by CCF***********')
ccfThre = np.max(ccfDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(ccftable.index):],ccfThre,1,'case2',FeaJanelia['Soma'],20)
#%%----------------------------------Case3
#Use normalized (1)(2) of 1327 regions as features
print('***********CASE3 for 1002 Features indexed by CCF***********')
ccfThre = np.max(ccfDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,0:2*len(ccftable.index)],ccfThre,1,'case3',FeaJanelia['Soma'],20)  
#%%----------------------------------Case4
#Use normalized (1)(2) of one-child regions as features
print('***********CASE4 for 1002 Features indexed by CCF***********')
ccfThre = 1
JaneliaAnalysis(FeaJanelia.iloc[:,0:2*len(ccftable.index)],ccfThre,1,'case4',FeaJanelia['Soma'],20)  
#%%----------------------------------Case5
#%Use normalized (1)(2) of less-than-median regions as features
print('***********CASE5 for 1002 Features indexed by CCF***********')
ccfThre =np.median(ccfDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,0:2*len(ccftable.index)],ccfThre,1,'case5',FeaJanelia['Soma'],20)  
#%%----------------------------------Case6
#Use normalized (3) of one-child regions as features
print('***********CASE6 for 1002 Features indexed by CCF***********')
ccfThre =1
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(ccftable.index):],ccfThre,1,'case6',FeaJanelia['Soma'],20)  
#%%----------------------------------Case7
#Use normalized (3) of less-than-median regions as features    
print('***********CASE7 for 1002 Features indexed by CCF***********')
ccfThre = np.median(ccfDF['Child num'])
JaneliaAnalysis(FeaJanelia.iloc[:,2*len(ccftable.index):],ccfThre,1,'case7',FeaJanelia['Soma'],20)  
#%%VIsualize the confusion matrix
import matplotlib.pyplot as plt
iterrow=0
for info in os.listdir('/home/penglab/Documents/confuMccf'):
    domain = os.path.abspath('/home/penglab/Documents/confuMccf') #Obtain the path of the file
    infofull = os.path.join(domain,info) #Obtain the thorough path       
    caseDF = pd.read_excel(infofull, index_col=0)
    plt.figure()
    #print('\n  Print the heatmap of confusion matrix')
    sns.heatmap(caseDF,cmap='YlGnBu')

 


 

 

 

