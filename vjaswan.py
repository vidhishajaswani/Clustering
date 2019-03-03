#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:48:22 2018

@author: vidhishajaswani
"""
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import mixture
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


# read the data
df=pd.read_csv("vjaswan.csv",header=None)
df.columns=['X1','X2','X3']
data=df.values


#------------- Task 1 -----------
print("------------- Task 1 -----------")

#find where to cut the dendogram
dend = shc.linkage(data, method='ward',metric='euclidean') 

max_diff=0
delta_star=0
curr_diff=0
for i in range(1,len(dend)):
    curr_diff = dend[i][2] - dend[i-1][2]
    if(curr_diff > max_diff):
        max_diff=curr_diff
        delta_star=(dend[i][2]+dend[i-1][2])/2
        
print('Number of clusters is 2 and Cut off point is at: ',delta_star)

#plot the dendogram
fig=shc.dendrogram(dend,color_threshold=delta_star)
plt.title("Dendogram for Hierarchial Clustering")
plt.axhline(y=delta_star)
plt.show()

#distance matrix
sns.clustermap(df,method='single')
plt.show()

 
#perform clustering
nClust=2
hc = AgglomerativeClustering(linkage="ward",n_clusters=nClust,affinity = 'euclidean')
hc.fit(df)

#3d plot for clustering
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:,0],data[:,1],data[:,2],c=hc.labels_, cmap='rainbow')  
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D plot for hierarchial clustering for n clusters: %d"  %nClust)
plt.show()



#------------- Task 2 -----------
print("------------- Task 2 -----------")
#find best k using sse and silhoutte score
sse=[]
k=[]
score=[]
for i in range(2, 15):
    model = KMeans(n_clusters = i)
    kmeans = model.fit_predict(df)
    sse.append(model.inertia_)
    k.append(i)
    score.append(silhouette_score(df, kmeans, metric='euclidean'))
    
#k versus sse    

plt.plot(k,sse)
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()
kn = KneeLocator(k,sse,curve='convex',direction='decreasing')
print("Knee at %d" %kn.knee)   

#k versus silhoutte score
plt.plot(k,score)
plt.xlabel('k')
plt.ylabel('Silhoutte Score')
plt.title('k versus Silhoutte Score')
plt.show()


#apply model for best k
myK=2
model = KMeans(n_clusters=myK)
kmeans = model.fit(df[['X1','X2','X3']])
centroids = kmeans.cluster_centers_
labels = kmeans.labels_


#make 3d plot

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2],c=labels.astype(np.float))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("K-Means for k: %d" %myK)
plt.show()


#------------- Task 3 -----------
print("#------------- Task 3 -----------")
#determine best epsilon
def epsilon(n):
    model = NearestNeighbors(n_neighbors=n).fit(data)
    distances, indices = model.kneighbors(data)
    dist = sorted(distances[:,n-1], reverse=True)
    plt.plot(indices[:,0], dist,label=n+1)
    plt.title("Elbow Plot to find best epsilon for different min points")
    plt.legend(loc="best")

#apply dbscan    
def clustering(df,epsilon,minPts):
    db = DBSCAN(eps=epsilon,  min_samples=minPts).fit(df)
    labels = db.labels_
    n_clusters_ = len(set(labels)) 
    fig = plt.figure()
    ax = Axes3D(fig)

    for l in np.unique(labels):

        ax.scatter(df.iloc[labels == l, 0], df.iloc[labels == l, 1], df.iloc[labels == l, 2],
                   s=20,label=minPts)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Scatter Plot for DBSCAN for epsilon %d and MinPts %d with number of clusters %d" %(epsilon, minPts,n_clusters_))
    plt.show()
#test for different values of minPts  
for i in range(3,8):
    epsilon(i-1)
plt.show()
   
clustering(df,65,3) #gives 9 clusters
clustering(df,70,3) #gives 7 clusters
clustering(df,90,4)
clustering(df,100,5)
clustering(df,120,6)
clustering(df,120,7)




#------------- Extra Credit -----------
print("#------------- Extra Credit -----------")
#likelihood versus k plot
bicVals=[]
aicVals=[]
likelihood=[]
n=[]
for i in range(1,10):
    model=mixture.GaussianMixture(n_components=i)
    train=model.fit(df)
    n.append(i)
    bic=model.bic(df)
    bicVals.append(model.bic(df))
    aicVals.append(model.aic(df))
    likelihood.append((2.9*i)-(bic/2))
    


plt.plot(n,likelihood)
plt.xlabel("number of clusters")
plt.title("Likelihood versus number of clusters")
plt.show()


bestN=n[likelihood.index(max(likelihood))]
print('Best k for minimum BIC is : %d' %bestN)


gmm=mixture.GaussianMixture(n_components=6)
train=gmm.fit(data)
labels=gmm.predict(data)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2],c=labels.astype(np.float))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("GMM for N: %d" %6)
plt.show()




