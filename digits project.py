# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:32:57 2020

@author: SOUPTIK
"""
####PCA BASED CLUSTERIN ON MFEAT-MO
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("mfeat-morphological_csv.csv")
df1=df.iloc[:,[0,1,2,3,4,5]]


scaler=StandardScaler()
seg=scaler.fit_transform(df1)


pca=PCA(n_components=3)
s=pca.transform(seg)

####applying K means clustering

wcss=[]
for i in range(1,20):
    kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans_pca.fit(s)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10,8))
plt.plot(range(1,20),wcss,marker='o',linestyle='--')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('K-Means with PCA clustering')
plt.show()

####number of clusters ===6
kmeans_pca=KMeans(n_clusters=6,init='k-means++',random_state=42)
kmeans_pca.fit(s)
df_segm=pd.concat([df1.reset_index(drop = True),pd.DataFrame(s)],axis = 1)
df_segm.columns.values[-3:]=["COMPONENT 1","COMPONENT 2","COMPONENT 3"]
df_segm['Segmwnt K-means PCA']=kmeans_pca.labels_
df_segm['Segment']=df_segm['Segmwnt K-means PCA'].map({0:'first',1:'second',2:'third',3:'fourth',4:'fifth',5:'sixth'})
x=df_segm['COMPONENT 2']
y=df_segm['COMPONENT 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x, y, hue = df_segm['Segment'], palette=['g','r','c','m','#95a5a6','#3498db'])
plt.title("CLUSTERS BY PCA COMPONENTS")
plt.show()
sns.scatterplot

######SPECTRAL CLUSTERING ON MFEAT-MO
from sklearn.cluster import SpectralClustering
spectral_model_rbf=SpectralClustering(n_clusters=6,affinity='rbf')
labels_rbf=spectral_model_rbf.fit_predict(s)
colours={}
colours[0]='b'
colours[1]='y'
colours[2]='g'
colours[3]='c'
colours[4]='r'
colours[5]='m'
cvec=[colours[label]for label in labels_rbf]
b=plt.scatter(s[:,0],s[:,1],color='b')
y=plt.scatter(s[:,0],s[:,1],color='y')
g=plt.scatter(s[:,0],s[:,1],color='g')
c=plt.scatter(s[:,0],s[:,1],color='c')
r=plt.scatter(s[:,0],s[:,1],color='r')
m=plt.scatter(s[:,0],s[:,1],color='m')
plt.figure(figsize=(9,9))
plt.scatter(s[:,0],s[:,1], c=cvec)
plt.legend((b,y,g,c,r,m),('Label 0','Label 1','Label 2','Label 3','Label 4','Label 5'))
plt.show()

####PCA BASED CLUSTERIN ON MFEAT-ZER
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("mfeat-zer.csv")
df1=df.iloc[:,0:47]


scaler=StandardScaler()
seg=scaler.fit_transform(df1)


pca=PCA(n_components=5)
s=pca.fit_transform(seg)

####applying K means clustering

wcss=[]
for i in range(1,20):
    kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans_pca.fit(s)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10,8))
plt.plot(range(1,20),wcss,marker='o',linestyle='--')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('K-Means with PCA clustering')
plt.show()

####number of clusters === 8
kmeans_pca=KMeans(n_clusters=6,init='k-means++',random_state=42)
kmeans_pca.fit(s)
df_segm=pd.concat([df1.reset_index(drop = True),pd.DataFrame(s)],axis = 1)
df_segm.columns.values[-3:]=["COMPONENT 1","COMPONENT 2","COMPONENT 3"]
df_segm['Segmwnt K-means PCA']=kmeans_pca.labels_
df_segm['Segment']=df_segm['Segmwnt K-means PCA'].map({0:'first',1:'second',2:'third',3:'fourth',4:'fifth',5:'sixth'})
x=df_segm['COMPONENT 2']
y=df_segm['COMPONENT 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x, y, hue = df_segm['Segment'], palette=['g','r','c','m','#95a5a6','#3498db'])
plt.title("CLUSTERS BY PCA COMPONENTS")
plt.show()
sns.scatterplot

######SPECTRAL CLUSTERING ON MFEAT-ZER
from sklearn.cluster import SpectralClustering
spectral_model_rbf=SpectralClustering(n_clusters=6,affinity='rbf')
labels_rbf=spectral_model_rbf.fit_predict(s)
colours={}
colours[0]='b'
colours[1]='y'
colours[2]='g'
colours[3]='c'
colours[4]='r'
colours[5]='m'
cvec=[colours[label]for label in labels_rbf]
b=plt.scatter(s[:,0],s[:,1],color='b')
y=plt.scatter(s[:,0],s[:,1],color='y')
g=plt.scatter(s[:,0],s[:,1],color='g')
c=plt.scatter(s[:,0],s[:,1],color='c')
r=plt.scatter(s[:,0],s[:,1],color='r')
m=plt.scatter(s[:,0],s[:,1],color='m')
plt.figure(figsize=(9,9))
plt.scatter(s[:,0],s[:,1], c=cvec)
plt.legend((b,y,g,c,r,m),('Label 0','Label 1','Label 2','Label 3','Label 4','Label 5'))
plt.show()


##### MFEAT-PIX
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("mfeat-pixel_csv.csv")
df1=df.iloc[:,0:241]


scaler=StandardScaler()
seg=scaler.fit_transform(df1)



pca=PCA(n_components=5)
s=pca.fit_transform(seg)

####applying K means clustering

wcss=[]
for i in range(1,20):
    kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans_pca.fit(s)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10,8))
plt.plot(range(1,20),wcss,marker='o',linestyle='--')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('K-Means with PCA clustering')
plt.show()

####number of clusters === 7
kmeans_pca=KMeans(n_clusters=6,init='k-means++',random_state=42)
kmeans_pca.fit(s)
df_segm=pd.concat([df1.reset_index(drop = True),pd.DataFrame(s)],axis = 1)
df_segm.columns.values[-3:]=["COMPONENT 1","COMPONENT 2","COMPONENT 3"]
df_segm['Segmwnt K-means PCA']=kmeans_pca.labels_
df_segm['Segment']=df_segm['Segmwnt K-means PCA'].map({0:'first',1:'second',2:'third',3:'fourth',4:'fifth',5:'sixth'})
x=df_segm['COMPONENT 2']
y=df_segm['COMPONENT 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x, y, hue = df_segm['Segment'], palette=['g','r','c','m','#95a5a6','#3498db'])
plt.title("CLUSTERS BY PCA COMPONENTS")
plt.show()
sns.scatterplot

######SPECTRAL CLUSTERING ON MFEAT-PIX
from sklearn.cluster import SpectralClustering
spectral_model_rbf=SpectralClustering(n_clusters=6,affinity='rbf')
labels_rbf=spectral_model_rbf.fit_predict(s)
colours={}
colours[0]='b'
colours[1]='y'
colours[2]='g'
colours[3]='c'
colours[4]='r'
colours[5]='m'
cvec=[colours[label]for label in labels_rbf]
b=plt.scatter(s[:,0],s[:,1],color='b')
y=plt.scatter(s[:,0],s[:,1],color='y')
g=plt.scatter(s[:,0],s[:,1],color='g')
c=plt.scatter(s[:,0],s[:,1],color='c')
r=plt.scatter(s[:,0],s[:,1],color='r')
m=plt.scatter(s[:,0],s[:,1],color='m')
plt.figure(figsize=(9,9))
plt.scatter(s[:,0],s[:,1], c=cvec)
plt.legend((b,y,g,c,r,m),('Label 0','Label 1','Label 2','Label 3','Label 4','Label 5'))
plt.show()


