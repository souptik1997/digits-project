# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:38:56 2020

@author: HP PC
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.linalg as la
import plotly.plotly as py
import chart_studio.plotly as py

df=pd.read_csv("mfeat-pixel_csv.csv")
df1=df.iloc[:,0:241]


scaler=StandardScaler()
seg=scaler.fit_transform(df1)


####Finding Covariance matrix
matr=pd.DataFrame.to_numpy(df1)
cov_matr=np.cov(matr.T)


###FINDING EIGEN VALUES AND VECTORS
values, vectors = la.eig(cov_matr)
eigvals=values.real

u,s,v=np.linalg.svd(seg.T)
u

for ev in vectors:
    np.testing.assert_array_almost_equal(1.0,np.linalg.norm(ev))
    
    
eig_pairs=[(np.abs(values[i]),vectors[:,i])for i in range(len(values))]
eig_pairs.sort()
eig_pairs.reverse()    
    
for i in eig_pairs:
    print(i[0])
    
tot=sum(values)
var_exp=[(i/tot)*100 for i in sorted(values,reverse=True)]
cum_var_exp=np.cumsum(var_exp)
trace1=dict(type='bar',x=['PC %s' %i for i in range(1,5)],y=var_exp,name='Individual')
trace2=dict(type='scatter',x=['PC %s' %i for i in range(1,5)],y=cum_var_exp,name='Cumulative')
data=[trace1,trace2]
layout=dict(title='EXPLAINED VARIANCE BY DIFFERENT PRINCIPAL COMPONENTS',yaxis=dict(title='EXPLAINED VARIANCE IN PERCENT'),annotations==list([dict(x=1.16,y=1.05,xref='paper',yref='paper',text='explained variance',showarrow=False)]))
fig=dict(data=data)
py.iplot(fig)
