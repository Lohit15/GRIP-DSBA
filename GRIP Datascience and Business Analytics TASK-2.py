#!/usr/bin/env python
# coding: utf-8

# DATASCIENCE AND BUSINESS ANALYTICS
# Batch:- FEB2021
# TASK:- #2
# PREDICTION USING UNSUPERVISED ML
# From the given 'Iris' dataset,predict the optimum number of clusters and represent it visually
# Author:- Akkineni Lohit

# # IMPORTING LIBRARIES

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline 
get_ipython().run_line_magic('matplotlib', 'inline')


# # IMPORTING AND READING DATASET

# In[15]:


dataset=pd.read_csv('Iris.csv')
print('Import Successful')


# # DATA PROCESSING

# In[16]:


dataset.head()


# In[17]:


dataset.shape


# In[18]:


dataset.describe()


# In[20]:


dataset.info()


# In[21]:


dataset.corr()


# In[19]:


dataset.isnull(). sum()


# # OUTLINER CHECK

# In[28]:


def outdet(dataset):
    r = []
    for col in dataset.columns:
        for i in dataset.index:
            if dataset.loc[i, col]=='NULL' or dataset.loc[i, col] == np.nan:
                r.append(i)
    dataset = dataset.drop(list(set(r)))
    dataset = dataset.reset_index()
    dataset = dataset.drop('index', axis=1)
    
    num_cols = []
    for col in dataset.columns:
        if dataset[col].dtype == 'object' :
            try:
                df[col] = pd.to_numeric(dataset[col])
                num_cols.append(col)
            except ValueError:
                pass
            
    count = 0
    t = []
    for i in num_cols:
        z = np.abs(stats.zscore(dataset[i]))
        for j in range(len(z)):
            if z[j]>3 or z[j]<-3:
                t.append(j)
                count+=1
    dataset = dataset.drop(list(set(t)))
    dataset = dataset.reset_index()
    dataset = dataset.drop('index', axis=1)
    print(count)
    return dataset


# In[29]:


dataset = outdet(dataset)


# # DATA VISUALIZATION

# In[30]:


sns.countplot(dataset['Species'])


# In[31]:


sns.pairplot(data=dataset)


# In[33]:


sns.catplot("Species","PetalLengthCm", data =  dataset)


# # GRAPH TO FIND K

# In[34]:


x = dataset.iloc[:, [1,2,3]].values
inertias = []

for i in range(1, 8):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(x)
    inertias.append(kmeans.inertia_)
    
plt.plot(range(1, 8), inertias)
plt.title("Elbow Graph")
plt.xlabel('number of clusters')
plt.ylabel('inertia')
plt.show()


# In[35]:


x


# In[36]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
               max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[37]:


labels = kmeans.predict(x)
labels


# In[38]:


plt.scatter(x[labels == 0, 0], x[labels == 0, 1],
           s = 100, c = 'green', label = 'Iris setosa')
plt.scatter(x[labels == 1, 0], x[labels == 1, 1],
           s = 100, c = 'yellow', label = 'Iris versicolour')
plt.scatter(x[labels == 2, 0], x[labels == 2, 1],
           s = 100, c = 'red', label = 'Iris virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],
           s = 100, c = 'blue', label = 'Centroids')
plt.legend()


# In[42]:


variety = ['Iris setosa', 'Iris versicolour', 'Iris virginica'] 
varety = []
for i in labels:
    varety.append(variety[i])


# In[43]:


varety


# In[45]:


dataset['predicted_varities'] = varety
sns.countplot(dataset['predicted_varities'])


# In[ ]:




