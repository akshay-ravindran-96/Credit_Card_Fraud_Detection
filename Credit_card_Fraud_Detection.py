#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy


# In[2]:


import pandas


# In[3]:


import sys


# In[1]:


import pandas


# In[2]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy 
import sklearn

print("Python:{}".format(sys.version))
print("numpy:{}".format(numpy.__version__))
print("pandas:{}".format(pandas.__version__))
print("matplotlib:{}".format(matplotlib.__version__))
print("seaborn:{}".format(seaborn.__version__))
print("scipy:{}".format(scipy.__version__))
print("sklearn:{}".format(skelarn.__version__))
print("sklearn:{}".format(sklearn.__version__))



#import the necessary packagges
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# loading the dataset
data =pd.read_csv('creditcard.csv')


# exploring the dataset a little bit
print(data.columns)
print(data.colums[0])
print(data.shape)
print(data.describe)
print(data.describe())


data = data.sample(frac = 0.1, random_state=1)

print(data.shape)


# plot histogram
data.hist(figsize = (20,20))
plt.show()


# Determin number of fraud cases

fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]
outlier_fraction = len(fraud)/float(len(Valid))


print(outlier_fraction)
print('Fraud Cases : {}'.format(len(fraud)))
print('Valid Cases: {}'.format(len(Valid)))


#corelation Matrix
corr_mat = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corr_mat, vmax=.8, square = True)

#get all columns
columns=  data.columns.tolist()

#filter the Columns to remove data that we do not want
columns = [c for c in columns if c not in ["Class"]]

target ="Class"

X= data[columns]
Y= data[target]

print(X.shape)
print(Y.shape)


# In[29]:


from sklearn.metrics import Classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state

state =1

# define outlier detection methods

classifiers ={
    "IsolationForest" : IsolationForest(max_samples = len(X), contamination = outlier_fraction, random_state = state),
    "LocalOutlierFactor": LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
}


# In[30]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state

state =1

# define outlier detection methods

classifiers ={
    "IsolationForest" : IsolationForest(max_samples = len(X), contamination = outlier_fraction, random_state = state),
    "LocalOutlierFactor": LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
}


# In[31]:


# Fit the model
n_oultiers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit data and tag outliers
    if(clf_name == "LocalOutlierFactor"):
        y_pred =clf.fit_predict(X)
        scores_pred =clf.negative_outlier_factor_
    else:
        clf.fit(x)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
        
    y_pred[y_pred == 1] = 0
    y_pred[y_pred== -1] = 1
    
    
    n_errors = (y_pred != Y).sum()
    
    # Class_Matrix:
    
    print("{} :{}".format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))
    


# In[32]:


# Fit the model
n_oultiers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit data and tag outliers
    if(clf_name == "LocalOutlierFactor"):
        y_pred =clf.fit_predict(X)
        scores_pred =clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
        
    y_pred[y_pred == 1] = 0
    y_pred[y_pred== -1] = 1
    
    
    n_errors = (y_pred != Y).sum()
    
    # Class_Matrix:
    
    print("{} :{}".format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


# In[ ]:




