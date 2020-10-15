#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd

# data visuzlization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

# warnings
import warnings
warnings.filterwarnings('ignore')

# modeling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score

import joblib
import os
import glob

files = glob.glob('downloaded_models/*')
for f in files:
    os.remove(f)



from google.cloud import storage
import re
# Instantiates a client
storage_client = storage.Client()

# Get GCS bucket
bucket = storage_client.get_bucket('healthcarediabetes')

# Get blobs in bucket (including all subdirectories)
blobs_all = list(bucket.list_blobs())

pre=[x for x in blobs_all if str(x).startswith('<Blob: healthcarediabetes, TRAINED/')]
dates=[re.findall('<Blob: healthcarediabetes, TRAINED/(.*)/.',str(x)) for x in pre]
notnull_Dates=[]
for x in dates:
    for y in x:
        if len(x)>0:
            notnull_Dates.append(y)
dmodel=max(notnull_Dates)


pre=[x for x in blobs_all if str(x).startswith('<Blob: healthcarediabetes, TESTED/')]
dates=[re.findall('<Blob: healthcarediabetes, TESTED/(.*)/.',str(x)) for x in pre]
notnull_Dates=[]
i=0
for x in dates:
    for y in x:
        i=i+1
        if len(x)>0:
            notnull_Dates.append(y)
ddata=max(notnull_Dates)


path="gs://healthcarediabetes/TESTED/"+str(ddata)+'/'+'test_data'
df = pd.read_csv(path)


pre=[x for x in blobs_all if str(x).startswith('<Blob: healthcarediabetes, TRAINED/')]
dates=[re.findall('<Blob: healthcarediabetes, TRAINED/'+str(dmodel)+'/(.*)',str(x)) for x in pre]
notnull_Dates=[]
i=0
MODELS=[]
for x in dates:
    for y in x:
        i=i+1
        if len(x)>0:
            
            MODELS.append(y.split(',')[0])
            
            
if 'readmitted' in df.columns:
    df=df.drop('readmitted',1)
    df=df.iloc[:,1:]

for x in MODELS:
    model_path=r"gs://healthcarediabetes/TRAINED/"+str(dmodel).replace(' ','\ ')+'/'+str(x)
    get_ipython().system('gsutil -m cp -r  {model_path}  "downloaded_models"')
    model=joblib.load('downloaded_models/'+x)
    
    preds = model.predict(df)


# In[ ]:


pre=[x for x in blobs_all if str(x).startswith('<Blob: healthcarediabetes, TRAINED/')]
pre
dates=[re.findall('<Blob: healthcarediabetes, TRAINED/'+str(dmodel)+'/(.*)',str(x)) for x in pre]


# In[96]:


df


# In[ ]:




