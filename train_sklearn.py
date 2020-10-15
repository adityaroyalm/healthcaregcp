#!/usr/bin/env python
# coding: utf-8

# In[159]:


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

import pandas as pd

import re


from google.cloud import storage

# Instantiates a client
storage_client = storage.Client()

# Get GCS bucket
bucket = storage_client.get_bucket('healthcarediabetes')

# Get blobs in bucket (including all subdirectories)
blobs_all = list(bucket.list_blobs())

import os
import glob

files = glob.glob('models/*')
for f in files:
    os.remove(f)
    
    
    

pre=[x for x in blobs_all if str(x).startswith('<Blob: healthcarediabetes, PREPROCESSED/')]
dates=[re.findall('<Blob: healthcarediabetes, PREPROCESSED/(.*)/train_data',str(x)) for x in pre]
notnull_Dates=[]
for x in dates:
    for y in x:
        if len(x)>0:
            notnull_Dates.append(y)
d=max(notnull_Dates)





import joblib
import os
import glob

def copy_local_directory_to_gcs(local_path, bucket, gcs_path):
    """Recursively copy a directory of files to GCS.

    local_path should be a directory and not have a trailing slash.
    """
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            continue
        remote_path = os.path.join(gcs_path, local_file[1 + len(local_path) :])
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)
bucket = storage_client.get_bucket('healthcarediabetes')
  


                
                

path="gs://healthcarediabetes/PREPROCESSED/"+str(d)+'/'+'train_data/'
df = pd.read_csv(path)
Xs = df.drop('readmitted',axis=1)
y = df['readmitted']
X_train,X_test,y_train,y_test = train_test_split(Xs,y,test_size=0.20,random_state=0)
X_train.shape,X_test.shape

ML_models = {}
model_index = ['LR','RF','NN']
model_sklearn = [LogisticRegression(solver='liblinear',random_state=0),
                 RandomForestClassifier(n_estimators=100,random_state=0),
                 MLPClassifier([100]*5,early_stopping=True,learning_rate='adaptive',random_state=0)]
model_summary = []
for name,model in zip(model_index,model_sklearn):
    ML_models[name] = model.fit(X_train,y_train)
    preds = model.predict(X_test)
    model_summary.append([name,f1_score(y_test,preds,average='weighted'),accuracy_score(y_test,preds),
                          roc_auc_score(y_test,model.predict_proba(X_test)[:,1])])
print(ML_models)

model_summary = pd.DataFrame(model_summary,columns=['Name','F1_score','Accuracy','AUC_ROC'])
model_summary = model_summary.reset_index()
display(model_summary)
from datetime import datetime
d=datetime.today()
d.strftime('%m/%d/%y')



for x in ML_models.keys():
    joblib.dump(ML_models[x],'models/'+str(x)+'_model.joblib') 


copy_local_directory_to_gcs('models',bucket,'TRAINED/'+str(d))   


# In[ ]:




