#!/usr/bin/env python
# coding: utf-8

# In[20]:





# path="gs://healthcarediabetes/RAW/diabetic_data.csv"
# df = pd.read_csv(path)
# df.replace('?',np.nan,inplace=True)
# df.head()

# #dropping columns with high NA percentage
# df.drop(['weight','medical_specialty','payer_code'],axis=1,inplace=True)
# # dropping columns related to IDs
# df.drop(['encounter_id','patient_nbr','admission_type_id',
#          'discharge_disposition_id','admission_source_id'],axis=1,inplace=True)
# #removing invalid/unknown entries for gender
# df=df[df['gender']!='Unknown/Invalid']
# # dropping rows with NAs.
# df.dropna(inplace=True)

# diag_cols = ['diag_1','diag_2','diag_3']
# for col in diag_cols:
#     df[col] = df[col].str.replace('E','-')
#     df[col] = df[col].str.replace('V','-')
#     condition = df[col].str.contains('250')
#     df.loc[condition,col] = '250'

# df[diag_cols] = df[diag_cols].astype(float)

# # diagnosis grouping
# for col in diag_cols:
#     df['temp']=np.nan
    
#     condition = df[col]==250
#     df.loc[condition,'temp']='Diabetes'
    
#     condition = (df[col]>=390) & (df[col]<=458) | (df[col]==785)
#     df.loc[condition,'temp']='Circulatory'
    
#     condition = (df[col]>=460) & (df[col]<=519) | (df[col]==786)
#     df.loc[condition,'temp']='Respiratory'
    
#     condition = (df[col]>=520) & (df[col]<=579) | (df[col]==787)
#     df.loc[condition,'temp']='Digestive'
    
#     condition = (df[col]>=580) & (df[col]<=629) | (df[col]==788)
#     df.loc[condition,'temp']='Genitourinary'
    
#     condition = (df[col]>=800) & (df[col]<=999)
#     df.loc[condition,'temp']='Injury'
    
#     condition = (df[col]>=710) & (df[col]<=739)
#     df.loc[condition,'temp']='Muscoloskeletal'
    
#     condition = (df[col]>=140) & (df[col]<=239)
#     df.loc[condition,'temp']='Neoplasms'
    
#     condition = df[col]==0
#     df.loc[condition,col]='?'
#     df['temp']=df['temp'].fillna('Others')
#     condition = df['temp']=='0'
#     df.loc[condition,'temp']=np.nan
#     df[col]=df['temp']
#     df.drop('temp',axis=1,inplace=True)

# df.dropna(inplace=True)

# df['age'] = df['age'].str[1:].str.split('-',expand=True)[0]
# df['age'] = df['age'].astype(int)
# max_glu_serum_dict = {'None':0,
#                       'Norm':100,
#                       '>200':200,
#                       '>300':300
#                      }
# df['max_glu_serum'] = df['max_glu_serum'].replace(max_glu_serum_dict)

# A1Cresult_dict = {'None':0,
#                   'Norm':5,
#                   '>7':7,
#                   '>8':8
#                  }
# df['A1Cresult'] = df['A1Cresult'].replace(A1Cresult_dict)

# change_dict = {'No':-1,
#                'Ch':1
#               }
# df['change'] = df['change'].replace(change_dict)

# diabetesMed_dict = {'No':-1,
#                     'Yes':1
#                    }
# df['diabetesMed'] = df['diabetesMed'].replace(diabetesMed_dict)

# d24_feature_dict = {'Up':10,
#                     'Down':-10,
#                     'Steady':0,
#                     'No':-20
#                    }
# d24_cols = ['metformin','repaglinide','nateglinide','chlorpropamide',
#  'glimepiride','acetohexamide','glipizide','glyburide',
#  'tolbutamide','pioglitazone','rosiglitazone','acarbose',
#  'miglitol','troglitazone','tolazamide','examide',
#  'citoglipton','insulin','glyburide-metformin','glipizide-metformin',
#  'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']
# for col in d24_cols:
#     df[col] = df[col].replace(d24_feature_dict)

# condition = df['readmitted']!='NO'
# df['readmitted'] = np.where(condition,1,0)

# df.head()



# cat_cols = list(df.select_dtypes('object').columns)
# class_dict = {}
# for col in cat_cols:
#     df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col])], axis=1)
    
# from datetime import datetime
# d=datetime.today()
# d.strftime('%m/%d/%y')
# df.head()

# if 'readmitted' not in df.columns:
#     df.to_csv("gs://healthcarediabetes/TESTED/"+str(d)+'/'+'test_data',index=False)
# else:
#     df.to_csv("gs://healthcarediabetes/PREPROCESSED/"+str(d)+'/'+'train_data',index=False)


# In[64]:


# Removing skewnewss and kurtosis using log transformation if it is above a threshold value -  2

path="gs://healthcarediabetes/RAW/diabetic_data.csv"
df = pd.read_csv(path)
df.replace('?',np.nan,inplace=True)
numerics = list(set(list(df._get_numeric_data().columns))- {'readmitted'})
num_col = list(set(list(df._get_numeric_data().columns))- {'readmitted'})
train_data=df.drop('readmitted',axis=1)

statdataframe = pd.DataFrame()
statdataframe['numeric_column'] = num_col
skew_before = []
skew_after = []

kurt_before = []
kurt_after = []

standard_deviation_before = []
standard_deviation_after = []

log_transform_needed = []

log_type = []

for i in num_col:
    skewval = df.loc[:,i].skew()
    skew_before.append(skewval)
    
    kurtval = df[i].kurtosis()
    kurt_before.append(kurtval)
    
    sdval = df[i].std()
    standard_deviation_before.append(sdval)
    print(skewval)
    print(i)
    if (abs(skewval) >2) & (abs(kurtval) >2):
        log_transform_needed.append('Yes')
        
        if len(df[df[i] == 0])/len(df) <=0.02:
            log_type.append('log')
            skewvalnew = np.log(pd.DataFrame(df[train_data[i] > 0])[i]).skew()
            skew_after.append(skewvalnew)
            
            kurtvalnew = np.log(pd.DataFrame(df[train_data[i] > 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)
            
            sdvalnew = np.log(pd.DataFrame(df[train_data[i] > 0])[i]).std()
            standard_deviation_after.append(sdvalnew)
            
        else:
            log_type.append('log1p')
            skewvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).skew()
            skew_after.append(skewvalnew)
        
            kurtvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)
            
            sdvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).std()
            standard_deviation_after.append(sdvalnew)
            
    else:
        log_type.append('NA')
        log_transform_needed.append('No')
        
        skew_after.append(skewval)
        kurt_after.append(kurtval)
        standard_deviation_after.append(sdval)

statdataframe['skew_before'] = skew_before
statdataframe['kurtosis_before'] = kurt_before
statdataframe['standard_deviation_before'] = standard_deviation_before
statdataframe['log_transform_needed'] = log_transform_needed
statdataframe['log_type'] = log_type
statdataframe['skew_after'] = skew_after
statdataframe['kurtosis_after'] = kurt_after
statdataframe['standard_deviation_after'] = standard_deviation_after


for i in range(len(statdataframe)):
    if statdataframe['log_transform_needed'][i] == 'Yes':
        colname = str(statdataframe['numeric_column'][i])
        
        if statdataframe['log_type'][i] == 'log':
            df = df[df[colname] > 0]
            df[colname + "_log"] = np.log(df[colname])
            
        elif statdataframe['log_type'][i] == 'log1p':
            df = df[df[colname] >= 0]
            df[colname + "_log1p"] = np.log1p(df[colname])
            
            
            
# Feature Scaling
datf = pd.DataFrame()
datf['features'] = numerics
datf['std_dev'] = datf['features'].apply(lambda x: df[x].std())
datf['mean'] = datf['features'].apply(lambda x: df[x].mean())

df2=df
def standardize(raw_data):
    return ((raw_data - np.mean(raw_data, axis = 0)) / np.std(raw_data, axis = 0))
df2[numerics] = standardize(df2[numerics])
import scipy as sp
df2 = df2[(np.abs(sp.stats.zscore(df2[numerics])) < 3).all(axis=1)]


#df2['level1_diag1'] = df2['level1_diag1'].astype('object')
df_pd = pd.get_dummies(df2, columns=['gender', 'admission_type_id', 'discharge_disposition_id','admission_source_id', 'max_glu_serum', 'A1Cresult'], drop_first = True)
just_dummies = pd.get_dummies(df_pd['race'])
df_pd = pd.concat([df_pd, just_dummies], axis=1)      
df_pd.drop(['race'], inplace=True, axis=1)

non_num_cols = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 
                'max_glu_serum', 'A1Cresult', 'level1_diag1' ]
num_cols = list(set(list(df._get_numeric_data().columns))- {'readmitted', 'change'})
num_cols
new_non_num_cols = []
for i in non_num_cols:
    for j in df_pd.columns:
        if i in j:
            new_non_num_cols.append(j)
l = []
for feature in list(df_pd.columns):
    if '|' in feature:
        l.append(feature)
df=df_pd










#dropping columns with high NA percentage
df.drop(['weight','medical_specialty','payer_code'],axis=1,inplace=True)
# dropping columns related to IDs
df.drop(['encounter_id','patient_nbr'],axis=1,inplace=True)
#removing invalid/unknown entries for gender
#df=df[df['gender']!='Unknown/Invalid']
# dropping rows with NAs.
df.dropna(inplace=True)

diag_cols = ['diag_1','diag_2','diag_3']
for col in diag_cols:
    df[col] = df[col].str.replace('E','-')
    df[col] = df[col].str.replace('V','-')
    condition = df[col].str.contains('250')
    df.loc[condition,col] = '250'

df[diag_cols] = df[diag_cols].astype(float)

# diagnosis grouping
for col in diag_cols:
    df['temp']=np.nan
    
    condition = df[col]==250
    df.loc[condition,'temp']='Diabetes'
    
    condition = (df[col]>=390) & (df[col]<=458) | (df[col]==785)
    df.loc[condition,'temp']='Circulatory'
    
    condition = (df[col]>=460) & (df[col]<=519) | (df[col]==786)
    df.loc[condition,'temp']='Respiratory'
    
    condition = (df[col]>=520) & (df[col]<=579) | (df[col]==787)
    df.loc[condition,'temp']='Digestive'
    
    condition = (df[col]>=580) & (df[col]<=629) | (df[col]==788)
    df.loc[condition,'temp']='Genitourinary'
    
    condition = (df[col]>=800) & (df[col]<=999)
    df.loc[condition,'temp']='Injury'
    
    condition = (df[col]>=710) & (df[col]<=739)
    df.loc[condition,'temp']='Muscoloskeletal'
    
    condition = (df[col]>=140) & (df[col]<=239)
    df.loc[condition,'temp']='Neoplasms'
    
    condition = df[col]==0
    df.loc[condition,col]='?'
    df['temp']=df['temp'].fillna('Others')
    condition = df['temp']=='0'
    df.loc[condition,'temp']=np.nan
    df[col]=df['temp']
    df.drop('temp',axis=1,inplace=True)

df.dropna(inplace=True)

df['age'] = df['age'].str[1:].str.split('-',expand=True)[0]
df['age'] = df['age'].astype(int)
max_glu_serum_dict = {'None':0,
                      'Norm':100,
                      '>200':200,
                      '>300':300
                     }
#df['max_glu_serum'] = df['max_glu_serum'].replace(max_glu_serum_dict)

A1Cresult_dict = {'None':0,
                  'Norm':5,
                  '>7':7,
                  '>8':8
                 }
#df['A1Cresult'] = df['A1Cresult'].replace(A1Cresult_dict)

change_dict = {'No':-1,
               'Ch':1
              }
df['change'] = df['change'].replace(change_dict)

diabetesMed_dict = {'No':-1,
                    'Yes':1
                   }
df['diabetesMed'] = df['diabetesMed'].replace(diabetesMed_dict)

d24_feature_dict = {'Up':10,
                    'Down':-10,
                    'Steady':0,
                    'No':-20
                   }
d24_cols = ['metformin','repaglinide','nateglinide','chlorpropamide',
 'glimepiride','acetohexamide','glipizide','glyburide',
 'tolbutamide','pioglitazone','rosiglitazone','acarbose',
 'miglitol','troglitazone','tolazamide','examide',
 'citoglipton','insulin','glyburide-metformin','glipizide-metformin',
 'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']
for col in d24_cols:
    df[col] = df[col].replace(d24_feature_dict)

condition = df['readmitted']!='NO'
df['readmitted'] = np.where(condition,1,0)

df.head()



cat_cols = list(df.select_dtypes('object').columns)
class_dict = {}
for col in cat_cols:
    df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col])], axis=1)


# In[65]:



from datetime import datetime
d=datetime.today()
d.strftime('%m/%d/%y')
df.head()

if 'readmitted' not in df.columns:
    df.to_csv("gs://healthcarediabetes/TESTED/"+str(d)+'/'+'test_data',index=False)
else:
    df.to_csv("gs://healthcarediabetes/PREPROCESSED/"+str(d)+'/'+'train_data',index=False)


# In[ ]:




