#!/usr/bin/env python
# coding: utf-8

# DataPreprocessing

# Importing necessary libraries

# In[ ]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='A6Zk5A0Thzm9eU8uO0vmrK21s3fO8GntV3E4gKVQCTNc',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'rainfall-donotdelete-pr-plcbbgusacmaho'
object_key = 'rainfall in india 1901-2015.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)
df.head()


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings


# Import the dataset

# In[ ]:


rain_df= pd.read_csv('/content/rainfall _in _india _1901-2015.csv')
rain_df.head()


# Analyse the data and handling the missing values

# In[ ]:


rain_df=rain_df[["SUBDIVISION","YEAR","JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]]
rain_df.head()


# In[ ]:


rain_df.count()


# In[ ]:


rain_df.info()


# In[ ]:


rain_df.isnull().sum()


# In[ ]:


rain_df=rain_df.fillna(rain_df.mean())


# In[ ]:


rain_df.isnull().sum()


# In[ ]:


rain_df.rename(columns={"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}, inplace = True)


# In[ ]:


rain_df.head()


# In[ ]:


df1=pd.read_csv('/content/District_Rainfall.csv')


# In[ ]:


df1.isnull().any()


# In[ ]:


df1.isnull().any()


# In[ ]:


df1.describe()


# In[ ]:


df1.head()


# In[ ]:


df1.DISTRICT.value_counts()


# In[ ]:


df1.isnull().any()


# Data visualisation ,dependent & independent,train & test

# In[ ]:


rain_df.hist(figsize=(24,24));


# In[ ]:


rain_df[['YEAR','Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("YEAR").sum().plot(figsize=(13,8));


# In[ ]:


rain_df[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").mean().plot.barh(stacked=True,figsize=(13,8));


# In[ ]:


rain_df[['SUBDIVISION', 'Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("SUBDIVISION").sum().plot.barh(stacked=True,figsize=(16,8));


# In[ ]:


plt.figure(figsize=(11,4))
sns.heatmap(rain_df[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].corr(),annot=True)
plt.show()


# In[ ]:


mintemp_df=pd.read_csv('/content/temp1.csv')
mintemp_df.head()


# In[ ]:


mintemp_df.count()


# In[ ]:


mintemp_df.isnull().sum()


# In[ ]:


mintemp_df.rename(columns={"Year":"YEAR","Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}, inplace = True)
mintemp_df.head()


# In[ ]:


windspeed_df=pd.read_csv('/content/wind.csv')
windspeed_df.head()


# In[ ]:


windspeed_df.count()


# In[ ]:


windspeed_df.isnull().sum()


# In[ ]:


windspeed_df=windspeed_df.drop(['ANN'], axis=1)
windspeed_df.rename(columns={"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}, inplace = True)
windspeed_df.head()


# In[ ]:


sub=rain_df["SUBDIVISION"].unique()


# In[ ]:


year=pd.Series(np.arange(1981,2016))
year


# In[ ]:


a=pd.DataFrame({"SUBDIVISION":[],"YEAR":[],"MONTH":[],"MAX_TEMP":[],"MIN_TEMP":[],"MEAN_TEMP":[],"PRECEPTIONS":[],"PRESSURE":[],"WIND_SPEED":[],"RAINFALL":[],})
a


# In[ ]:


t=pd.DataFrame({"RAINFALL":[]})


# In[ ]:


for i in range(0,len(sub)):
    for j in range(0,len(year)):
        month=rain_df[rain_df["YEAR"]==year[j]]
        month=month[month["SUBDIVISION"]==sub[i]]
        t["RAINFALL"]=month[[1,2,3,4,5,6,7,8,9,10,11,12]].T
        #t=t.rename(columns={j:"RAINFALL"})
        month=month[["SUBDIVISION","YEAR"]]
#        month=maxtemp_df[maxtemp_df["YEAR"]==year[j]]
 #       t["MAX_TEMP"]=month[[1,2,3,4,5,6,7,8,9,10,11,12]].T
        month=mintemp_df[mintemp_df["YEAR"]==year[j]]
        t["MIN_TEMP"]=month[[1,2,3,4,5,6,7,8,9,10,11,12]].T
       # month=meantemp_df[meantemp_df["YEAR"]==year[j]]
        #t["MEAN_TEMP"]=month[[1,2,3,4,5,6,7,8,9,10,11,12]].T
        #month=preceptions_df[preceptions_df["YEAR"]==year[j]]
        #t["PRECEPTIONS"]=month[[1,2,3,4,5,6,7,8,9,10,11,12]].T
        #month=pressure_df[pressure_df["YEAR"]==year[j]]
        #t["PRESSURE"]=month[[1,2,3,4,5,6,7,8,9,10,11,12]].T
        #month=windspeed_df[windspeed_df["YEAR"]==year[j]]
        t["WIND_SPEED"]=month[[1,2,3,4,5,6,7,8,9,10,11,12]].T
        t["SUBDIVISION"]=sub[i]
        t["YEAR"]=year[j]
        t["MONTH"]=[1,2,3,4,5,6,7,8,9,10,11,12]
        a=a.append(t)
a.head(20)


# In[ ]:


a.count()


# In[ ]:


a.dropna()


# In[ ]:


a.dtypes


# In[ ]:


a=a.astype({"MONTH": np.int64,"YEAR":np.int64})


# In[ ]:


a.tail()


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


# In[2]:


rain_df=pd.read_csv('/content/dataset_rainfall.csv')
rain_df.head()


# In[3]:


rain_df.tail()


# In[ ]:


sns.displot(rain_df["MAX_TEMP"])


# In[ ]:


sns.displot(rain_df["RAINFALL"])


# In[ ]:


sns.displot(rain_df["MEAN_TEMP"])


# In[ ]:


sns.barplot(x=rain_df.MONTH,y=rain_df.RAINFALL)


# In[ ]:


sns.barplot(x=rain_df.MONTH,y=rain_df.WIND_SPEED)


# In[ ]:


sns.pointplot(x=rain_df.MONTH,y=rain_df.RAINFALL)


# In[ ]:


sns.pairplot(data=rain_df)


# In[ ]:


rain_df.info()


# In[ ]:


rain_df.SUBDIVISION.unique()


# In[ ]:


rain_df.describe()


# In[ ]:


rain_df.corr()


# In[ ]:


rain_df.isnull().sum()


# In[ ]:


s=rain_df.corr()
s


# In[ ]:


sns.heatmap(s,annot=True)


# In[ ]:


rain_df.info()


# In[ ]:


rain_df.head()


# In[4]:


from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()
rain_df.SUBDIVISION = lab.fit_transform(rain_df.SUBDIVISION)

rain_df.head()


# In[5]:


rain_df.SUBDIVISION.unique()


# In[6]:


feature=rain_df[["SUBDIVISION","MONTH","MAX_TEMP","MIN_TEMP","MEAN_TEMP","PRECEPTIONS","PRESSURE","WIND_SPEED"]]
target=rain_df["RAINFALL"]


# In[7]:


acc=[]
model=[]


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(feature,target,test_size=0.2,random_state=2)


# In[9]:


X_train


# In[10]:


y_train


# In[11]:


X_test


# In[12]:


y_test


# In[13]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score,classification_report,mean_squared_error,r2_score


# In[14]:


# create a regressor object
dtregressor = DecisionTreeRegressor(random_state = 0)

# fit the regressor with X and Y data
dtregressor.fit(X_train, y_train)


# In[15]:


# predicting with regression model with X and Y
y_train_pred=dtregressor.predict(X_train)
y_test_pred=dtregressor.predict(X_test)


# In[16]:


#Mean Squared Error and r2 Score
print("MSE",mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred))
print((r2_score(y_train,y_train_pred),(r2_score(y_test_pred,y_test))))


# In[17]:


#Accuracy Score
model.append('Decision Tree')
acc.append(dtregressor.score(X_test,y_test))
print(dtregressor.score(X_test,y_test))


# In[18]:


from xgboost import XGBRegressor


# In[23]:


# create a regressor object
xgb = XGBRegressor()

# fit the regressor with X and Y data
xgb.fit(X_train,y_train)


# In[24]:


# predicting with regression model with X and Y
y_train_pred=xgb.predict(X_train)
y_test_pred=xgb.predict(X_test)


# In[25]:


#Mean Squared Error and r2 Score
print("MSE",mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred))
print((r2_score(y_train,y_train_pred),(r2_score(y_test_pred,y_test))))


# In[26]:


#Accuracy Score
model.append('XGB Boost')
acc.append(xgb.score(X_test,y_test))
print(xgb.score(X_test,y_test))


# IBM Deployment

# In[1]:


get_ipython().system('pip install -U ibm-watson-machine-learning')


# In[2]:


from ibm_watson_machine_learning import APIClient
import json


# Authentication and set space

# In[11]:


wml_credentials={
    "apikey":"wCTxKj3uU-9T1gYEQQAHQVdQm9ci3a7fRqS806X1p2ZQ",
    "url":"https://us-south.ml.cloud.ibm.com"
    
}


# In[12]:


wml_client = APIClient(wml_credentials)


# In[14]:


wml_client.spaces.list()


# In[15]:


SPACE_ID="1a9590f9-fd89-4ae8-b226-1796437c07a4"


# In[16]:


wml_client.set.default_space(SPACE_ID)


# In[17]:


wml_client.software_specifications.list(500)


# Save and deploy the model

# In[19]:


import sklearn
sklearn.__version__


# In[ ]:


MODEL_NAME = 'predict'
DEPLOYMENT_NAME = 'rainfall'
DEMO_MODEL = model


# In[ ]:


#set python version
software_spec_uid=wml.client.software_specifications.get_id_by_name('runtime-22.1-py3.9')


# In[ ]:


#setup model meta
model_props={
    wml_client.repository.ModelMetaNames.NAME:MODEL_NAME.
    wml_client.repository.ModelMetaNames.TYPE:'scikit-learn_1.0',
     wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid
}


# In[ ]:


#save model
model_details=wml_client.repository.store_model(
    model=DEMO_MODEL,
    meta_props=model_props,
    training_data=x_train,
    training_target=y_train
)


# In[ ]:


model_details


# In[ ]:


model_id=wml_client.repository.get_model_id(model_details)
model_id


# In[ ]:


#set meta
deployment_props={
    wml_clients.deployments.ConfigurationMetaNames.NAME:DEPLOYMENT_NAME,
    wml_clients.deployments.ConfigurationMetaNames.ONLINE:{}
}


# In[ ]:


#deploy
deployment=wml_clients.deployments.create(
     artifact_uid=model_id,
     meta_props=deployment_props)

