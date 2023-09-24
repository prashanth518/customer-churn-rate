#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np   
import pandas as pd    
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt   
import matplotlib.style
import matplotlib.ticker as mtick
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df = pd.read_csv("C:\\Users\\LAKSHMI PRASANNA\\Desktop\\train.csv")


# In[12]:


df.head()


# In[13]:


df.tail()


# In[14]:


# for getting information about data
df.info()


# In[15]:


#descriptive statistics of the numerical variables
df.describe()


# 75% customers have tenure less than 55 months
# average monthly charges are 64.856626

# In[16]:


df['Churn'].value_counts()


# In[17]:


100*df['Churn'].value_counts()/len (df['Churn'])


# Data is highly imbalanced i.e 73.45 so,it is better to analyze the data with other parameters while taking the target values separately to get the INSIGHTS

# In[18]:


df['Churn'].value_counts().plot(kind='barh',figsize=(10,6))
plt.xlabel("count",labelpad=15)
plt.ylabel("Target variable",labelpad = 15)
plt.title("count of target variable per category",y=1.00)


# In[19]:


# To visualise chrning trends
sns.countplot(x='Churn', data=df)
plt.title('Churn Count')
plt.show()


# In[20]:


# To visualise the churning trends as per distribution of Tenure
sns.histplot(data=df, x='tenure', kde=True)
plt.title('Distribution of Tenure')
plt.show()


# # Obsevation
#     # Either New or Old customers are churning more, Customers
#     # Customers with Tenure 25 to 65 are somewhat stable as compared to very new and very old customers

# In[21]:


df.drop(['customerID', 'TotalCharges'] , axis=1, inplace=True)


# In[22]:


df.head(3)


# In[23]:


categorical_cols = df.select_dtypes(include=['object'])
categorical_cols_encoded = pd.get_dummies(categorical_cols, drop_first=True)
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
encoded_df = pd.concat([numeric_cols, categorical_cols_encoded], axis=1)


# In[24]:


encoded_df.info()


# In[25]:


df = encoded_df


# In[26]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges']])


# In[27]:


df.head()


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


df_train, df_test = train_test_split(df, test_size=0.2, random_state=13)


# In[30]:


len(df_test), len(df_train)


# In[31]:


y_train = df_train.Churn
y_test = df_test.Churn


# In[32]:


X_train = df_train.drop(['Churn'], axis = True)
X_test = df_test.drop(['Churn'], axis = True)


# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


model = LinearRegression()


# In[35]:


model.fit(X_train, y_train)


# In[36]:


y_pred = model.predict(X_test)


# In[37]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[38]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

