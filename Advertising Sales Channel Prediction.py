#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


advertising = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/Advertising.csv"))
advertising.head()

Data Inspection
# In[3]:


advertising.shape


# In[4]:


advertising.info()


# In[5]:


advertising.describe()

Data Cleaning# Checking Null values
# In[6]:


advertising.isnull().sum()*100/advertising.shape[0]

# There are no NULL values in the dataset, hence it is clean.# Outlier Analysis
# In[8]:


fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(advertising['TV'], ax = axs[0])
plt2 = sns.boxplot(advertising['newspaper'], ax = axs[1])
plt3 = sns.boxplot(advertising['radio'], ax = axs[2])
plt.tight_layout()

# There are no considerable outliers present in the data.
# # Exploratory Data Analysis
# Univariate Analysis
# Sales (Target Variable)

# In[10]:


sns.boxplot(advertising['sales'])
plt.show()


# In[12]:


# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(advertising, x_vars=['TV', 'newspaper', 'radio'], y_vars='sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[13]:


# Let's see the correlation between different variables.
sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()

Generic Steps in model building using statsmodels
We first assign the feature variable, TV, in this case, to the variable X and the response variable, Sales, to the variable y.
# In[15]:


X = advertising['TV']
y = advertising['sales']

Train-Test Split
# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[17]:


# Let's now take a look at the train dataset

X_train.head()


# In[18]:


y_train.head()

Building a Linear Model
You first need to import the statsmodel.api library using which you'll perform the linear regression.
# In[19]:


import statsmodels.api as sm


# In[20]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[21]:


lr.params

The fit is significant. Let's visualize how well the model fit the data.

From the parameters that we get, our linear regression equation becomes:

Sales=6.948+0.054×TV
# In[24]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[ ]:




