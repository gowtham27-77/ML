#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv("data_clean.csv")
data


# In[5]:


data.info()


# In[13]:


# data structure
print(type(data))
print(data.shape)


# In[15]:


data.dtypes


# In[17]:


# Data dupplicate column and unnamed column
data1 = data.drop(['Unnamed: 0', "Temp C"], axis = 1)
data1


# In[19]:


data1.info()


# In[21]:


# convert the month column data type to float data type
data1['Month']=pd.to_numeric(data['Month'], errors='coerce')
data1.info()


# In[23]:


# checkimg for duplicated rows in the table
# print the duplicated row (one) only
data1[data1.duplicated(keep = False)]


# In[ ]:




