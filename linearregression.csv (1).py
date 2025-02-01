#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewsPaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


#Boxplot for daily column
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[7]:


sns.histplot(data1['daily'], kde = True, stat='density',)
plt.show()


# In[8]:


sns.histplot(data1['sunday'], kde = True, stat='density')
plt.show()


# In[9]:


plt.figure(figsize=(6,3))
plt.title("Box plot for sunday Sales")
plt.boxplot(data1["sunday"], vert = False)
plt.show()


# In[25]:


# Scatter plot and Correlation Strength
x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[29]:


data1["daily"].corr(data1["sunday"])


# In[31]:


# Build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[35]:


model1.summary()


# In[ ]:




