#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
data


# In[3]:


data.info()


# In[4]:


# data structure
print(type(data))
print(data.shape)


# In[5]:


data.dtypes


# In[6]:


# Data dupplicate column and unnamed column
data1 = data.drop(['Unnamed: 0', "Temp C"], axis = 1)
data1


# In[7]:


data1.info()


# In[8]:


# convert the month column data type to float data type
data1['Month']=pd.to_numeric(data['Month'], errors='coerce')
data1.info()


# In[9]:


# checkimg for duplicated rows in the table
# print the duplicated row (one) only
data1[data1.duplicated(keep = False)]


# In[10]:


# change column names (Rename the columns)
data1.rename({'Solar.R': 'Solar','Temp': 'Temperature'}, axis = 1, inplace = True)
data1


# In[11]:


#display data1 info()
data1.info()


# In[12]:


# display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[13]:


#visualoize data1 missing values using graph
cols = data1.columns
colours = ['black', 'red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[14]:


#Find the mean and median values of ech numeric column
#Imputation of missing value with median
median_ozone=data1["Ozone"].median()
mean_ozone=data1["Ozone"].mean()
print("Median of Ozone :",median_ozone)
print("Mean of Ozone :",mean_ozone)


# In[15]:


# Replace the Ozone missing values with median value
data1['Ozone'] = data1['Ozone'].fillna (median_ozone)
data1.isnull().sum()


# In[32]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[34]:


# impute missing values (Replace NaN with mode) of "weather" using fillna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[36]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[38]:


# impute missing values (Replace NaN with mode) of "month" using fillna()
data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[40]:


# reset the index column
data1.reset_index(drop=True)


# In[44]:


# create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

# plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient= 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

# plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(data1["Ozone"],kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
 
#Adjust layout for better spacing
plt.tight_layout()

#Show the plot
plt.show()


# In[48]:


# create a figure for violin plot 
sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Violin Plot")
plt.show()


# In[50]:


# create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

# plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Solar"], ax=axes[0], color='skyblue', width=0.5, orient= 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

# plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(data1["Solar"],kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")
 
#Adjust layout for better spacing
plt.tight_layout()

#Show the plot
plt.show()


# In[ ]:




