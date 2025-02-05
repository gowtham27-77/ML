#!/usr/bin/env python
# coding: utf-8

# Assumptions in Multilinear Regression
# 
# 1. Linearity: The relationship between the predictors and the response is linear.
# 
# 2. Independence: Observations are independent of each other.
# 
# 3. Homoscedasticity: The residuals (Y - Y_hat)) exhibit constant variance at all levels of the predictor.
# 
# 4. Normal Distribution of Errors: The residuals of the model are normally distributed.
# 
# 5. No multicollinearity: The independent variables should not be too highly correlated with each other.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels. formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


# Rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# Description of columns
# 
# * MPG : Milege of the car (Mile per Gallon) (This is Y-column to be predicted)
# * HP : Horse Power of the car (X1 column)
# * VOL : Volume of the car (size) (X2 column)
# * SP : Top speed of the car (Miles per Hour) (X3 column)
# * WT : Weight of the car (Pounds) (X4 Column)

# In[6]:


cars.info()


# In[7]:


cars.isnull().sum()


# #### Observations
# * There are no missing values
# * There are 81 observations (81 diffrent cars data)
# * The data types of the columns are also relevant and valid

# In[17]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt. show()


# ####Observations from boxplot and histograms
# There are some extreme values observed in towards the right tail of SP and HP distributions.
# In VOL and WL columns, a few outliers are observed in both tails of their distributions
# The extreme values of cars data may have come from the specially designed nature of cars

# In[20]:


cars[cars.duplicated()]


# In[22]:


cars.corr()


# ####Observations from correlation plots and Coefficients
# Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG
# Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# Among x columns (x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP, VOL vs WT
# The high correlation among x columns is not desirable as it might lead to multicollinearity problem

# In[26]:


# Build model
#import statsmodels.formula.api as smf
model = smf.ols('MPG~WT+VOL+SP+HP', data=cars).fit()


# In[28]:


model.summary()


# In[ ]:




