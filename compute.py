#!/usr/bin/env python
# coding: utf-8

# In[1]:


#python modules
# module is a collection of fumctions
# The file should be saved as .py file(python script file)


# In[9]:


def mean_value(*n):
    sum = 0
    counter = 0
    for x in n:
        counter = counter +1
        sum+=x
        mean = sum/counter
        return mean


# In[17]:


mean_value(3,5,5,56,4,3)


# In[11]:


def product(*n):
    result = 1
    for i in range(len(n)):
        result *=n[i]
    return result


# In[13]:


product(5,6,43,5,2)


# In[ ]:




