
# coding: utf-8

# In[3]:

import pandas as pd
import json
import csv


# In[4]:

data = pd.read_json('https://s3.amazonaws.com/temp-data-pulls/newdump.json')


# In[10]:

csv = data.to_csv()


# In[5]:

data


# In[6]:

data['type'].value_counts()


# In[26]:

instagram_data_pic = data[(data['type']  == 'instagram pic')]
instagram_data_vid = data[(data['type']  == 'instagram vid')]                   
instagram_data = instagram_data_pic.append(instagram_data_vid)


# In[27]:

instagram_data.shape


# In[15]:

faceboook_data = data[(data['type']  == 'facebook post')]


# In[28]:

faceboook_data.shape


# In[34]:

instagram_data.sort_values(by='engagement', ascending=0).head()


# In[35]:

faceboook_data.sort_values(by='engagement', ascending=0).head()


# In[37]:

faceboook_data['brand'].value_counts()


# In[39]:

instagram_data['brand'].value_counts()


# In[ ]:

faceboook_data.descrbe()


# In[ ]:

instagram_data.describe()


# In[ ]:



