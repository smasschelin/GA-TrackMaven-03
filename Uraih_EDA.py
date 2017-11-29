
# coding: utf-8

# In[1]:

import pandas as pd
import json
import csv


# In[2]:

data = pd.read_json('https://s3.amazonaws.com/temp-data-pulls/newdump.json')


# In[ ]:

data


# In[3]:

data['type'].value_counts()


# In[4]:

instagram_data_pic = data[(data['type']  == 'instagram pic')]
instagram_data_vid = data[(data['type']  == 'instagram vid')]                   
instagram_data = instagram_data_pic.append(instagram_data_vid)


# In[5]:

faceboook_data = data[(data['type']  == 'facebook post')]


# In[ ]:

instagram_data.sort_values(by='engagement', ascending=0).head()


# In[6]:

c.sort_values(by='engagement', ascending=0).head()


# In[96]:

# Review facebook data types
faceboook_data.info()


# In[7]:

faceboook_data['brand'].value_counts()


# In[8]:

instagram_data['brand'].value_counts()


# In[99]:

combo_data = instagram_data.append(faceboook_data)


# In[81]:

combo_data.to_csv('combo_data.csv', sep=',')


# In[100]:

combo_data.describe()


# In[131]:

# Combine instagram and facebook data
combo_data.head(10)


# In[ ]:

combo_data.c


# In[102]:

# Split up the 'channel info' column which is a series of dictionaries
combo_data['channel_type'] = [x['type'] for x in combo_data['channel_info']]
combo_data['channel'] = [x['channel'] for x in combo_data['channel_info']]

# Drop the channel info column now that we have split it into two new columns
combo_data.drop('channel_info', axis = 1, inplace=True)


# In[103]:

combo_data['channel_type'].head()


# In[104]:

# Remove brackets from from elements in column 
combo_data['channel_type'] = combo_data['channel_type'].apply(lambda x: x[0])


# In[105]:

combo_data['channel_type'].head()


# In[106]:

# Create csv of data
combo_data.to_csv('combo_data.csv')


# In[107]:

# Review column names
combo_data.columns


# In[ ]:

# Going back and update facebook df


# In[141]:

# Group  by brand and channel type for IG and facebook
brand_channel = combo_data.groupby('brand').channel_type.value_counts()
brand_channel = pd.DataFrame(brand_channel)

brand_channel


# In[143]:

# Split up the 'channel info' column which is a series of dictionaries
faceboook_data['channel_type'] = [x['type'] for x in faceboook_data['channel_info']]
faceboook_data['channel'] = [x['channel'] for x in faceboook_data['channel_info']]

# Drop the channel info column now that we have split it into two new columns
faceboook_data.drop('channel_info', axis = 1, inplace=True)


# In[144]:

faceboook_data['channel_type'] = faceboook_data['channel_type'].apply(lambda x: x[0])


# In[146]:

# Group  by brand and channel type for IG and facebook
brand_channel_f = faceboook_data.groupby('brand').channel_type.value_counts()
brand_channel_f = pd.DataFrame(brand_channel)

brand_channel_f


# In[ ]:



