
# coding: utf-8

# In[1]:


import pandas as pd
import json
import csv
import numpy as np


# In[2]:


data = pd.read_json('https://s3.amazonaws.com/temp-data-pulls/newdump.json')


# # Perform EDA

# In[3]:


data


# In[4]:


data['type'].value_counts()


# In[5]:


instagram_data_pic = data[(data['type']  == 'instagram pic')]
instagram_data_vid = data[(data['type']  == 'instagram vid')]                   
instagram_data = instagram_data_pic.append(instagram_data_vid)


# In[6]:


facebook_data = data[(data['type']  == 'facebook post')]


# In[7]:


instagram_data.sort_values(by='engagement', ascending=0).head()


# In[8]:


# Review facebook data types
facebook_data.info()


# In[9]:


# Review the distributions of brands in the facebook df
facebook_data['brand'].value_counts()


# In[10]:


# Review the distributions of brands in the instagram df
instagram_data['brand'].value_counts()


# In[11]:


# Create a combined df for later analysis
combo_data = instagram_data.append(facebook_data)


# In[12]:


combo_data.describe()


# In[13]:


# Combine instagram and facebook data
combo_data.head(3)


# In[14]:


# Review Missing Values
combo_data.isnull().sum()


# In[15]:


combo_data.sort_values(by='engagement', ascending=0).head()


# In[16]:


# Split up the 'channel info' column which is a series of dictionaries
combo_data['channel_type'] = [x['type'] for x in combo_data['channel_info']]
combo_data['channel'] = [x['channel'] for x in combo_data['channel_info']]

# Drop the channel info column now that we have split it into two new columns
combo_data.drop('channel_info', axis = 1, inplace=True)


# In[17]:


combo_data['channel_type'].head()


# In[18]:


# Remove brackets from from elements in column 
combo_data['channel_type'] = combo_data['channel_type'].apply(lambda x: x[0])


# In[19]:


combo_data['channel_type'].head()


# In[20]:


# Create csv of the combined df
# combo_data.to_csv('combo_data.csv')


# In[21]:


# Review column names
combo_data.columns


# In[22]:


# Group  by brand and channel type for IG and facebook
brand_channel = combo_data.groupby('brand').channel_type.value_counts()
brand_channel = pd.DataFrame(brand_channel)

brand_channel


# In[23]:


facebook_data.head(2)


# In[27]:


# Split up the 'channel info' column which is a series of dictionaries
facebook_data['channel_type'] = [x['type'] for x in facebook_data['channel_info']]
facebook_data['channel'] = [x['channel'] for x in facebook_data['channel_info']]

# Drop the channel info column now that we have split it into two new columns
facebook_data.drop('channel_info', axis = 1, inplace=True)


# In[28]:


facebook_data['channel_type'] = facebook_data['channel_type'].apply(lambda x: x[0])


# In[30]:


# Group by brand and channel type for IG and facebook
brand_channel_f = facebook_data.groupby('brand').channel_type.value_counts()
brand_channel_f = pd.DataFrame(brand_channel)

brand_channel_f


# In[31]:


# Examine contents column contents
facebook_data['content'][0]


# In[32]:


facebook_data.reset_index(inplace=True)


# In[33]:


facebook_data.head()


# In[34]:


# # Define a function that looks up keys 
# def lookup(dic, key, *keys):
#     if keys:
#         return lookup(dic.get(key, {}), *keys)
#     return dic.get(key)


# In[41]:


# Look up values in content column
# print(lookup(facebook_data.content,'love_count'))


# In[42]:


# facebook_data['content'][:][0]['reactions']['love_count']


# In[ ]:


# wow_count = [x['angry_count'] for x in facebook_data['content']['reactions']['wow_count']]


# In[38]:


# Spliting individual content into seperate columns
facebook_data['comment_count'] = [x['comment_count'] for x in facebook_data['content']]
facebook_data['content_type'] = [x['content_type'] for x in facebook_data['content']]
facebook_data['hashtags'] = [x['hashtags'] for x in facebook_data['content']]
facebook_data['like_count'] = [x['like_count'] for x in facebook_data['content']]
facebook_data['link_url'] = [x['link_url'] for x in facebook_data['content']]
facebook_data['links'] = [x['links'] for x in facebook_data['content']]
facebook_data['media_caption'] = [x['media_caption'] for x in facebook_data['content']]
facebook_data['media_description'] = [x['media_description'] for x in facebook_data['content']]
facebook_data['media_name'] = [x['media_name'] for x in facebook_data['content']]
facebook_data['message'] = [x['message'] for x in facebook_data['content']]
facebook_data['permalink'] = [x['permalink'] for x in facebook_data['content']]
facebook_data['picture_url'] = [x['picture_url'] for x in facebook_data['content']]
facebook_data['post_id'] = [x['post_id'] for x in facebook_data['content']]
# facebook_data['reactions'] = [x['reactions'] for x in facebook_data['content']]
# # facebook_data['haha_count'] = [x['haha_count'] for x in facebook_data['content']] 
# facebook_data['love_count'] = [x['love_count'] for x in facebook_data['content']]
# facebook_data['sad_count'] = [x['sad_count'] for x in facebook_data['content']]
# facebook_data['wow_count'] = [x['wow_count'] for x in facebook_data['content']]
# facebook_data['share_count'] = [x['share_count'] for x in facebook_data['content']]


# In[39]:


# Review content of the contents column
facebook_data['content'][0]['reactions']['love_count']


# In[43]:


# Assert that all keys in the facebook 'content' column are the same.
assert sum([facebook_data['content'][i].keys() != facebook_data['content'][0].keys() for i in range(10)]) == 0


# In[114]:


# content_keys.remove('reactions)

content_keys = list(facebook_data['content'][0].keys())


# In[46]:


facebook_data.head(2)


# In[47]:


for key in content_keys:
    facebook_data[key] = [facebook_data['content'][i].get(key) for i in range(facebook_data.shape[0])]


# In[48]:


reactions_keys = list(facebook_data['reactions'][0].keys())

reactions_keys


# In[115]:


facebook_data.head(3)


# In[59]:


facebook_data.reset_index(inplace=True)


# In[116]:


# Create columns out of dictionaries within the reations column 
for key in reactions_keys:
    facebook_data[key] = [facebook_data['reactions'][i][key] for i in range(facebook_data.shape[0])]


# In[134]:


# Code from James

angry = [0]*len(facebook_data.content)
laugh = [0]*len(facebook_data.content)
loves = [0]*len(facebook_data.content)
sadss = [0]*len(facebook_data.content)
wowss = [0]*len(facebook_data.content)


# In[137]:


for ii in range(len(facebook_data.content)):
    try:
        angry[ii] = facebook_data.iloc[ii].content['reactions']['angry_count']
        laugh[ii] = facebook_data.iloc[ii].content['reactions']['haha_count']
        loves[ii] = facebook_data.iloc[ii].content['reactions']['love_count']
        sadss[ii] = facebook_data.iloc[ii].content['reactions']['sad_count']
        wowss[ii] = facebook_data.iloc[ii].content['reactions']['wow_count']
    except:
        angry[ii] = 0
        laugh[ii] = 0
        loves[ii] = 0
        sadss[ii] = 0
        wowss[ii] = 0


# In[138]:


angry_count = pd.Series(angry)
haha_count = pd.Series(laugh)
love_count = pd.Series(loves)
sad_count = pd.Series(sadss)
wow_count = pd.Series(wowss)


# In[140]:


facebook_data['angry_count'] = angry_count.values
facebook_data['haha_count'] = haha_count.values
facebook_data['love_count'] = love_count.values
facebook_data['sad_count'] = sad_count.values
facebook_data['wow_count'] = wow_count.values


# In[141]:


# Review missing values
facebook_data.isnull().sum()


# In[142]:


def safe_string_add(*args):
    string = ''
    for arg in args:
        if type(arg) == str:
            string += ' ' + arg
    return(string)        


# In[143]:


facebook_data['text'] = [safe_string_add(facebook_data['media_description'][i],
         facebook_data['media_name'][i]) for i in range(facebook_data.shape[0])]


# In[144]:


facebook_data.drop('content', axis=1, inplace=True)
facebook_data.drop('reactions', axis=1, inplace=True)


# In[145]:


facebook_data.columns


# In[153]:


# Remove columns that bring no value.
new_facebook_df = facebook_data.drop( ['level_0','index','id','share_token','urls','link_url','links','permalink', 'picture_url', 'post_id', 'Wows'], axis=1)


# In[154]:


# Create a new df for analysis
new_facebook_df.head(3)


# In[155]:


new_facebook_df.info()


# In[152]:


new_facebook_df.describe()

