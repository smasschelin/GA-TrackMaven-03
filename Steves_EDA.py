
# coding: utf-8

# # Change Working Directory

# In[1]:


import pandas as pd
import os


# In[2]:


abspath = os.path.abspath('OO_Importing_Data.py') # Get filepath
dname = os.path.dirname(abspath) # Get directory
os.chdir(dname) # Make directory working directory


# # EDA

# ## Reading in the Data

# In[3]:


data = pd.read_json('assets/newdump.json')


# ## Splitting up the 'Channel Info' dictionaries into seperate columns

# In[4]:


data['channel_type'] = [x['type'] for x in data['channel_info']]
data['channel'] = [x['channel'] for x in data['channel_info']]


# In[5]:


data.drop('channel_info', axis = 1, inplace=True)


# ## Showing only Facebook and Instagram Data

# In[6]:


FB_and_IG_data = data.loc[(data['channel'] == 'facebook') | (data['channel'] == 'instagram')]


# ## Removing '' and [] from 'type' column for queries

# In[7]:


FB_and_IG_data['channel_type'] = FB_and_IG_data['channel_type'].apply(lambda x: x[0])


# ## Breaking down number of entries for each type of post. Looks like Facebook is a clear winner

# In[8]:


FB_and_IG_data['type'].value_counts()


# ## Replacing Values in 'brand' with the actual publication

# In[9]:


FB_and_IG_data['brand'].value_counts()


# Found these by plugging urls into google and seeing what showed up
# * Brand 137314 = Conde Naste Traveler
# * Brand 137329 = W Magazine
# * Brand 137321 = OnSelf Magazine
# * Brand 137325 = Vanity Fair
# * Brand 137300 = Clever
# * Brand 137322 = Teen Vogue
# * Brand 137299 = Allure
# * Brand 137326 = Vogue
# * Brand 137316 = Glamor

# In[10]:


brands = {137314 : 'Conde_Naste_Traveler', 
          137329 : 'W_Magazine',
          137321 : 'Onself',
          137325 : 'Vanity_Fair', 
          137300 : 'Clever', 
          137322 : 'Teen_Vogue', 
          137299 : 'Allure', 
          137326 : 'Vogue',137316 : 'Glamor'
         }
FB_and_IG_data['brand'] = FB_and_IG_data['brand'].map(brands)                                                     


# ## Taking a subset of just the Instagram data

# In[11]:


instagram = FB_and_IG_data.loc[FB_and_IG_data['type'].isin(['instagram pic', 'instagram vid'])]


# In[12]:


instagram = instagram.reset_index(drop=True)


# ## Turning post_id into urls

# In[13]:


instagram['post_id'] = [x['post_id'] for x in instagram['content']]


# In[14]:


instagram['urls'] = 'http://instdrive.com/p/' + instagram['post_id'].astype(str)


# In[18]:


instagram.drop(['post_id'], axis=1, inplace=True)


# In[19]:


instagram.head()

