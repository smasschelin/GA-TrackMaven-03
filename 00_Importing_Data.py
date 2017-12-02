# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:03:40 2017

@author: benps
"""
#%%
# Change working directory:
import os

abspath = os.path.abspath('OO_Importing_Data.py') # Get filepath
dname = os.path.dirname(abspath) # Get directory
os.chdir(dname) # Make directory working directory


#%%
# Import modules:
import pandas as pd

#%%
# Load JSON data:
data = pd.read_json('assets/newdump.json')

#%%
# Split up the 'channel info' column which is a series of dictionaries
data['channel_type'] = [x['type'] for x in data['channel_info']]
data['channel'] = [x['channel'] for x in data['channel_info']]

#%%
# Drop the channel info column now that we have split it into two new columns
data.drop('channel_info', axis = 1, inplace=True)

#%%
# Filter only instagram and facebook data
data = data.loc[(data['channel'] == 'facebook') | (data['channel'] == 'instagram')]

#%%

# Check and make sure every row in the 'channel type' series has exacly 1 element
assert data['channel_type'].apply(lambda x: len(x) != 1).sum() == 0

data['channel_type'] = data['channel_type'].apply(lambda x: x[0])

#%%

brands = {137314 : 'Conde_Naste_Traveler', 
          137329 : 'W_Magazine',
          137321 : 'Onself',
          137325 : 'Vanity_Fair', 
          137300 : 'Clever', 
          137322 : 'Teen_Vogue', 
          137299 : 'Allure', 
          137326 : 'Vogue',137316 : 'Glamor'
         }

data['brand_name'] = data['brand'].map(brands)      


#%%
instagram = data[data['channel'] == 'instagram']
instagram = instagram.reset_index(drop=True)
facebook  = data[data['channel'] == 'facebook']
facebook = facebook.reset_index(drop=True)

#%%

# Turning post_id into urls


instagram['post_id'] = [x['post_id'] for x in instagram['content']]

instagram['urls'] = 'http://instdrive.com/p/' + instagram['post_id'].astype(str)

instagram.drop(['post_id'], axis=1, inplace=True)


#%%
# data.to_csv('assets/FB_and_IG_data.csv')
