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
FB_and_IG_data = data.loc[(data['channel'] == 'facebook') | (data['channel'] == 'instagram')]

#%%

FB_and_IG_data.to_csv('assets/FB_and_IG_data.csv')
