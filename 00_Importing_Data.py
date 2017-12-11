# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:03:40 2017

@author: benps
"""
#%% Define working directory

import os

abspath = os.path.abspath('OO_Importing_Data.py') # Get filepath
dname = os.path.dirname(abspath) # Get directory
os.chdir(dname) # Make directory working directory


#%% Import modules and libraries

import pandas as pd
#import numpy as np

#%% Load JSON data

data = pd.read_json('assets/newdump.json')

#%% Split channel_info (series of dictionaries) into two columns

data['channel_type'] = [x['type'] for x in data['channel_info']]
data['channel'] = [x['channel'] for x in data['channel_info']]

#%% Drop channel_info column after splitting into two columns

data.drop('channel_info', axis = 1, inplace=True)

#%% Filter Instagram and Facebook data from overall DataFrame

data = data.loc[(data['channel'] == 'facebook') | (data['channel'] == 'instagram')]

#%% Check and ensure channel_type series has exactly one element

assert data['channel_type'].apply(lambda x: len(x) != 1).sum() == 0

data['channel_type'] = data['channel_type'].apply(lambda x: x[0])

#%% Create brand dictionary and new brand_name column

brands = {137314 : 'Conde_Nast_Traveler', 
          137329 : 'W_Magazine',
          137321 : 'Self',
          137325 : 'Vanity_Fair', 
          137300 : 'Clever', 
          137322 : 'Teen_Vogue', 
          137299 : 'Allure', 
          137326 : 'Vogue',
          137316 : 'Glamour'
         }

data['brand_name'] = data['brand'].map(brands)      

#%% Split data into Instagram and Facebook streams

instagram = data[data['channel'] == 'instagram']
instagram = instagram.reset_index(drop=True)
facebook  = data[data['channel'] == 'facebook']
facebook = facebook.reset_index(drop=True)

#%% Turn post_id into urls

instagram['post_id'] = [x['post_id'] for x in instagram['content']]

instagram['urls'] = 'http://instdrive.com/p/' + instagram['post_id'].astype(str)

instagram.drop(['post_id'], axis=1, inplace=True)

#%% Save whole DataFrame to CSV (deprecated)

# data.to_csv('assets/FB_and_IG_data.csv')
