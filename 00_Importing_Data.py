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

#Subsetting instagram data from the full set.
instagram = FB_and_IG_data.loc[FB_and_IG_data['type'].isin(['instagram pic', 'instagram vid'])]
instagram = instagram.reset_index(drop=True)

#Getting data from the content column
instacontent = instagram['content']

#Seeing how many image entries we have.
sum([instacontent[i].keys() == instacontent[0].keys() for i in range(len(instacontent))])

#Seeing how many video entries we have.
sum([instacontent[i].keys() != instacontent[0].keys() for i in range(len(instacontent))])

#Splitting Instagram data into seperate image and video frames
instapics = instagram.loc[(instagram['type'] == 'instagram pic')]
instapics = instapics.reset_index(drop=True)

instavids = instagram.loc[(instagram['type'] == 'instagram vid')]
instavids = instavids.reset_index(drop=True)
#Setting up seperate DataFrames for picture and video entries
instapicscontent = instapics['content']
picframe = pd.DataFrame.from_dict(instapicscontent)

instavidscontent = instavids['content']
vidframe = pd.DataFrame.from_dict(instavidscontent)

#Splitting the picture entries content dictionary into seperate columns based upon keys
picframe['caption']  = [x['caption'] for x in instapics['content']]
picframe['comment_count']  = [x['comment_count'] for x in instapics['content']]
picframe['filter_name'] = [x['filter_name'] for x in instapics['content']]
picframe['hashtags'] = [x['hashtags'] for x in instapics['content']]
picframe['image_url'] = [x['image_url'] for x in instapics['content']]
picframe['like_count'] = [x['like_count'] for x in instapics['content']]
picframe['link'] = [x['link'] for x in instapics['content']]
picframe['links'] = [x['links'] for x in instapics['content']]
picframe['post_id'] = [x['post_id'] for x in instapics['content']]

picframe.drop(['content'], axis = 1, inplace = True)
picframe = picframe.reset_index(drop=True)
picframe.head()

#Ditto, but now for the video entries
vidframe['caption'] = [x['caption'] for x in instavids['content']]
vidframe['comment_count'] = [x['comment_count'] for x in instavids['content']]
vidframe['filter_name'] = [x['filter_name'] for x in instavids['content']]
vidframe['hashtags'] = [x['hashtags'] for x in instavids['content']]
vidframe['image_url'] = [x['image_url'] for x in instavids['content']]
vidframe['like_count'] = [x['like_count'] for x in instavids['content']]
vidframe['link'] = [x['link'] for x in instavids['content']]
vidframe['links'] = [x['links'] for x in instavids['content']]
vidframe['post_id'] = [x['post_id'] for x in instavids['content']]
vidframe['video_url'] = [x['video_url'] for x in instavids['content']]

vidframe.drop(['content'], axis = 1, inplace = True)
vidframe = vidframe.reset_index(drop=True)
vidframe.head()
