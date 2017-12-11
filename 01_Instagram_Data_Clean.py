# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 18:04:00 2017

Subsets Instagram data from the full "data" DataFrame. Requires the following:
    
    DataFrame named Instagram
    Libraries in environment: pandas as pd

@author: smasschelin
"""

#%% Subsets Instagram data from full data DataFrame

#instagram = FB_and_IG_data.loc[FB_and_IG_data['type'].isin(['instagram pic', 'instagram vid'])]
# instagram = instagram.reset_index(drop=True)

# Getting data from the content column

instacontent = instagram['content']

#%% Determine number of image and video entries

# images
sum([instacontent[i].keys() == instacontent[0].keys() for i in range(len(instacontent))])

# videos
sum([instacontent[i].keys() != instacontent[0].keys() for i in range(len(instacontent))])

#%% Split Instagram data into separate image and video DataFrames

instapics = instagram.loc[(instagram['type'] == 'instagram pic')]
instapics = instapics.reset_index(drop=True)

instavids = instagram.loc[(instagram['type'] == 'instagram vid')]
instavids = instavids.reset_index(drop=True)

#%% Set up separate DataFrames for picture and video entries

instapicscontent = instapics['content']
picframe = pd.DataFrame.from_dict(instapicscontent)

instavidscontent = instavids['content']
vidframe = pd.DataFrame.from_dict(instavidscontent)

#%% Split picture entries content dictionary into columns based on keys

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

#%% Repeat content dictionary split for video entries

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
