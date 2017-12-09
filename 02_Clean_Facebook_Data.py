#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:16:47 2017
Updated on Sat Dec 09 12:41:00 2017

Takes up where 00_Importing_Data and 01_Instagram_Data_Clean leave off, by
separating the Facebook data from the full data set. This script cleans data by
removing/replacing nulls, creating columns out of content dictionaries, and
other operations necessary to create a well-behaved data set for analysis.

This script assumes the presence of the following:
    
    pandas is imported as pd
    numpy is imported as np
    facebook is a DataFrame currently in your operating environment

@author: jamesdenney
"""

#%% Separate the Facebook data from the full set

# deprecated from first iteration in which output of 00_script was FB_and_IG_data DF
#facebook = FB_and_IG_data.loc[FB_and_IG_data.type == 'facebook post'].copy(deep=True)
#facebook.reset_index(drop=True,inplace=True)

#%% Break out the content dictionary from the content column

facebook['comments'] = [row['comment_count'] for row in facebook.content]
facebook['content_type'] = [row['content_type'] for row in facebook.content]
facebook['like_count'] = [row['like_count'] for row in facebook.content]
facebook['media_caption'] = [row['media_caption'] for row in facebook.content]
facebook['media_title'] = [row['media_name'] for row in facebook.content]
facebook['message'] = [row['message'] for row in facebook.content]
facebook['permalink'] = [row['permalink'] for row in facebook.content]
facebook['picture_url'] = [row['picture_url'] for row in facebook.content]

#%% Turn the reaction dictionary into arrays because of None values

angry = [0]*len(facebook.content)
laugh = [0]*len(facebook.content)
loves = [0]*len(facebook.content)
sadss = [0]*len(facebook.content)
wowss = [0]*len(facebook.content)

for ii in range(len(facebook.content)):
    try:
        angry[ii] = facebook.iloc[ii].content['reactions']['angry_count']
        laugh[ii] = facebook.iloc[ii].content['reactions']['haha_count']
        loves[ii] = facebook.iloc[ii].content['reactions']['love_count']
        sadss[ii] = facebook.iloc[ii].content['reactions']['sad_count']
        wowss[ii] = facebook.iloc[ii].content['reactions']['wow_count']
    except:
        angry[ii] = 0
        laugh[ii] = 0
        loves[ii] = 0
        sadss[ii] = 0
        wowss[ii] = 0

angry_count = pd.Series(angry)
haha_count = pd.Series(laugh)
love_count = pd.Series(loves)
sad_count = pd.Series(sadss)
wow_count = pd.Series(wowss)

facebook['angry_count'] = angry_count.values
facebook['haha_count'] = haha_count.values
facebook['love_count'] = love_count.values
facebook['sad_count'] = sad_count.values
facebook['wow_count'] = wow_count.values

facebook['angry_count'].fillna(0, inplace=True)
facebook['haha_count'].fillna(0, inplace=True)
facebook['love_count'].fillna(0, inplace=True)
facebook['sad_count'].fillna(0, inplace=True)
facebook['wow_count'].fillna(0, inplace=True)

facebook['urls'].fillna('no_url', inplace=True)

facebook['shares'] = [row['share_count'] for row in facebook.content]

#%% Add reaction column that sums all the likes, angry, haha, love, sad, wow

facebook['reaction_count'] = facebook.apply(lambda row: row.like_count + row.angry_count
                                    +row.haha_count + row.love_count
                                    +row.sad_count  + row.wow_count, axis = 1)

#%% Turn channel_type column into list of strings

# deprecated; handled in 00_ script
#facebook.channel_type = facebook.channel_type.apply(lambda row: row[0] for row in facebook.channel_type)

#%% Save Facebook data to its own CSV

facebook.to_csv('assets/facebook.csv')
