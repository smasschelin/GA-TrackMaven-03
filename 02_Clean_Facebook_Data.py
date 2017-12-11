#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:16:47 2017

Takes up where 00_Importing_Data and 01_Instagram_Data_Clean leave off, by
separating the Facebook data from the full data set. This script cleans data by
removing/replacing nulls, creating columns out of content dictionaries, and
other operations necessary to create a well-behaved data set for analysis.

This script assumes the presence of the following:
    
    pandas is imported as pd
    numpy is imported as np
    FB_and_IG_data is a DataFrame currently in your operating environment

@author: jamesdenney
"""

#%% Separate the Facebook data from the full set

facebook = FB_and_IG_data.loc[FB_and_IG_data.type == 'facebook post'].copy(deep=True)
facebook.reset_index(drop=True,inplace=True)

#%% Break out the content dictionary from the content column

facebook['comments'] = [xx['comment_count'] for xx in facebook.content]
facebook['content_type'] = [xx['content_type'] for xx in facebook.content]
facebook['like_count'] = [xx['like_count'] for xx in facebook.content]
facebook['media_caption'] = [xx['media_caption'] for xx in facebook.content]
facebook['media_title'] = [xx['media_name'] for xx in facebook.content]
facebook['message'] = [xx['message'] for xx in facebook.content]
facebook['permalink'] = [xx['permalink'] for xx in facebook.content]
facebook['picture_url'] = [xx['picture_url'] for xx in facebook.content]

#%% turn the reaction dictionary into something iterable because for some reason
#   it doesn't like me trying to directly access some of them (nulls?)

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

facebook['shares'] = [xx['share_count'] for xx in facebook.content]

#%% Add reaction column that sums all the likes, angry, haha, love, sad, wow

facebook['reaction_count'] = facebook.apply(lambda row: row.like_count + row.angry_count
                                    +row.haha_count + row.love_count
                                    +row.sad_count  + row.wow_count, axis = 1)

#%% Turn channel_type column into list of strings

facebook.channel_type = facebook.channel_type.apply(lambda x: x[0] for x in facebook.channel_type)


#%%

def safe_string_add(*args):
    string = ''
    for arg in args:
        if type(arg) == str:
            string += ' ' + arg
    return(string)        
    
#%%
facebook['text'] = [safe_string_add(facebook['media_description'][i],
         facebook['media_name'][i]) for i in range(facebook.shape[0])]

#%% Save Facebook data to its own CSV

facebook.to_csv('assets/facebook_data.csv')
