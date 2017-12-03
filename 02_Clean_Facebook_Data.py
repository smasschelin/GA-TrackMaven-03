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

facebookdata = FB_and_IG_data.loc[FB_and_IG_data.type == 'facebook post'].copy(deep=True)
facebookdata.reset_index(drop=True,inplace=True)

#%% Break out the content dictionary from the content column

facebookdata['comments'] = [xx['comment_count'] for xx in facebookdata.content]
facebookdata['content_type'] = [xx['content_type'] for xx in facebookdata.content]
facebookdata['like_count'] = [xx['like_count'] for xx in facebookdata.content]
facebookdata['media_caption'] = [xx['media_caption'] for xx in facebookdata.content]
facebookdata['media_title'] = [xx['media_name'] for xx in facebookdata.content]
facebookdata['message'] = [xx['message'] for xx in facebookdata.content]
facebookdata['permalink'] = [xx['permalink'] for xx in facebookdata.content]
facebookdata['picture_url'] = [xx['picture_url'] for xx in facebookdata.content]

#%% turn the reaction dictionary into something iterable because for some reason
#   it doesn't like me trying to directly access some of them (nulls?)

angry = [0]*len(facebookdata.content)
laugh = [0]*len(facebookdata.content)
loves = [0]*len(facebookdata.content)
sadss = [0]*len(facebookdata.content)
wowss = [0]*len(facebookdata.content)

for ii in range(len(facebookdata.content)):
    try:
        angry[ii] = facebookdata.iloc[ii].content['reactions']['angry_count']
        laugh[ii] = facebookdata.iloc[ii].content['reactions']['haha_count']
        loves[ii] = facebookdata.iloc[ii].content['reactions']['love_count']
        sadss[ii] = facebookdata.iloc[ii].content['reactions']['sad_count']
        wowss[ii] = facebookdata.iloc[ii].content['reactions']['wow_count']
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

facebookdata['angry_count'] = angry_count.values
facebookdata['haha_count'] = haha_count.values
facebookdata['love_count'] = love_count.values
facebookdata['sad_count'] = sad_count.values
facebookdata['wow_count'] = wow_count.values

facebookdata['shares'] = [xx['share_count'] for xx in facebookdata.content]

#%% Add reaction column that sums all the likes, angry, haha, love, sad, wow

facebookdata['reaction_count'] = facebookdata.apply(lambda row: row.like_count + row.angry_count
                                    +row.haha_count + row.love_count
                                    +row.sad_count  + row.wow_count, axis = 1)

#%% Turn channel_type column into list of strings

facebookdata.channel_type = facebookdata.channel_type.apply(lambda x: x[0] for x in facebookdata.channel_type)

#%% Save Facebook data to its own CSV

facebookdata.to_csv('assets/facebookdata.csv')
