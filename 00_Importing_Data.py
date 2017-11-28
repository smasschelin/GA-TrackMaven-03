# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:03:40 2017

@author: benps
"""

#%%

import pandas as pd

#%%

data = pd.read_json('https://s3.amazonaws.com/temp-data-pulls/newdump.json')

#%%

data['channel_info']['channel'] in ['facebook','instagram']

#%%
data['type'].value_counts()