# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:32:33 2017

@author: benps
"""

#%%
import pandas as pd

#%%
# This presumes you have just run the 00_Importing_Data.py file, and that you have
# the 'facebook' dataframe in your environment.

facebook['content'][0]

#%%

# Assert that all keys in the facebook 'content' column are the same.
assert sum([facebook['content'][i].keys() != facebook['content'][0].keys() for i in range(10)]) == 0

#%%

content_keys = list(facebook['content'][0].keys())
# content_keys.remove('reactions')


for key in content_keys:
    facebook[key] = [facebook['content'][i].get(key) for i in range(facebook.shape[0])]
    
    
#%%

reactions_keys = list(facebook['reactions'][0].keys())

#%%

for key in reactions_keys:
    facebook[key] = [facebook['reactions'][i] for i in range(facebook.shape[0])]
    
#%%
facebook.isnull().sum()