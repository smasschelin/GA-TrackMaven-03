# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:32:33 2017

@author: benps
"""

#%%
import pandas as pd

facebook = facebook.sample(1000)
facebook = facebook.reset_index(drop=True)

#%%
# This presumes you have just run the 00_Importing_Data.py file, and that you have
# the 'facebook' dataframe in your environment.

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
def safe_dict_lookup(entry, key, fill=0):
    try:
        entry.get(key)
    except:
        return(fill)
#%%
for key in reactions_keys:
    facebook[key] = [facebook.loc[i, 'reactions'].get(key) for i in range(facebook.shape[0])]

facebook.drop('reactions', axis=1, inplace=True)
facebook.drop('content', axis=1, inplace=True)

    
#%%
facebook.isnull().sum()

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
    
#%%

