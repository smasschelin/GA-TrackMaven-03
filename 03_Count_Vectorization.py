#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:07:09 2017

This script defines a function that performs count vectorization on a text
column from a DataFrame. Necessary libraries and modules are included. Nothing
is output from running this script; the code beneath simply imports the modules
and readies the function for use.

Example code for running the function is shown below.

@author: jamesdenney
"""

#%% Library and module imports

import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#%% Explain function inputs

# REQUIRED INPUTS
#     df: Single media stream DataFrame to analyze (e.g., Facebook)
# stream: Media portfolio to analyze (e.g., Glamour Magazine)
#    col: DataFrame column to run count vectorization on ("message", "caption")
# target: Target column (likes, engagement, impact) to analyze and predict on
#           MUST BE A COLUMN TITLE WITH TYPE INT OR FLOAT
#
# OPTIONAL INPUTS
#  st_wd: binary input that determines whether stop words are included in
#         count vectorizer (1 removes them, 0 keeps them in); default is 1
#  ngram: tuple that defines the range of word/phrase lengths to run into the
#         vectorizer; default is (1,1), which looks at individual words
#           THIS MUST BE A TUPLE
#           example: (1,3) looks for single words, and 2- and 3-word phrases
#           example: (2,4) looks for two- and three-word phrases
#    top:  number of top phrases/words to display; default is 30
#
# EXAMPLE FUNCTION CALL
#
#   doCountVectorizer(data, 'facebook', 'message', 1, (1,3), 25)
#       reads in data DataFrame, analyzes "message" column
#       uses stop_words, looks for 1-3 word phrases, and displays top 25

#%% Function definition

def doCountVectorizer(data, stream, col, target, stop_wds=1, ngram=(1,1), top=25):
    
    #%% Check inputs for proper dtype
    
    
    #%% Set up stop word set to add to it if desired
    
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('http')
    stopwords.append('asdf') # this removes the placeholder for empty posts
    stopwords.append('nast')
    
    #%% Initialize count vectorizer
    
    if stop_wds:
        cvt = CountVectorizer(stop_words=stopwords, 
                              strip_accents="unicode",
                              ngram_range=ngram,
                              lowercase=True,
                              max_df=0.95,
                              max_features=5000)
    else:
        cvt = CountVectorizer(strip_accents="unicode",
                              ngram_range=ngram,
                              lowercase=True,
                              max_df=0.95,
                              max_features=5000)
    
    #%% Parse input DataFrame to only feature desired data
    
    # copy portion of DF, to prevent any unwanted edits to main DF
    df = data[data['brand_name']==stream].copy()
    
    # fill any NAN / NA values with gibberish and add gibberish to stop_words
    df[col].fillna('asdf', inplace=True)
    
    #%% Perform train/test split
    
    X_trn, X_tst, y_trn, y_tst = train_test_split(df[col],df[target],
                                                  test_size=0.3, random_state=6)
    
    #%% Pass training data through vectorizer
    
    fit_data = cvt.fit_transform(X_trn)
    
    # store words and frequencies in arrays, then combine them and make a DF
    allwords = cvt.get_feature_names()
    allfreqs = fit_data.toarray().sum(axis=0)
    
    stream_array = [[allwords[idx], allfreqs[idx]] for idx in range(len(allwords))]
    
    streamDF = pd.DataFrame(data=stream_array)
    streamDF.columns = ['phrase', 'frequency']
    
    #%% Return the input number of top phrases
    
    streamDF.sort_values('frequency', ascending=False)[:top]
    