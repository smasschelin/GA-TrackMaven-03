
# coding: utf-8

# Amazon Rekognition – Image Detection and Recognition 

#%% Import Libraries

import boto
import boto3
conn = boto.connect_s3()
import requests
import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import regex as re

#%% Read in CSVs that contain columns of url images and their IDs

ig_urls_pics = pd.read_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/instagram_pic_urls.csv', encoding='ISO-8859-1', index_col=0)


#%% EDA

# show Instagram image URL header and shape
ig_urls_pics.head(3)
ig_urls_pics.shape

#%% Create a list of urls from instagram image urls

image_list = ig_urls_pics.image_url


#%% Use the credentials in ~/.aws/credentials

s3 = boto3.resource('s3')
bucket_name_to_upload_image_to = 'igimages'

#%% Do this as a quick and easy check to make sure your S3 access is OK

for bucket in s3.buckets.all():
    if bucket.name == bucket_name_to_upload_image_to:
        print('Good to go. Found the bucket to upload the image into.')
        good_to_go = True

if not good_to_go:
    print('Not seeing your s3 bucket, might want to double check permissions in IAM')

#%% Allows user to upload pictures to Amazon AWS
    
# This is done to prevent saving pictures to the local machine.
# Code from Natalie Olivo's Medium post. 
# https://medium.com/@NatalieOlivo/use-python-to-collect-image-tags-using-aws-reverse-image-search-engine-rekognition-eccf1f259a8d

mapping_dict ={}
for i, img_url in enumerate(image_list[:]):
    
    img_name = "img_%05d" % (i,)
    mapping_dict[img_name] = img_url
    
    if (img_url == np.nan) | (str(img_url) == "nan"):
        continue
    else:
        # Uses the creds in ~/.aws/credentials
        s3_image_filename = img_name
        internet_image_url = img_url

        # Given an Internet-accessible URL, download the image and upload it to S3,
        # without needing to persist the image to disk locally
        req_for_image = requests.get(internet_image_url, stream=True)
        file_object_from_req = req_for_image.raw
        req_data = file_object_from_req.read()

        # Do the actual upload to s3
        s3.Bucket(bucket_name_to_upload_image_to).put_object(Key=s3_image_filename, Body=req_data)


#%% Save down the mapping dictionary
# This allows future re-mapping of image tags to the full dataframe.

mapping_dict = pd.DataFrame(mapping_dict, index = range(0,len(mapping_dict)))
mapping_dict = pd.DataFrame(mapping_dict.T[0])
mapping_dict.to_csv('mappingdict_Date.csv')


#%% Creates both wide and long df's with image tags from Rekognition

bucket_name = 'igimages'
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)
images = [img.key for img in bucket.objects.all()]
client = boto3.client('rekognition')

#%%


results_wide = []
results_long = []

for img in images:
    img_dict_wide = {'img': img}
    print(img)
    try:
        labels = client.detect_labels(Image={'S3Object':{'Bucket':bucket_name,'Name':img}},MinConfidence=75)
        if 'Labels' in labels:
            for l, label in enumerate(labels['Labels']):
                results_long.append({'img': img, 'type': 'Label', 'label': label['Name'], 
                                     'confidence': label['Confidence']})
                col = 'label_' + str(l)
                img_dict_wide[col] = label['Name']
                img_dict_wide[col + '_confidence'] = label['Confidence'] 
    except:
        continue
    try:        
        celebrities = client.recognize_celebrities(Image={'S3Object':{'Bucket':bucket_name,'Name':img}})
        if 'CelebrityFaces' in celebrities:
            for f, face in enumerate(celebrities['CelebrityFaces']):
                results_long.append({'img': img, 'type': 'Celebrity', 'label': face['Name'], 
                                     'confidence': face['Face']['Confidence']})
                col = 'celeb_' + str(f)
                img_dict_wide[col] = face['Name']
                img_dict_wide[col + '_confidence'] = face['Face']['Confidence']
    except:
        continue
    try:
        text_in_image = client.detect_text(Image={'S3Object':{'Bucket':bucket_name,'Name':img}})
        if "TextDetections" in text_in_image:
            for w, word in enumerate(text_in_image["TextDetections"]):
                results_long.append({'img': img, 'type': "Text", 'label': word["DetectedText"],
                                    'confidence': word["Confidence"]})
                col = 'word_' + str(w)
                img_dict_wide[col] = word["DetectedText"]
                img_dict_wide[col+ '_confidence'] = word["Confidence"]
    except:
        continue
            
    if 'Labels' not in labels and 'CelebrityFaces' not in celebrities and "TextDetections" not in text_in_image:
        results_long.append({'img': img, 'type': None, 'label': None, 'confidence': None})
        
    results_wide.append(img_dict_wide)
####
####
img_df_long = pd.DataFrame(results_long, columns=['img', 'type', 'label', 'confidence'])
img_df_wide = pd.DataFrame(results_wide)
cols = sorted(img_df_wide.columns)
cols.remove('img')
img_df_wide = img_df_wide[['img'] + cols]


#%% Save long/wide dataframes for topic modeling and mapping to dataframe


# For our topic modelers only focused on images data!
img_df_long.to_csv("instagram_pics_text_long_Date.csv")

# For mapping to the dataframe provided to us.
img_df_wide.to_csv("instagram_pics_text_wide_Date.csv")

#%% Create word cloud

Rekogd_images = pd.read_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/instagram_pics_text_long_11DEC2017_Final.csv' , encoding='ISO-8859-1', index_col=0)

#%% Generate word cloud for celebrities

cloud = WordCloud(background_color="white", max_words=20, stopwords=stopwords.words('english'))

positive_cloud = cloud.generate(Rekogd_images.loc[Rekogd_images.type =='Celebrity','label'].str.cat(sep='\n'))
plt.figure()
plt.imshow(positive_cloud)
plt.axis("off")
plt.show()


#%% Generate word cloud for text in the images

cloud = WordCloud(background_color="white", max_words=20, stopwords=stopwords.words('english'))

positive_cloud = cloud.generate(Rekogd_images.loc[Rekogd_images.type =='Text','label'].str.cat(sep='\n'))
plt.figure()
plt.imshow(positive_cloud)
plt.axis("off")
plt.show()


#%% Generate word cloud for image labels

cloud = WordCloud(background_color="white",max_words=20, stopwords=stopwords.words('english'))

positive_cloud = cloud.generate(Rekogd_images.loc[Rekogd_images.type =='Label','label'].str.cat(sep='\n'))
plt.figure()
plt.imshow(positive_cloud)
plt.axis("off")
plt.show()


#%% Inspect the wide data frame

# This data frame includes unique tags and their respective confidence score
# as columns. This makes it easier to merge with the mapping and master
# data frames after some data cleaning.

img_df_wide = Rekogd_images = pd.read_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/instagram_pics_text_wide_11DEC2017_Final.csv' , encoding='ISO-8859-1', index_col=0)


#%% Check dataframe head, and remove columns containing confidence metric

img_df_wide.head(3)

AWS_tags  = img_df_wide.select(lambda x: not re.search('confidence', x), axis=1)
AWS_tags.head(3)


#%% Combine all labels into one column

AWS_tags['all_tags'] = AWS_tags[AWS_tags.columns[2:]].apply(lambda x: ', '.join(x.dropna().astype(str)),axis=1)
AWS_tags['all_tags'].head(3)
AWS_tags = AWS_tags[['img','all_tags']]
AWS_tags.columns = ['image_number', 'all_tags']


#%% Join the dataframe with consolidated tags with the mapping dataframe

mapping_df_master  = pd.read_csv("/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/mappingdict_11DEC2017_Final.csv", encoding='ISO-8859-1')
mapping_df_master.columns = ['image_number', 'image_url']
mapping_df_master= pd.DataFrame(data = mapping_df_master)
mapping_df_master.head()


#%%

AWS_tags = pd.DataFrame(data = AWS_tags)
AWS_tags.head()

#%% Merge consolidated dataframe with the mapping dataframe

mapped_AWS_tags = mapping_df_master.merge(AWS_tags, how='outer')

mapped_AWS_tags.shape
mapped_AWS_tags.head()


#%% Join combined dataframe with the master Instagram dataframe

master_IG_df = pd.read_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/instagram_pic_master.csv', encoding='ISO-8859-1', index_col=0)
# master_IG_df


#%% Merge mapped data frame with the master IG data frame

IG_Data_master = master_IG_df.merge(mapped_AWS_tags, how='outer')
IG_Data_master.shape
IG_Data_master.sort_values(by=['impact'], ascending=False, inplace=True)
IG_Data_master.describe()
IG_Data_master['impact'].quantile([.25,.75])


#%% Define high and low impact Instagram posts

high_impact = IG_Data_master['impact'].quantile(.75)
low_impact = IG_Data_master['impact'].quantile(.25)


#%% Perform Topic Modeling

# Topic Modeling on Track Maven Social Media Data
#
# Ben Shaver
# December 2017

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


#%% Define LDA class

class LDA:
    '''A class that takes a list or pandas Series of strings as input and outputs a trained LDA model'''
    # Credit to Matt Brems' LDA lecture for the LDA basics
    
    def __init__(self, num_topics=5, passes=20):
        # Number of topics to find
        self.num_topics = num_topics
        
        # Number of passes over the data to make. More passes will ensure the convergence on the 'correct' 
        #  latent distribution of topics across documents and words across topics.
        self.passes = passes
        
        # Initialize the tokenizer object
        self.tokenizer = RegexpTokenizer(r'\w+')

        # Fetch an English stop words list from the NLTK package
        self.en_stop = get_stop_words('en')

        # Initialize a 'stemmer' object which will reduce words to 'stems'
        self.stemmer = PorterStemmer()    

    def transform(self, text_series):
        '''Transforms a series of texts into a dictionary and a corpus, both saved as attributes of the object'''
        self.text_series = text_series
        
        # Initialize empty list to contain tokenized strings
        tokenized_text = []
        
        # Loop through text_series
        for text in text_series:

            # Turn each string into a series of lowercase words
            raw = text.lower()
            tokens = self.tokenizer.tokenize(raw)

            # Remove stop words
            tokens = [text for text in tokens if not text in self.en_stop]

            # Turn words into 'stems,' to reduce the total number of unique words
            tokens = [self.stemmer.stem(text) for text in tokens]

            # Remove strings shorter than 4 elements
            tokens = [text for text in tokens if len(text) > 3]

            # Add tokens to list
            tokenized_text.append(tokens)

        # Create a id:term dictionary from our tokenized series of strings
        self.dictionary = corpora.Dictionary(tokenized_text)

        # Create a document-term matrix from our tokenized series of strings
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_text]   
        
     
    def train_model(self):
        '''Train the model. Uses Gensims multiple core implementation of the LDA model.''' 
        self.model = gensim.models.ldamulticore.LdaMulticore(self.corpus, num_topics=self.num_topics, id2word = self.dictionary, passes=self.passes)


#%% Functions to fetch topic lists and most represented topic


def fetch_topic_string(topic, n_words=5, join=True):
    '''Return a list of words charcterizing each topic'''
    topic_words = [lda.model.show_topic(topic)[i][0] for i in range(n_words)]
    if join:
        topic_words = ' '.join(topic_words)
    return(topic_words)

def fetch_doc_topic(document, n_words=5, num_topics=5):
    '''Return the topic most represented by a text. Minimum string length (for error handling) is 5.'''
    if type(document) != str:
        return('')
    if len(document) < 5:
        return('')
    probs = lda.model[lda.dictionary.doc2bow(document.split())]
    probs = [probs[i][1] for i in range(len(probs))]
    topic = np.argmax(probs)
    return(fetch_topic_string(topic, n_words=n_words))


#%% Read data

# Read in the data. Be careful with encoding! There are strange characters.
# Or use master Instagram data with AWS tags.

instagram = IG_Data_master
instagram.head()

#%% Safe string add function definition

def safe_string_add(*args):
    '''Safely adds multiple strings, ignores non-string inputs.'''
    string = ''
    for arg in args:
        if type(arg) == str:
            string += ' ' + arg
    return(string)        
    
#%% Create a column of all the text from each Instagram post
    
instagram['Text'] = [safe_string_add(instagram['hashtags'][i],
         instagram['all_tags'][i]) for i in range(instagram.shape[0])]

#%% Map brand dictionaries


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

IG_Data_master['brand'] = IG_Data_master['brand'].map(brands)

#%% Identify the unique brands represented

brands = instagram['brand'].unique()
brands


#%% Initialize new dataframe to store data

instagram_topics = pd.DataFrame(columns=list(instagram.columns) + ['Topic'])

# For each brand, train an LDA model and assign each observation to one of 5 topics. Append to pre-existing dataframe.
# This will take a while.

# for brand in brands:
#     try:
#         brand_data = facebook[facebook['brand_name'] == brand]

#         lda = LDA(num_topics=5, passes=20)
#         lda.transform(brand_data['Text'])
#         lda.train_model()
#         print(brand + ' analyzed.')
#         brand_data['Topic'] = [fetch_doc_topic(text, num_topics=5) for text in brand_data['Text']]
#         facebook_topics = facebook_topics.append(brand_data)
#         facebook_topics.to_csv('assets/fb_w_topics.csv')
#     except:
#         pass


#%% Do LDA on a brand

brand = 'Teen_Vogue'
brand_data = instagram[instagram['brand'] == brand]

num_topics = 5

lda = LDA(num_topics=num_topics, passes=30)
lda.transform(brand_data['Text'])
lda.train_model()

print(brand + ' analyzed.')

#%% Define function to return most-represented topic

def fetch_doc_topics(document, n_words=5, num_topics=5):
    '''Return the topic most represented by a text. Minimum string length (for error handling) is 5.'''
    if type(document) != str:
        return([1/num_topics]*num_topics)
        # If the document is not a string, there is a uniform likelihood across all topics
    if len(document) < 5:
        return([1/num_topics]*num_topics)
        # If the document is fewer than 5 characters, lets also say it could be from any topic
    probs = lda.model[lda.dictionary.doc2bow(document.split())]
    # Returns num_topics (topic,probability) tuples
    probs = [item[1] for item in probs] # Extract just the probabilities

    return(probs)

#%% Construct data frame of probabilities corresponding to each topic
    
foo = pd.DataFrame([fetch_doc_topics(doc) for doc in brand_data['Text']]) 

# Add a row identifying the most likely category for each.
foo['Main_topic'] = foo.idxmax(axis=1)

foo


#%% Reset original data frame index and concatenate column-wise

brand_data.reset_index(inplace=True) # Reset index of original data frame and:
brand_data = pd.concat([brand_data, foo], axis=1) # Concatenate together column-wise

#%% Get raw text

raw_text = [brand_data.loc[brand_data['Main_topic'] == i,'Text'].str.cat(sep = ';') for i in range(num_topics)]

#%% Import TFIDF Vectorizer and Initialize with Encoding for Emojis

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfIdf vectorizer, with encoding to handle strange values.
# We're going to look at the most 'distinctive' 2 or 3 word phrases across each
# of the raw strings with all the text from each topic.

# So, for example, we can discover that the two word phrase 'Selena Gomez' is
# very distinctive and unique to only one of the topics.

tf = TfidfVectorizer(encoding='ISO-8859-1', ngram_range=(2,4), max_features = num_topics*3)

#%% Return most distincitve 2-, 3-, or 4-word phrase for each topic
#   Sets the returned phrase as column name

topic_names = [tf.get_feature_names()[np.argmax(i)] for i in tf.fit_transform(raw_text).todense()]

#%% Assign generated topic names

topic_dict = dict({(i,topic_names[i]) for i in range(num_topics)})
brand_data = brand_data.rename(columns = topic_dict)

#%% Save brand data to CSV

brand_data.to_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/TeenVogueFB_wTopics.csv')

brand_data

#%% Import xgboost

# from xgboost import XGBClassifier
# brand_data['impact'].quantile([.3,.7])


#%% Library and function imports for count vectorization

import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#%% Define Count Vectorization function


def doCountVectorizer(data, stream, col, target, stop_wds=1, ngram=(1,1), top=25):
    
    # REQUIRED INPUTS
    #     df: Single media stream DataFrame to analyze (e.g., Facebook, Instagram)
    # stream: Media portfolio to analyze (e.g., Glamour Magazine)
    #    col: DataFrame column to run count vectorization on ("message", "caption")
    # target: Target column (likes, engagement, impact) to analyze and predict on
    #           MUST BE A COLUMN TITLE WITH TYPE INT OR FLOAT
        
    #%% Set up stop word set to add to it if desired
    
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('http')
    stopwords.append('nast')
    stopwords.append('link')
    stopwords.append('click')
    stopwords.append('asdf') # this removes the placeholder for empty posts
    
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
    df = data[data['brand']==stream].copy()
    
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
    
    return streamDF.sort_values('frequency', ascending=False)[:top]


#%% Check Instagram DF columns

instagram.columns

#%%

brands

#%% REQUIRED INPUTS FOR COUNT VECTORIZER

#     df: Single media stream DataFrame to analyze (e.g., Facebook, Instagram)
# stream: Media portfolio to analyze (e.g., Glamour Magazine)
#    col: DataFrame column to run count vectorization on ("message", "caption")
# target: Target column (likes, engagement, impact) to analyze and predict on
#           MUST BE A COLUMN TITLE WITH TYPE INT OR FLOAT

#%% Count Vectorization and plots for Vogue

Vogue = doCountVectorizer(instagram, 'Vogue', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Vogue['percentage'] = round(((Vogue['frequency']/Vogue['frequency'].sum())*100),2)
Vogue.sort_values(by='percentage', ascending=False).head(10)

import seaborn as sns

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.style.use(['fivethirtyeight'])

sns.set_style("whitegrid")

# Highest-percentage phrases
sns.barplot(x=Vogue['phrase'][0:10], y= Vogue['percentage'], data=Vogue)

# Lowest-percentage phrases
sns.barplot(x=Vogue['phrase'][-10:-1], y= Vogue['percentage'], data=Vogue)

# Lowest-frequency phrases
# ax = sns.barplot(x=Vogue['phrase'][-10:-1], y= Vogue['frequency'], data=Vogue)


#%% Clever vectorization and plots

Clever = doCountVectorizer(instagram, 'Clever', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Clever

# Highest-frequency phrases
sns.barplot(x=Clever['phrase'][0:10], y= Clever['frequency'], data=Clever)

# Lowest-frequency phrases
sns.barplot(x=Clever['phrase'][-11:-1], y= Clever['frequency'], data=Clever)


#%% Teen Vogue vectorization and plots

TeenVogue = doCountVectorizer(instagram, 'Teen_Vogue', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
TeenVogue['Percentage'] = round(((TeenVogue['frequency']/TeenVogue['frequency'].sum())*100),2)
TeenVogue.sort_values(by='Percentage', ascending=False).head(10)
#TeenVogue['High_Impact'] = [1 if TeenVogue['']]

# Highest-percentage phrases
sns.barplot(x=TeenVogue['phrase'][0:10], y= TeenVogue['percentage'], data=TeenVogue)

# Lowest-percentage phrases
sns.barplot(x=TeenVogue['phrase'][-11:-1], y= TeenVogue['percentage'], data=TeenVogue)

#%% Vanity Fair vectorization and plots

VanityFair = doCountVectorizer(instagram, 'Vanity_Fair', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
VanityFair

# Highest-frequency
sns.barplot(x=VanityFair['phrase'][0:10], y= VanityFair['frequency'], data=VanityFair)

# Lowest-frequency
sns.barplot(x=VanityFair['phrase'][-11:-1], y= VanityFair['frequency'], data=VanityFair)


#%% Conde Nast Traveler vectorization and plots

CondeNastTraveler = doCountVectorizer(instagram, 'Conde_Nast_Traveler', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
CondeNastTraveler

# Highest-frequency phrases
sns.barplot(x=CondeNastTraveler['phrase'][0:10], y= CondeNastTraveler['frequency'], data=CondeNastTraveler)

# Lowest-frequency phrases
sns.barplot(x=CondeNastTraveler['phrase'][-11:-1], y= CondeNastTraveler['frequency'], data=CondeNastTraveler)


#%% W Magazine vectorization and plots

WMagazine = doCountVectorizer(instagram, 'W_Magazine', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
WMagazine

# Highest-frequency phrases
sns.barplot(x=WMagazine['phrase'][0:10], y= WMagazine['frequency'], data=WMagazine)

# Lowest-frequency phrases
sns.barplot(x=WMagazine['phrase'][-11:-1], y= WMagazine['frequency'], data=WMagazine)


#%% Self Magazine vectorization and plots

Onself = doCountVectorizer(instagram, 'Self', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Onself

# Highest-frequency phrases
sns.barplot(x=Onself['phrase'][0:10], y= Onself['frequency'], data=Onself)

# Lowest-frequency phrases
sns.barplot(x=Onself['phrase'][-11:-1], y= Onself['frequency'], data=Onself)


#%% Glamour Magazine vectorization and plots

Glamour = doCountVectorizer(instagram, 'Glamour', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Glamour

# Highest-frequency phrases
sns.barplot(x=Glamour['phrase'][0:10], y= Glamour['frequency'], data=Glamour)

# Lowest-frequency phrases
sns.barplot(x=Glamour['phrase'][-11:-1], y= Glamour['frequency'], data=Glamour)


#%% Allure Magazine vectorization and plots

Allure = doCountVectorizer(instagram, 'Allure', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Allure

# Highest-frequency phrases
sns.barplot(x=Allure['phrase'][0:10], y= Allure['frequency'], data=Allure)

# Lowest-frequency phrases
sns.barplot(x=Allure['phrase'][-11:-1], y= Allure['frequency'], data=Allure)

