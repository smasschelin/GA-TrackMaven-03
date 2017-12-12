
# coding: utf-8

# # Amazon Rekognition – Image Detection and Recognition 

# In[2]:


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


# In[3]:


# Read in CSVs that contain columns of url images and their IDs.
ig_urls_pics = pd.read_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/instagram_pic_urls.csv', encoding='ISO-8859-1', index_col=0)


# In[4]:


ig_urls_pics.head(3)


# In[5]:


ig_urls_pics.shape


# # Amazon Image Rekognition

# In[6]:


# Create a list of urls from instagram image urls
image_list = ig_urls_pics.image_url


# In[7]:


# Uses the creds in ~/.aws/credentials
s3 = boto3.resource('s3')
bucket_name_to_upload_image_to = 'igimages'


# In[8]:


# Do this as a quick and easy check to make sure your S3 access is OK
for bucket in s3.buckets.all():
    if bucket.name == bucket_name_to_upload_image_to:
        print('Good to go. Found the bucket to upload the image into.')
        good_to_go = True

if not good_to_go:
    print('Not seeing your s3 bucket, might want to double check permissions in IAM')


# In[ ]:


# Code from Natalie Olivo's Medium post. 
# https://medium.com/@NatalieOlivo/use-python-to-collect-image-tags-using-aws-reverse-image-search-engine-rekognition-eccf1f259a8d
# This following will allow user to upload pics to a bucket on Amazon AWS verses saving it their local machine.

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


# In[ ]:


# Save down your mapping dict so that you can eventually re-map your image tags to your full dataframe.
mapping_dict = pd.DataFrame(mapping_dict, index = range(0,len(mapping_dict)))
mapping_dict = pd.DataFrame(mapping_dict.T[0])
mapping_dict.to_csv('mappingdict_Date.csv')


# In[ ]:


# Creates both wide and long df's with image tags from Rekognition:
bucket_name = 'igimages'
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)
images = [img.key for img in bucket.objects.all()]
client = boto3.client('rekognition')


# In[ ]:


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


# In[ ]:


# For our topic modelers only focused on images data!
img_df_long.to_csv("instagram_pics_text_long_Date.csv")

# For mapping to the dataframe provided to us.
img_df_wide.to_csv("instagram_pics_text_wide_Date.csv")


# # Create a word cloud

# In[14]:


Rekogd_images = pd.read_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/instagram_pics_text_long_11DEC2017_Final.csv' , encoding='ISO-8859-1', index_col=0)


# Generate word cloud for celebrities

# In[15]:


cloud = WordCloud(background_color="white", max_words=20, stopwords=stopwords.words('english'))

positive_cloud = cloud.generate(Rekogd_images.loc[Rekogd_images.type =='Celebrity','label'].str.cat(sep='\n'))
plt.figure()
plt.imshow(positive_cloud)
plt.axis("off")
plt.show()


# Generate word cloud for text in the images

# In[16]:


cloud = WordCloud(background_color="white", max_words=20, stopwords=stopwords.words('english'))

positive_cloud = cloud.generate(Rekogd_images.loc[Rekogd_images.type =='Text','label'].str.cat(sep='\n'))
plt.figure()
plt.imshow(positive_cloud)
plt.axis("off")
plt.show()


# Generate word cloud for image labels

# In[17]:


cloud = WordCloud(background_color="white",max_words=20, stopwords=stopwords.words('english'))

positive_cloud = cloud.generate(Rekogd_images.loc[Rekogd_images.type =='Label','label'].str.cat(sep='\n'))
plt.figure()
plt.imshow(positive_cloud)
plt.axis("off")
plt.show()


# In[19]:


# Inspect the wide data frame.  This data frame includes unique tags as and their respective confidence score
# as columns.  This will make it easier to merge with the mapping and master data frames after some data cleaning.
img_df_wide = Rekogd_images = pd.read_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/instagram_pics_text_wide_11DEC2017_Final.csv' , encoding='ISO-8859-1', index_col=0)


# In[20]:


img_df_wide.head(3)


# Remove all of the columns that contain the confidence metric

# In[21]:


AWS_tags  = img_df_wide.select(lambda x: not re.search('confidence', x), axis=1)


# In[22]:


AWS_tags.head(3)


# Combine all lables into one column

# In[23]:


AWS_tags['all_tags'] = AWS_tags[AWS_tags.columns[2:]].apply(lambda x: ', '.join(x.dropna().astype(str)),axis=1)


# In[24]:


AWS_tags['all_tags'].head(3)


# In[25]:


AWS_tags = AWS_tags[['img','all_tags']]


# In[26]:


AWS_tags.columns = ['image_number', 'all_tags']


# Join df with consolidated tags with mapping df

# In[36]:


mapping_df_master  = pd.read_csv("/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/mappingdict_11DEC2017_Final.csv", encoding='ISO-8859-1')


# In[38]:


mapping_df_master.columns = ['image_number', 'image_url']


# In[29]:


mapping_df_master= pd.DataFrame(data = mapping_df_master)
mapping_df_master.head()


# In[40]:


AWS_tags = pd.DataFrame(data = AWS_tags)
AWS_tags.head()


# In[41]:


# Merge consolidated df with the mapping df.
mapped_AWS_tags = mapping_df_master.merge(AWS_tags, how='outer')


# In[42]:


mapped_AWS_tags.shape


# In[43]:


mapped_AWS_tags.head()


# Join combined df with the master IG df

# In[44]:


master_IG_df = pd.read_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/instagram_pic_master.csv', encoding='ISO-8859-1', index_col=0)


# In[48]:


# master_IG_df


# In[49]:


#Merge mapped data frame with the master IG data frame.
IG_Data_master = master_IG_df.merge(mapped_AWS_tags, how='outer')


# In[50]:


IG_Data_master.shape


# In[51]:


IG_Data_master.sort_values(by=['impact'], ascending=False, inplace=True)


# In[52]:


IG_Data_master.describe()


# In[55]:


IG_Data_master['impact'].quantile([.25,.75])
b


# In[59]:


# Define high and low impact. 
high_impact = IG_Data_master['impact'].quantile(.75)
low_impact = IG_Data_master['impact'].quantile(.25)


# # Perform Topic Modeling

# In[60]:


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


# In[61]:


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


# In[62]:


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


# In[63]:


# Read in the data. Be careful with encoding! There are strange characters.
# Or use master Instagram data with AWS tags.
instagram = IG_Data_master


# In[64]:


instagram.head()


# In[65]:


def safe_string_add(*args):
    '''Safely adds multiple strings, ignores non-string inputs.'''
    string = ''
    for arg in args:
        if type(arg) == str:
            string += ' ' + arg
    return(string)        
    
# Create a column of all the text from each Instagram post
instagram['Text'] = [safe_string_add(instagram['hashtags'][i],
         instagram['all_tags'][i]) for i in range(instagram.shape[0])]


# In[66]:


brands = {137314 : 'CondeNasteTraveler', 
          137329 : 'WMagazine',
          137321 : 'Onself',
          137325 : 'VanityFair', 
          137300 : 'Clever', 
          137322 : 'TeenVogue', 
          137299 : 'Allure', 
          137326 : 'Vogue',137316 : 'Glamor'
         }
IG_Data_master['brand'] = IG_Data_master['brand'].map(brands)   


# In[67]:


# Identify the unique brands represented
brands = instagram['brand'].unique()


# In[68]:


brands


# In[69]:


# Initiate new dataframe to store data.
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


# In[70]:


brand = 'TeenVogue'
brand_data = instagram[instagram['brand'] == brand]

num_topics = 5

lda = LDA(num_topics=num_topics, passes=30)
lda.transform(brand_data['Text'])
lda.train_model()

print(brand + ' analyzed.')


# In[71]:


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

# Construct data frame of probabilities corresponding to each topic:
foo = pd.DataFrame([fetch_doc_topics(doc) for doc in brand_data['Text']]) 
# Add a row identifying the most likely category for each.
foo['Main_topic'] = foo.idxmax(axis=1)


# In[ ]:


foo


# In[72]:


brand_data.reset_index(inplace=True) # Reset index of original data frame and:
brand_data = pd.concat([brand_data, foo], axis=1) # Concatenate together column-wise


# In[73]:


raw_text = [brand_data.loc[brand_data['Main_topic'] == i,'Text'].str.cat(sep = ';') for i in range(num_topics)]


# In[74]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[75]:


# Initialize TfIdf vectorizer, with encoding to handle strange values.
# We're goint to look at the most 'distinctive' 2 or 3 word phrases across each of the raw strings with all the text from each topic
# So, for example, we can discover that the two word phrase 'Selena Gomez' is very distinctive and unique to only one of the topics.
tf = TfidfVectorizer(encoding='ISO-8859-1', ngram_range=(2,4), max_features = num_topics*3)


# In[76]:


# Returns the most 'distinctive' 2-, 3-, or 4- word phrase for each topic, and sets as column name
topic_names = [tf.get_feature_names()[np.argmax(i)] for i in tf.fit_transform(raw_text).todense()]


# In[77]:


# Assign generated topic names.
topic_dict = dict({(i,topic_names[i]) for i in range(num_topics)})
brand_data = brand_data.rename(columns = topic_dict)


# In[78]:


brand_data.to_csv('/Users/Chudi8GB/Desktop/REPII/1.10-pandas-ii/TeenVogueFB_wTopics.csv')


# In[79]:


brand_data


# In[80]:


# # Import xgboost
# from xgboost import XGBClassifier


# In[81]:


# brand_data['impact'].quantile([.3,.7])


# In[83]:


import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[84]:


def doCountVectorizer(data, stream, col, target, stop_wds=1, ngram=(1,1), top=25):
    
    #%% Check inputs for proper dtype
    
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


# In[85]:


instagram.columns


# In[86]:


# REQUIRED INPUTS
#     df: Single media stream DataFrame to analyze (e.g., Facebook, Instagram)
# stream: Media portfolio to analyze (e.g., Glamour Magazine)
#    col: DataFrame column to run count vectorization on ("message", "caption")
# target: Target column (likes, engagement, impact) to analyze and predict on
#           MUST BE A COLUMN TITLE WITH TYPE INT OR FLOAT


# In[87]:


# Run vectorizer for brands and compare Top 10 and Bottom 10 with bar charts.


# In[88]:


brands


# In[89]:


Vogue = doCountVectorizer(instagram, 'Vogue', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Vogue['percentage'] = round(((Vogue['frequency']/Vogue['frequency'].sum())*100),2)
Vogue.sort_values(by='percentage', ascending=False).head(10)


# In[90]:


import seaborn as sns

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.style.use(['fivethirtyeight'])

sns.set_style("whitegrid")
sns.barplot(x=Vogue['phrase'][0:10], y= Vogue['percentage'], data=Vogue)


# In[ ]:


sns.barplot(x=Vogue['phrase'][-10:-1], y= Vogue['percentage'], data=Vogue)


# In[ ]:


# ax = sns.barplot(x=Vogue['phrase'][-10:-1], y= Vogue['frequency'], data=Vogue)


# In[ ]:


Clever = doCountVectorizer(instagram, 'Clever', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Clever


# In[ ]:


sns.barplot(x=Clever['phrase'][0:10], y= Clever['frequency'], data=Clever)


# In[ ]:


sns.barplot(x=Clever['phrase'][-11:-1], y= Clever['frequency'], data=Clever)


# In[91]:


TeenVogue = doCountVectorizer(instagram, 'TeenVogue', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
TeenVogue['Percentage'] = round(((TeenVogue['frequency']/TeenVogue['frequency'].sum())*100),2)
TeenVogue.sort_values(by='Percentage', ascending=False).head(10)
TeenVogue['High_Impact'] = [1 if TeenVogue['']]


# In[ ]:


sns.barplot(x=TeenVogue['phrase'][0:10], y= TeenVogue['percentage'], data=TeenVogue)


# In[ ]:


sns.barplot(x=TeenVogue['phrase'][-11:-1], y= TeenVogue['percentage'], data=TeenVogue)


# In[ ]:


VanityFair = doCountVectorizer(instagram, 'VanityFair', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
VanityFair


# In[ ]:


sns.barplot(x=VanityFair['phrase'][0:10], y= VanityFair['frequency'], data=VanityFair)


# In[ ]:


sns.barplot(x=VanityFair['phrase'][-11:-1], y= VanityFair['frequency'], data=VanityFair)


# In[ ]:


CondeNasteTraveler = doCountVectorizer(instagram, 'CondeNasteTraveler', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
CondeNasteTraveler


# In[ ]:


sns.barplot(x=CondeNasteTraveler['phrase'][0:10], y= CondeNasteTraveler['frequency'], data=CondeNasteTraveler)


# In[ ]:


sns.barplot(x=CondeNasteTraveler['phrase'][-11:-1], y= CondeNasteTraveler['frequency'], data=CondeNasteTraveler)


# In[ ]:


WMagazine = doCountVectorizer(instagram, 'WMagazine', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
WMagazine


# In[ ]:


sns.barplot(x=WMagazine['phrase'][0:10], y= WMagazine['frequency'], data=WMagazine)


# In[ ]:


sns.barplot(x=WMagazine['phrase'][-11:-1], y= WMagazine['frequency'], data=WMagazine)


# In[ ]:


Onself = doCountVectorizer(instagram, 'Onself', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Onself


# In[ ]:


sns.barplot(x=Onself['phrase'][0:10], y= Onself['frequency'], data=Onself)


# In[ ]:


sns.barplot(x=WMagazine['phrase'][-11:-1], y= WMagazine['frequency'], data=WMagazine)


# In[ ]:


Glamor = doCountVectorizer(instagram, 'Glamor', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Glamor


# In[ ]:


sns.barplot(x=Glamor['phrase'][0:10], y= Glamor['frequency'], data=Glamor)


# In[ ]:


sns.barplot(x=Glamor['phrase'][-11:-1], y= Glamor['frequency'], data=Glamor)


# In[ ]:


Allure = doCountVectorizer(instagram, 'Allure', 'Text','impact', stop_wds=1, ngram=(1,1), top=25)
Allure


# In[ ]:


sns.barplot(x=Allure['phrase'][0:10], y= Allure['frequency'], data=Allure)


# In[ ]:


sns.barplot(x=Allure['phrase'][-11:-1], y= Allure['frequency'], data=Allure)

