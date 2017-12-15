
# coding: utf-8

# In[1]:


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

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.style.use(['fivethirtyeight'])


# In[2]:


# Topic modeling is a form of unsupervised learning. Like other unsupervised learning techniques, it learns patterns in a set of unlabelled 
# data, or data without a target variable to be predicted. In this case, the patterns learned are latent 'topics' that appear in a set of 
# texts, or documents. 
# Latent Dirichlet Allocation, or LDA, is a form of topic modeling that assumes documents are just bags of words, ignoring syntax and grammar.
# LDA assumes documents are a mix of topics. Each word in a document belongs to each topic with a fixed probabiltiy unique to that document, and
# each topic in turn returns a given word with a set of unique probabilities unique to that topic.
# The purpose of LDA is to approximate the assumed 'latent' distribution which represents the mix of topics across documents AND the mix of
# words across topics. Once the LDA model is trained, it can be used to compute the mix of topics for a particular document, and to compute
# the mix of words per topic. Note that each word is not unique to a topic, but merely more likely to appear for a given topic.
# 
# Below, I wrap the Python package Gensim's functionality into a class and some helper functions in order to train a model on social media
# data from Track Maven. For each brand within the Conde Nast umbrella, LDA trains a model on the unique corpus of its facebook posts
# (combining the different text fields first). Then a topic is assigned to each observation according to the topic most highly represnted
# by that post. Finally, a CSV file is saved which is identical to the file read in except 'text' and 'topic' columns have been added.


# In[3]:


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

        # Save texts for later
        self.texts = tokenized_text
            
        # Create a id:term dictionary from our tokenized series of strings
        self.dictionary = corpora.Dictionary(tokenized_text)

        # Create a document-term matrix from our tokenized series of strings
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_text]   
        
     
    def train_model(self):
        '''Train the model. Uses Gensims multiple core implementation of the LDA model.''' 
        self.model = gensim.models.ldamulticore.LdaMulticore(self.corpus, num_topics=self.num_topics, id2word = self.dictionary, passes=self.passes)


# In[4]:


def fetch_topic_string(topic, n_words=5, join=True):
    '''Return a list of words charcterizing each topic'''
    topic_words = [lda.model.show_topic(topic)[i][0] for i in range(n_words)]
    if join:
        topic_words = ' '.join(topic_words)
    return(topic_words)

def fetch_doc_topics(document, num_topics=5):
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


# In[5]:


# Read in the data. Be careful with encoding! There are strange characters.
facebook = pd.read_csv('assets/facebook_data.csv', encoding='ISO-8859-1', index_col=0)


# In[6]:


def safe_string_add(*args):
    '''Safely adds multiple strings, ignores non-string inputs.'''
    string = ''
    for arg in args:
        if type(arg) == str:
            string += ' ' + arg
    return(string)        
    
# Create a column of all the text from each FB post
facebook['Text'] = [safe_string_add(facebook['media_title'][i],
         facebook['message'][i]) for i in range(facebook.shape[0])]

# If you'd like to do a similar analysis for Instagram data, 
# simply import the IG data and combine all text fields in a 'Text' column.


# In[98]:


# Identify the unique brands represented
brands = facebook['brand_name'].unique()

# Replace 'facebook' with 'intsagram' here and below...


# In[99]:


# Initiate new dataframe to store data.
facebook_topics = pd.DataFrame(columns=list(facebook.columns) + ['Topic'])

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


# In[100]:


# Teen Vogue has the most FB posts, by a significant margin. Let's focus on Teen Vogue, and try and determine how to
# develop distinct topics (not too many, not too few) and a sensible name for a topic.

# May take about 20 minutes with 20 passes. 

# Using 3 topics based on topic coherence metric. See below.

brand = 'Teen_Vogue'
brand_data = facebook[facebook['brand_name'] == brand]

num_topics = 3

lda = LDA(num_topics=num_topics, passes=20)
lda.transform(brand_data['Text'])

lda.train_model()

print(brand + ' analyzed.')


# In[101]:


# Construct data frame of probabilities corresponding to each topic:
foo = pd.DataFrame([fetch_doc_topics(doc) for doc in brand_data['Text']]) 
# Add a row identifying the most likely category for each.
foo['Main_topic'] = foo.idxmax(axis=1)


# In[102]:


brand_data.reset_index(inplace=True) # Reset index of original data frame and:
brand_data = pd.concat([brand_data, foo], axis=1) # Concatenate together column-wise


# In[103]:


# Now we have a dataframe representing the original Teen Vogue data, plus additional columns for each topic
# that LDA found, and a column representing the 'main topic' for each post, we want to provide some label to the topics


# In[104]:


# Group by main topics and concatenate all text into a list of raw text for each topics.
raw_text = [brand_data.loc[brand_data['Main_topic'] == i,'Text'].str.cat(sep = ';') for i in range(num_topics)]


# In[105]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[106]:


# Initialize TfIdf vectorizer, with encoding to handle strange values.
# We're goint to look at the most 'distinctive' 2 or 3 word phrases across each of the raw strings with all the text from each topic
# So, for example, we can discover that the two word phrase 'Selena Gomez' is very distinctive and unique to only one of the topics.
tf = TfidfVectorizer(encoding='ISO-8859-1', ngram_range=(2,4), max_features = num_topics*3)


# In[107]:


# Returns the most 'distinctive' 2-, 3-, or 4- word phrase for each topic, and sets as column name
topic_names = [tf.get_feature_names()[np.argmax(i)] for i in tf.fit_transform(raw_text).todense()]


# In[108]:


# Assign generated topic names (if they're any good)

topic_dict = dict({(i,topic_names[i]) for i in range(num_topics)})

topic_dict = dict({0:'Topic 1', 1:'Topic 2', 2:'Topic 3'})

brand_data = brand_data.rename(columns = topic_dict)


# In[109]:


# Are these 2-4 word ngrams good topic names? Maybe we would be better off going with the first option, just 
# using the top outputed words per topic:
[fetch_topic_string(i, n_words=10) for i in range(num_topics)]


# In[110]:


brand_data.to_csv('assets/'+brand+'_FB_w3Topics.csv')


# In[111]:


# Iterate through a range of possible values for number of topics, so we can identify which maximizes topic coherence.

# from gensim.models.coherencemodel import CoherenceModel

# coherence_scores = []
# for i in range(2,10):
#     brand = 'Teen_Vogue'
#     brand_data = facebook[facebook['brand_name'] == brand]

#     num_topics = i

#     lda = LDA(num_topics=num_topics, passes=20)
#     lda.transform(brand_data['Text'])

#     lda.train_model()
#     cm = CoherenceModel(model=lda.model, texts=lda.texts, dictionary=lda.dictionary, coherence='u_mass')
#     coherence_scores.append(cm.get_coherence())


# In[112]:


cm = CoherenceModel(model=lda.model, texts=lda.texts, dictionary=lda.dictionary, coherence='u_mass')
print(cm.get_coherence())


# In[113]:


# How do we model with LDA results?


# In[116]:


brand_data.drop([3,4], axis=1, inplace=True) # Since we only ended up using 3 topics


# In[146]:


model_data.head(10)


# In[56]:


# As a regression problem:


# In[148]:


brand_data.columns


# In[57]:


from sklearn.linear_model import LinearRegression


# In[176]:


model_data = brand_data[['Topic 1','Topic 2','Topic 3','impact']]
model_data.dropna(inplace=True)
model_data = model_data[model_data['Topic 1'] != model_data['Topic 2']]
X = model_data[['Topic 1','Topic 2','Topic 3']]
y = model_data['impact']


# In[177]:


lr = LinearRegression()
lr.fit(X, y)
lr.coef_


# In[178]:


# Not a very good R-squared. Unfortunately just 3 topics by themselves can't predict impact as a linear process.
lr.score(X, y)


# In[179]:


# Lets take the log of the impact values and predict that instead.
y = np.log(y)
lr = LinearRegression()
lr.fit(X, y)
lr.coef_


# In[180]:


# Our R-squared is an order of magnitude better, but still bad.
lr.score(X, y)


# In[181]:


# As a classification task:
from sklearn.linear_model import LogisticRegressionCV


# In[182]:


# Build target variable here. We will just predict if post is above or below the median level of impact.
discrete_y = [1 if x > y.median() else 0 for x in y]


# In[183]:


# Not a great accuracy. But a direction of influence can still be inferred, below:

logreg = LogisticRegressionCV()
logreg.fit(X, discrete_y)
logreg.score(X, discrete_y)


# In[184]:


logreg.coef_

