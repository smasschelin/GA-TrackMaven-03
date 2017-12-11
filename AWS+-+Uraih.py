
# coding: utf-8

# # Amazon Rekognition – Image Detection and Recognition 

# In[11]:


import boto
import boto3
conn = boto.connect_s3()
import requests
import pandas as pd
import numpy as np


# In[12]:


# Read in CSVs from Steve
ig_urls_pics = pd.read_csv('/Users/Chudi8GB/Downloads/InstaPicsImageURLS.csv',names = ['url'])
ig_urls_vids = pd.read_csv('/Users/Chudi8GB/Downloads/InstaVidsVideoURLS.csv',names = ['url'])


# In[13]:


ig_urls_pics.head(3)


# In[14]:


ig_urls_pics.shape


# In[15]:


ig_urls_vids.head(3)


# In[16]:


ig_urls_vids.shape


# In[17]:


# Create a list of urls for instagram pics
image_list = ig_urls_pics.url


# In[18]:


len(image_list)


# In[19]:


# Uses the creds in ~/.aws/credentials
s3 = boto3.resource('s3')
bucket_name_to_upload_image_to = 'trackmavenig'


# In[20]:


# Do this as a quick and easy check to make sure your S3 access is OK
for bucket in s3.buckets.all():
    if bucket.name == bucket_name_to_upload_image_to:
        print('Good to go. Found the bucket to upload the image into.')
        good_to_go = True

if not good_to_go:
    print('Not seeing your s3 bucket, might want to double check permissions in IAM')


# In[22]:


# Code from Natalie's Medium post.
# Allows user to upload pics to a bucket on Amazon AWS verses saving it their local machine.

mapping_dict ={}
# for i, img_url in enumerate(image_list[0:10000]):
for i, img_url in enumerate(image_list[0:100]):
    
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


# # Save down your mapping dict so that you can eventually re-map your image tags to your full dataframe.

# In[24]:


mapping_dict = pd.DataFrame(mapping_dict, index = range(0,len(mapping_dict)))
# mapping_dict = pd.DataFrame(md_01.T[0])
mapping_dict.to_csv('mappingdict.csv')


# # Creates both wide and long df's with image tags from Rekognition:

# In[ ]:


bucket_name = 'trackmavenig'
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)
images = [img.key for img in bucket.objects.all()]
client = boto3.client('rekognition')

results_wide = []
results_long = []

for img in images:
    img_dict_wide = {'img': img}
    #print(img)
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
# cols.remove('img')
img_df_wide = img_df_wide[['img'] + cols]


# # Save down your dfs.
# 

# In[ ]:


#For our topic modelers only focused on images data!
img_df_long.to_csv("instagram_pics_text_long.csv")

#For mapping to the dataframe provided to us.
img_df_wide.to_csv("instagram_pics_text_wide.csv")

