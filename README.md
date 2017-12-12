# GA-TrackMaven-03

Data we will be using is [here](https://s3.amazonaws.com/temp-data-pulls/newdump.json). Download it into your own 'assets' folder. Since it is a large file lets not actually keep it in GitHub. But our scripts will point to it.

[00_Importing_Data.py](https://github.com/smasschelin/GA-TrackMaven-03/blob/master/00_Importing_Data.py) - Loads in the JSON, subsets only the FB and IG data, and saves it as a csv in the assets folder.

01_Instagram_Data_Clean.py -- Cleans Instagram data

02_Clean_Facebook_Data.py -- Cleans Facebook data

03_Count_Vectorization.py -- Natural Language Processing

Topic_Modeling_Final.ipynb -- An explanation of the Latent Dirichlet Allocation model for topic modeling, plus notes on how to automate the process and some next steps for making predictions based on topic distributions
