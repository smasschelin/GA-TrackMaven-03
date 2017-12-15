# GA-TrackMaven-03

Data we will be using is [here](https://s3.amazonaws.com/temp-data-pulls/newdump.json).

[00_Importing_Data.py](https://github.com/smasschelin/GA-TrackMaven-03/blob/master/00_Importing_Data.py) - Loads in the JSON, subsets only the FB and IG data, and saves it as a csv in the assets folder.

[01_Instagram_Data_Clean.py](https://github.com/smasschelin/GA-TrackMaven-03/blob/master/01_Instagram_Data_Clean.py) -- Cleans Instagram data

[02_Clean_Facebook_Data.py](https://github.com/smasschelin/GA-TrackMaven-03/blob/master/02_Clean_Facebook_Data.py) -- Cleans Facebook data

[03_Count_Vectorization.py](https://github.com/smasschelin/GA-TrackMaven-03/blob/master/03_Count_Vectorization.py) -- Natural Language Processing

[Topic_Modeling_Final.ipynb](https://github.com/smasschelin/GA-TrackMaven-03/blob/master/Topic_Modeling_Final.ipynb) -- An explanation of the Latent Dirichlet Allocation model for topic modeling, plus notes on how to automate the process and some next steps for making predictions based on topic distributions

[Topic_Modeling_Final.py](https://github.com/smasschelin/GA-TrackMaven-03/blob/master/Topic_Modeling_Final.py) -- Same as above but as plain .py file.
