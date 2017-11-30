# GA-TrackMaven-03

Data we will be using is [here](https://s3.amazonaws.com/temp-data-pulls/newdump.json). Download it into your own 'assets' folder. Since it is a large file lets not actually keep it in GitHub. But our scripts will point to it.

[00_Importing_Data.py](https://github.com/smasschelin/GA-TrackMaven-03/blob/master/00_Importing_Data.py) - Loads in the JSON, subsets only the FB and IG data, and saves it as a csv in the assets folder.

01_