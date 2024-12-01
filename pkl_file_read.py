import pickle
import numpy as np

# read the data from the pickle file
with open('/home/ahrilab/Desktop/Data_collect/data/observations_3.pkl', 'rb') as f:
    observations = pickle.load(f)

# Print the type of the data
print(observations)
