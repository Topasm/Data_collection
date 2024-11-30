import zarr
import numpy as np

# Open the Zarr file
zarr_file = zarr.open(
    '/home/ahrilab/Desktop/Data_collection/data/replay_buffer.zarr', mode='r')

# Accessing the timestamp data
timestamp_data = zarr_file['data/robot_eef_pose'][:]

# Set print options to display the entire array
np.set_printoptions(threshold=np.inf, precision=20, suppress=False)

# Print all timestamp data
print("Timestamp Data: ", timestamp_data)


# print data length
print("Data Length: ", len(timestamp_data))
