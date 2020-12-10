import cv2
import h5py
import numpy as np
from scipy import interpolate

with h5py.File('data/interim/DM108/DM108_video.h5', 'r') as f:
    movie = f['Object'][()]

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(
#     './data/interim/DM108/DM108_comp.avi',
#     fourcc, 10.0, (movie.shape[2], movie.shape[1]))

for frame_ in movie:
    value_index = {}
    frame_pixels = sorted(list(frame_.flatten()))
    max_i = len(frame_pixels)
    for i, v in enumerate(frame_pixels):
        value_index[v] = i / max_i * 255
        
    mp = np.arange(0, frame_.max()+1)
    mp[list(value_index.keys())] = list(value_index.values())
    frame_ = mp[frame_].astype('uint8')

    frame = np.stack((frame_, frame_, frame_)).transpose(1,2,0)
    cv2.imshow('frame', frame)
    # out.write(frame)
    cv2.waitKey(1)
