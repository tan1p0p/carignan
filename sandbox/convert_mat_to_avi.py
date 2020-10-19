import os
import cv2
import numpy as np
from src.data_handling import ready_mov_file

out_path, dim, Y = ready_mov_file(
    './data/raw/20191017_130713/20191017_130713.mat',
    'video', 500, return_mov_data=True)
os.remove(out_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    './data/interim/20191017_130713/20191017_130713.avi',
    fourcc, 10.0, (dim[1], dim[0]))

for _ in range(10):
    for frame_ in Y:
        frame_ = frame_.astype('float')
        frame_ = frame_ - frame_.min()
        frame_ *= 255.0 / frame_.max()
        frame = np.stack((frame_, frame_, frame_)).transpose(1,2,0)
        out.write(frame.astype('uint8'))

# Release everything if job is finished
out.release()
