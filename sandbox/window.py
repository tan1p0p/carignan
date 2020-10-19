import time
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
window_name = 'prev'
cv2.namedWindow(window_name)

# TODO: https://github.com/bothlab/pomidaq/blob/f8f1ce1703d4c8af1041c740c832d85eaeb2961a/libminiscope/miniscope.cpp#L485
# create switch for choose GAIN
def set_GAIN(x):
    gain = [16, 32, 64]
    print(cap.set(14, gain[x]))
    time.sleep(0.01)
    print(f'gain: {cap.get(14)}')
gain_text = 'gain'
cv2.createTrackbar(gain_text, window_name, 0, 2, set_GAIN)

# create switch for choose FPS
def set_FPS(x):
    fps = [10, 30, 60]
    print(cap.set(5, fps[x]))
    time.sleep(0.01)
    print(f'fps: {cap.get(5)}')
fps_text = '0: 10fps\n1: 30fps\n2: 60fps'
cv2.createTrackbar(fps_text, window_name, 0, 2, set_FPS)

def set_aaa(x, y):
    print(x, y)
    print('press button')
cv2.createButton('start analysis!', set_aaa)

while True:
    ret, frame = cap.read()
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.getTrackbarPos(gain_text, window_name)
    cv2.getTrackbarPos(fps_text, window_name)

cv2.destroyAllWindows()