import cv2

def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('R','image', 0, 255,nothing)

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    cv2.imshow('image', frame)

    r = cv2.getTrackbarPos('R','image')
    print(r)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

params = ['MSEC',
        'POS_FRAMES',
        'POS_AVI_RATIO',
        'FRAME_WIDTH',
        'FRAME_HEIGHT',
        'PROP_FPS',
        'PROP_FOURCC',
        'FRAME_COUNT',
        'FORMAT',
        'MODE',
        'BRIGHTNESS',
        'CONTRAST',
        'SATURATION',
        'HUE',
        'GAIN',
        'EXPOSURE',
        'CONVERT_RGB',
        'WHITE_BALANCE',
        'RECTIFICATION']
current_params = []
for num in range(19):
    current_params.append(cap.get(num))
    print(params[num], ':', cap.get(num))

['PROP_FPS', '']
for num in [9, 10, 12, 13, 14]:
    print(params[num], ':', cap.set(num, current_params[num]))

cap.release()