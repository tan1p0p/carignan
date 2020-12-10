import h5py

class H5VideoHandler():
    def __init__(self, mov_path, mov_key):
        with h5py.File(mov_path, 'r') as f:
            self.mov_data = f[mov_key][()]
        self.video_bit = 'uint16'
        self.i = 0

    def read(self):
        try:
            frame = self.mov_data[self.i]
            self.i += 1
            return True, frame
        except:
            return False, None

class CV2VideoHandler():
    def __init__(self, id_or_path):
        self.cap = cv2.VideoCapture(input_camera_id)
        self.video_bit = 'uint8'

    def read(self):
        try:
            _, bgr_frame = self.cap.read()
            frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            return True, frame
        except:
            return False, None