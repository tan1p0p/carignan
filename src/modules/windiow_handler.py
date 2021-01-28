import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
matplotlib.use('Agg')
import numpy as np

class Window():
    def __init__(self, video_bit):
        self.__init_status(video_bit)

    def __init_status(self, video_bit):
        self.window_name = 'microscope CNMF-E'

        # seekbar texts
        self.gain_text = 'gain'
        self.fps_text = '0: 05fps\n1: 10fps\n2: 15fps\n3: 30fps\n4: 60fps'
        self.x0_text, self.x1_text = 'set_x0', 'set_x1'
        self.y0_text, self.y1_text = 'set_y0', 'set_y1'
        self.dr_max_text = 'dynamic range: max'
        self.dr_min_text = 'dynamic range: min'
        self.start_text = 'start analysis!'
        self.demixed_bias_text = 'demixed color bias'

        # params
        h, w = frame_shape
        self.fps = 30
        self.demixed_bias = 1
        self.x0 = 0
        self.x1 = w
        self.y0 = 0
        self.y1 = h
        self.dr_min = 0
        if video_bit == 'uint8':
            self.dr_max = 255
        else:
            self.dr_max = 1000
            # self.dr_max = 65535

    def update(self):
        pass

class HeatMap():
    def __init__(self, figsize):
        self.figsize = figsize

        fig = plt.figure(figsize=(figsize[0]/100, figsize[1]/100), dpi=100) # If dpi=1, a fontsize error will occur.
        self.ax = fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasAgg(fig)
        self.ax.xaxis.tick_top()
        self.ax.invert_yaxis()

    def __normalize(self, np_tensor, axis=None):
        d_min = np_tensor.min(axis=axis, keepdims=True)
        d_max = np_tensor.max(axis=axis, keepdims=True)
        return (np_tensor - d_min) / (d_max - d_min)

    def get_heatmap_in_plt(self, data):
        self.ax.clear()
        self.ax.pcolor(data.T, cmap=plt.cm.Blues)
        s, (width, height) = self.canvas.print_to_buffer()
        image = np.fromstring(s, dtype='uint8').reshape((height, width, 4))[:, :, :3]
        return image

    def get_heatmap_in_cv2(self, data):
        data = (self.__normalize(data, 0)*255).astype('uint8')
        heatmap = cv2.resize(data.T, self.figsize, interpolation=cv2.INTER_NEAREST)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        return heatmap
