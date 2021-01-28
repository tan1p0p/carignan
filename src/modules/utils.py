import logging

import numpy as np

def show_logs():
    logging.basicConfig(
        format="%(relativeCreated)12d [%(filename)s:%(funcName)10s():%(lineno)s] [%(process)d] %(message)s",
        level=logging.DEBUG
    )

def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

class HeatMap():
    def __init__(self, figsize, use_plt=False):
        self.figsize = figsize
        self.use_plt = use_plt
        if self.use_plt:
            fig = plt.figure(figsize=(figsize[0]/100, figsize[1]/100), dpi=100) # If dpi=1, a fontsize error will occur.
            self.ax = fig.add_subplot(1, 1, 1)
            self.canvas = FigureCanvasAgg(fig)
            self.ax.xaxis.tick_top()
            self.ax.invert_yaxis()

    def __normalize(self, np_tensor, axis=None):
        d_min = np_tensor.min(axis=axis, keepdims=True)
        d_max = np_tensor.max(axis=axis, keepdims=True)
        return (np_tensor - d_min) / (d_max - d_min)

    # takes too much time
    def __get_heatmap_in_plt(self, data):
        self.ax.clear()
        self.ax.pcolor(data.T, cmap=plt.cm.Blues)
        s, (width, height) = self.canvas.print_to_buffer()
        image = np.fromstring(s, dtype='uint8').reshape((height, width, 4))[:, :, :3]
        return image

    def __get_heatmap_in_cv2(self, data):
        data = (self.__normalize(data, 0)*255).astype('uint8')
        heatmap = cv2.resize(data.T, self.figsize, interpolation=cv2.INTER_NEAREST)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        return heatmap

    def get_heatmap(self, data):
        if self.use_plt:
            return self.__get_heatmap_in_plt(data)
        else:
            return self.__get_heatmap_in_cv2(data)
