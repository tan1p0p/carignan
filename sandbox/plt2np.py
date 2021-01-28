import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

x = np.linspace(0,20,100) # 描画用サンプルデータ
y = x ** 0.5              # 描画用サンプルデータ

fig = plt.figure(figsize=(3.21, 3), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.plot(x,y)
canvas = FigureCanvasAgg(fig)
s, (width, height) = canvas.print_to_buffer()
image = np.frombuffer(s, dtype='uint8').reshape((height, width, 4))
print(image.shape)