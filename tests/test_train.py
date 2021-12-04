import math
x = [math.sin(i/20)+i/300 for i in range(600)]
from uniplot import plot
x = [1, 2]
y = [10, 20]
plot(y, x, x_min=1, x_gridlines=[-1], x_max=3, y_max=100, lines=True, title="Test accuracy v/s Epochs")
