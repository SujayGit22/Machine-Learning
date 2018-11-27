import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.random.rand(10))
def onclick(event):
    print(event)
    print(event.button,event.x,event.y,event,event.xdata,event.ydata)
cid = fig.canvas.mpl_connect('button_press_event',onclick)
plt.show()

#Example
