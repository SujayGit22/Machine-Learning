import matplotlib as mt
import matplotlib.pyplot as plt

import numpy as np
fig = plt.figure()
ax = fig.add_subplot(2,1,1)

#fig_2 = plt.figure()
#ax2 = fig_2.add_axes([0.15, 0.1, 0.7, 0.3])

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2*np.pi*t)
line, = ax.plot(t, s, color='blue', lw=2)
print(ax.lines[0])
plt.show()

# Customizing your objects

print(mt.artist.getp(fig.patch))

# Figure Container
fig = plt.figure()
l1 = mt.lines.Line2D([0,1],[0,1],transform= fig.transFigure,figure=fig)
l2 = mt.lines.Line2D([0,1],[1,0],transform= fig.transFigure,figure=fig)
fig.lines.extend([l1,l2])
fig.canvas.draw()
plt.show()

ax = fig.add_subplot(111)
rect = ax.patch
rect.set_facecolor('green')
plt.show()

x, y = np.random.rand(2, 100)
line, = ax.plot(x,y,'-',color='blue',linewidth=2)
print("---Line---")
print(ax.lines)

axis = ax.xaxis
print(axis.get_ticklocs())

fig = plt.figure()
rect = fig.patch
rect.set_facecolor('lightgoldenrodyellow')

ax1= fig.add_axes([0.1,0.3,0.4,0.4])
rect = ax1.patch
rect.set_facecolor('lightslategray')

for label in ax1.xaxis.get_ticklabels():
    label.set_color('red')
    label.set_rotation(45)
    label.set_fontsize(16)

for line in ax1.yaxis.get_ticklines():
    line.set_color('green')
    line.set_markersize(25)
    line.set_markeredgewidth(3)

plt.show()

# Tick container
import matplotlib.ticker as ticker

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(100*np.random.rand(20))

formatter = ticker.FormatStrFormatter('$%1.2f')
ax.yaxis.set_major_formatter(formatter)
for tick in ax.yaxis.get_major_ticks():
    tick.label1On = False
    tick.label2On = True
    tick.label2.set_color('green')
plt.show()

# Customizing location of Subplot using GridSpec

ax1 = plt.subplot2grid((3,3),(0,0),colspan=1)
ax2 = plt.subplot2grid((3,3),(0,1),colspan=2)
ax3 = plt.subplot2grid((3,3),(1,0),colspan=3)
ax4 = plt.subplot2grid((3,3),(2,0),colspan=2)
ax5 = plt.subplot2grid((3,3),(2,2),colspan=1)
plt.show()

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3,3)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1:])
ax3 = plt.subplot(gs[1,:])
ax4 = plt.subplot(gs[2,:-1])
ax5 = plt.subplot(gs[2,-1])
plt.show()

# Adjust GridSpec layout
gs1 = gridspec.GridSpec(3,3)
gs1.update(left=0.09,right=0.48, wspace=0.09)
ax1 = plt.subplot(gs1[0,0])
ax2= plt.subplot(gs1[0,-2])
ax3 = plt.subplot(gs1[0,-1])
ax4 = plt.subplot(gs1[1,:-1])
ax5 = plt.subplot(gs1[1:,-1])
ax6 = plt.subplot(gs1[2,0])
ax7= plt.subplot(gs1[2,-2])
plt.show()

# GridSpec using subplotspec
gs = gridspec.GridSpec(1,2)
gs0 = gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=gs[0])
gs1 = gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=gs[1])

ax1 = plt.subplot(gs0[0,0])
ax2= plt.subplot(gs0[0,-2])
ax3 = plt.subplot(gs0[0,-1])
ax4 = plt.subplot(gs0[1,:-1])
ax5 = plt.subplot(gs0[1:,-1])
ax6 = plt.subplot(gs0[2,0])
ax7= plt.subplot(gs0[2,-2])

ax1 = plt.subplot(gs1[0,0])
ax2= plt.subplot(gs1[0,-2])
ax3 = plt.subplot(gs1[0,-1])
ax4 = plt.subplot(gs1[1,:-1])
ax5 = plt.subplot(gs1[1:,-1])
ax6 = plt.subplot(gs1[2,0])
ax7= plt.subplot(gs1[2,-2])
plt.show()

# Grid Spec with varying cell sizes

gsV = gridspec.GridSpec(2,2,width_ratios=(1,2),height_ratios=(3,2))

ax1 = plt.subplot(gsV[0])
ax2 = plt.subplot(gsV[1])
ax3 = plt.subplot(gsV[2])
ax4 = plt.subplot(gsV[3])
plt.show()