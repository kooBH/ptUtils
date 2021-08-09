import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm

def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def MFCC2plot(MFCC):
    MFCC = np.transpose(MFCC)
    fig, ax = plt.subplots()
    im = plt.imshow(MFCC, cmap=cm.jet, aspect='auto')
    plt.colorbar(im)
    plt.clim(-80,20)
    fig.canvas.draw()

    data = fig2np(fig)
    plt.close()
    return data

def spec2plot(data,normalized=True):
    data = data.detach().cpu().numpy()
    spec = np.power(data[:,:,0],2) + np.power(data[:,:,1],2)
    spec = 10*np.log(spec)
    fig, ax = plt.subplots()
    im = plt.imshow(spec, cmap=cm.jet, aspect='auto',origin='lower')
    plt.colorbar(im)
    plt.clim(-80,20)
    
    plt.xlabel('Time')
    plt.ylabel('Freq')
    
    fig.canvas.draw()
    plot = fig2np(fig)
    return plot


