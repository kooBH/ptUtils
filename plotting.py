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

# Not working
def spec2plot(spectrogram):
    if len(np.shape(spectrogram)) == 3 :
        spectrogram = np.power(2,spectrogram[:,:,0]) + np.power(2,spectrogram[:,:,1])
        spectrogram = 10*np.log(spectrogram)
    else :
        raise ValueError("not implemented yet")

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


