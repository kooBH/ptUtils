import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
import io
import PIL.Image
from torchvision.transforms import ToTensor
import librosa as rs

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
    n_shape = len(data.shape)


    # [ l ] -> wav
    if n_shape == 1 :
        data = rs.stft(data,n_fft = 512)
        mag= np.abs(data)
    # [2, F, T] -> spec
    elif data.shape[0] == 2 :
        data = data[0] + data[1]*1j
        mag = np.abs(data)
    # or [F, T] with complex
    elif np.iscomplex(data).any() : 
        mag = np.abs(data)
    else : 
        mag = data
    # data is mag

    np.seterr(divide = 'warn') 

    mag = 10*np.log(mag)
    fig, ax = plt.subplots()
    im = plt.imshow(mag, cmap=cm.jet, aspect='auto',origin='lower')
    plt.colorbar(im)
    plt.clim(-80,20)
    
    plt.xlabel('Time')
    plt.ylabel('Freq')
    
    fig.canvas.draw()
    plot = fig2np(fig)
    return plot

def mag2plot(data):
    mag = data.detach().cpu().numpy()
    mag = 10*np.log(mag)
    fig, ax = plt.subplots()
    im = plt.imshow(mag, cmap=cm.jet, aspect='auto',origin='lower')
    plt.colorbar(im)
    plt.clim(-80,20)
    
    plt.xlabel('Time')
    plt.ylabel('Freq')
    
    fig.canvas.draw()
    plot = fig2np(fig)
    return plot

"""
wav2plotDOA : 
    wavfrom 2 tensorboard image
    Impelemented for DOA-Separation

input
    + waveform : numpy array[9,n_sample]
"""
def wav2plotDOA(waveform, sample_rate=16000):
    num_channels = waveform.shape[0]
    num_frames = waveform.shape[1]
    time_axis = np.arange(start=0, stop=num_frames) / sample_rate

    figure, axes = plt.subplots(2, 4)

    ## input plotting routine 
    #gs = axes[0,0].get_gridspec()
    #for ax in axes[0,:]:
    #    ax.remove()
    # big = figure.add_subplot(gs[0,:])
    # big.set_title('input')
    # big.plot(time_axis, waveform[0], linewidth=1)
        
    
    for c in range(4):
        idx_y = 0
        idx_x = c
        
        axes[idx_y,idx_x].plot(time_axis, waveform[0+c], linewidth=1)
        axes[idx_y,idx_x].grid(True)
        axes[idx_y,idx_x].set_title(f'target {c}')
        
    for c in range(4):
        idx_y = 1
        idx_x = c
        
        axes[idx_y,idx_x].plot(time_axis, waveform[4+c], linewidth=1)
        axes[idx_y,idx_x].grid(True)
        axes[idx_y,idx_x].set_title(f'output {c}')
        
    figure.set_size_inches(14, 10)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image
