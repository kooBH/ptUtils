import torch
import torch.nn as nn
import numpy as np
from scipy import signal

import numpy as np
from scipy.fft import fft, ifft
import librosa as rs
import soundfile as sf
from scipy import signal
from numpy.lib.stride_tricks import as_strided
from scipy.fftpack import fft, ifft

"""
S. Wang, G. Naithani, A. Politis and T. Virtanen,
"Deep Neural Network Based Low-Latency Speech Separation with Asymmetric Analysis-Synthesis Window Pair,"
2021 29th European Signal Processing Conference (EUSIPCO), Dublin, Ireland, 2021, pp. 301-305, doi: 10.23919/EUSIPCO54536.2021.9616165.
keywords: {Deep learning;Training;Time-frequency analysis;Source separation;Signal processing algorithms;Europe;Speech enhancement;Monaural speaker separation;Low latency;Asymmetric windows;Deep clustering},

REF : https://ieeexplore.ieee.org/abstract/document/9616165
REF : https://github.com/shanwangshan/asymmetric_window/blob/main/Asymmetric_window_on_oracle.ipynb
"""
def hann_window(M):
    window = torch.zeros(M)
    # (1)
    for n in range(M) : 
        window[n] = 0.5 * (1 - torch.cos(torch.tensor(2*torch.pi * n / M)))
    return window

def analysis_window(n_fft,winsize_long,winsize_short,hopsize) : 
    K = winsize_long
    M = winsize_short//2; # 64
    d = hopsize; #64
    #R = hopsize # frame advance
    
    pro_1 = K-M-d
    pro_2 = M
    win_pro_1 = hann_window(2*pro_1)
    win_pro_2 = hann_window(2*pro_2)

    # (3,4)
    win_an_1=torch.zeros((d))
    win_an_2=torch.sqrt(win_pro_1[:K-2*M-d])
    win_an_3=torch.sqrt(win_pro_1[K-2*M-d:K-2*M-d+M])
    # (2)
    win_an_4=torch.sqrt(win_pro_2[M:M+M])
    
    window_analysis = torch.cat((win_an_1,win_an_2,win_an_3,win_an_4))
    return window_analysis

def synthesis_widnow(n_fft,winsize_long,winsize_short,hopsize) :
    K = winsize_long
    M = winsize_short//2; # 64
    d = hopsize; #64
    
    pro_1 = K-M-d
    pro_2 = M
    win_pro_1 = hann_window(2*pro_1)
    win_pro_2 = hann_window(2*pro_2)

    # (5)
    win_sy_3=(win_pro_2[:M])/torch.sqrt(win_pro_1[K-2*M-d:K-2*M-d+M])
    # (2)
    win_sy_4=torch.sqrt(win_pro_2[M:M+M])
    
    window_synthesis = torch.cat((win_sy_3,win_sy_4)) 
    
    return window_synthesis

# STFT
def stft_asymmetric(x, frame_size, hop_size, window):
    n_frames = (len(x) - frame_size) // hop_size + 1
    stft_frames = []
    for i in range(n_frames):
        frame = x[i * hop_size: i * hop_size + frame_size] * window
        spectrum = torch.fft.rfft(frame)
        stft_frames.append(spectrum)
    stft_frames = torch.stack(stft_frames)
    return stft_frames

# iSTFT
def istft_asymmetric(stft_frames, frame_size, hop_size, window):
    K = frame_size
    M = hop_size
    winsize_short = len(window)
    print(stft_frames.shape)
    stft_frames = torch.transpose(stft_frames,0,1)
    print(stft_frames.shape)
    
    n_frames = stft_frames.shape[1]
    signal_length = (n_frames - 1) * hop_size + frame_size
    output = torch.zeros((n_frames*hop_size))
    for i in range(n_frames-1):
        a = stft_frames[:,i]
        b = torch.conj(torch.flip(stft_frames[:,i],[0])[1:])
        c = torch.cat((a,b))
       
        out_long = torch.fft.irfft(c)
        out_long_wnd = out_long[K-winsize_short:K]
        #out_long_wnd = out_long[-window_synthesis.shape[0]:]

        out_long_wnd = window*out_long_wnd
        output[i*hop_size:i*hop_size+2*M]=output[i*hop_size:i*hop_size+2*M]+ out_long_wnd

    return output

if __name__ == "__main__" : 
    # Test
    x = torch.randn(16000)
    n_fft = 256
    winsize_short = 128 # 8ms
    winsize_long = 256 # 16ms
    hop_size = 64

    # Analysis Phase
    analysis = analysis_window(n_fft,winsize_long,winsize_short,hop_size)
    synthesis = synthesis_widnow(n_fft,winsize_long,winsize_short,hop_size)

    X = stft_asymmetric(x,  n_fft,hop_size, analysis)
    y = istft_asymmetric(X, n_fft, hop_size, synthesis)

    print(f"{x.shape}  | {y.shape}")

    x = x[winsize_long-winsize_short:len(y)]
    y = y[:len(x)]
    print(f"{torch.sum(torch.abs(x - y))}")