import torch
import torch.nn.functional as F
import numpy as np

class AsymSTFT :
    def __init__(self, n_fft, n_hop): 
        self.n_fft = n_fft
        self.n_hop = n_hop

        self.analysis = self.make_analysis_window(n_fft, n_hop*2, n_hop)
        self.synthesis = self.make_synthesis_window(n_fft, n_hop*2, n_hop)


    def hann_window(self,M):
        window = torch.zeros(M)
        # (1)
        for n in range(M) : 
            window[n] = 0.5 * (1 - torch.cos(torch.tensor(2*torch.pi * n / M)))
        return window 

    def make_analysis_window(self,n_long, n_short, n_hop) : 

        K = n_long
        M = n_short//2
        d = n_hop

        pro_1 = K-M-d
        pro_2 = M
        win_pro_1 = self.hann_window(2*pro_1)
        win_pro_2 = self.hann_window(2*pro_2)

        # (3,4)
        win_an_1=torch.zeros((d))
        win_an_2=torch.sqrt(win_pro_1[:K-2*M-d])
        win_an_3=torch.sqrt(win_pro_1[K-2*M-d:K-2*M-d+M])
        # (2)
        win_an_4=torch.sqrt(win_pro_2[M:M+M])
        
        window = torch.cat((win_an_1,win_an_2,win_an_3,win_an_4))
        return window

    def make_synthesis_window(self,n_long, n_short, n_hop) :
        K = n_long 
        M = n_short//2
        d = n_hop
        
        pro_1 = K-M-d
        pro_2 = M
        win_pro_1 = self.hann_window(2*pro_1)
        win_pro_2 = self.hann_window(2*pro_2)

        # (5)
        win_sy_3=(win_pro_2[:M])/torch.sqrt(win_pro_1[K-2*M-d:K-2*M-d+M])
        # (2)
        win_sy_4=torch.sqrt(win_pro_2[M:M+M])
        
        window= torch.cat((win_sy_3,win_sy_4)) 
        
        return window

    def STFT(self,x) : 
        # x : [B,L]
        x = torch.tensor(x)
        B,L = x.shape
        n_frames = (x.shape[1] - self.n_fft) // self.n_hop + 1
        
        stft_shape = (B, self.n_fft//2+1, n_frames)
        stft_frames = torch.zeros(stft_shape,dtype=torch.complex64, device = x.device)

        self.analysis = self.analysis.to(x.device)
        for j in range(B) : 
            for i in range(n_frames):
                frame = x[j,i * self.n_hop: i * self.n_hop+ self.n_fft] * self.analysis
                spectrum = torch.fft.rfft(frame)
                stft_frames[j,:,i] = spectrum
        return stft_frames

    def iSTFT(self,X, len_signal=-1) : 
        K = self.n_fft
        M = self.n_hop
        winsize_short = len(self.synthesis)
        B,n_freq,T = X.shape

        if len_signal == -1 :
            signal_length = (T- 1) * self.n_hop+ self.n_fft
        else :
            signal_length = len_signal

        signal_shape = (B,signal_length)

        output_buffer = torch.zeros(signal_shape,device = X.device)

        self.synthesis = self.synthesis.to(X.device)

        for j in range(B) : 
            for i in range(T-1):
                a = X[j,:,i]
                out_long = torch.fft.irfft(a)
                out_long_wnd = out_long[K-winsize_short:K]
                out_long_wnd = self.synthesis*out_long_wnd

                output_buffer[j,i*self.n_hop:i*self.n_hop+2*M]=output_buffer[j,i*self.n_hop:i*self.n_hop+2*M]+ out_long_wnd

        # algin with input signal
        output_buffer = F.pad(output_buffer,(K - winsize_short,0),"constant",0)
        output_buffer = output_buffer[:,:signal_length]

        return output_buffer