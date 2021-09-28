import torch

# SI-SDR(Scale Invariant Source-to-Distortion Ratio)

# (2019,ICASSP)SDR – Half-baked or Well Done?
# https://ieeexplore.ieee.org/abstract/document/8683855

# NOTE ::  SDR == SI-SDR 

class SDR : 
    def __init__(self,device,n_fft=1024):
        self.n_fft = n_fft
        self.device = device

        self.window = torch.hann_window(window_length=self.n_fft,periodic=True, dtype=None, 
                           layout=torch.strided, device=device, requires_grad=False)

    def istft(self,x):
        return torch.istft(x, self.n_fft, hop_length=None, win_length=None, window=self.window, center=True, normalized=False, onesided=None, length=None, return_complex=False)

    # Based on https://github.com/sigsep/bsseval/issues/3
    def SISDR(self,output, target, inSTFT=True):
        if inSTFT : 
            output = self.istft(output)
            target = self.istft(target)

        Rss= torch.dot(target, target)
            
        e_target= (torch.dot( target, output) / Rss) * target
        e_res= e_target - output

        Sss= (e_target**2).sum()
        Snn= (e_res**2).sum()

        # SDR on dB scale
        #SISDR= 10 * torch.log10(Sss/Snn)
        SISDR= Sss/Snn
        
        # Get the SIR
        # Rsr= np.dot(target.transpose(), e_res)
        # b= np.linalg.solve(Rss, Rsr)

        #e_interf= np.dot(target, b)
        #e_artif= e_res - e_interf
        
        #SIR= 10 * math.log10(Sss / (e_interf**2).sum())
        #SAR= 10 * math.log10(Sss / (e_artif**2).sum())

        #return SDR, SIR,SAR
        return SISDR

    # to compare
    def SDRLoss(self,output,target, inSTFT=True, eps=2e-7):
        if inSTFT : 
            output = self.istft(output)
            target = self.istft(target)

        xy = torch.diag(output @ target.t())
        yy = torch.diag(target @ target.t())
        xx = torch.diag(output @ output.t())

        SDR = xy**2/ (yy*xx - xy**2 )
        return torch.mean(SDR)
    
    def iSDRLoss(self,output,target, inSTFT=True, eps=2e-7):
        sdr = self.SDRLoss(output,target,inSTFT,eps)
        return 1/sdr

    # mSDR is not Scale Invariant
    def mSDRLoss(self,output,target, inSTFT=True, eps=2e-7):
        if inSTFT : 
            output = self.istft(output)
            target = self.istft(target)
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        #xx = torch.dot(output,output)
        #xy = torch.dot(output,target)

        #return xx/(xy**2)
        correlation = torch.sum(target * output, dim=1)
        energies = torch.norm(target, p=2, dim=1) * torch.norm(output, p=2, dim=1)

        return torch.mean(-(correlation / (energies + eps)))
    
    def wSDRLoss(self,output,noisy,target,alpha=0.9,inSTFT=True,eps=2e-7):
        if inSTFT : 
            output = self.istft(output)
            target = self.istft(target)
            noisy  = self.istft(noisy)
        noise = noisy - target
        noise_est = noisy - output

        wSDR = alpha* self.mSDRLoss(output,target,inSTFT=False,eps=eps) + (1-alpha)*self.mSDRLoss(noise_est,noise,inSTFT=False,eps=eps)
        return wSDR
    
    def STFTSDRLoss(self,output,target):
        # [B, F, T, C]
        diff = torch.abs(output - target)        
