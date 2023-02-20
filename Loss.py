import torch
import torch.nn as nn
import librosa

"""
SI-SDR(Scale Invariant Source-to-Distortion Ratio)
== SI-SNR
(2019,ICASSP)SDR – Half-baked or Well Done?
https://ieeexplore.ieee.org/abstract/document/8683855

Based on https://github.com/sigsep/bsseval/issues/3
"""
def SISDRLoss(output, target):
    # scaling factor 
    alpha =  torch.mul(output,target)/torch.sum(target**2)

    numer =  torch.sum((alpha*target)**2)
    denom =  torch.sum((output-alpha*target)**2)
    #dB scale
    sdr = 20*torch.log10(numer/denom)

    return -sdr

# mSDR == CosineSimilarity
# mSDR is not Scale Invariant
def mSDRLoss(output,target, eps=1e-7):
    # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
    # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
    #  > Maximize Correlation while producing minimum energy output.
    #xx = torch.dot(output,output)
    #xy = torch.dot(output,target)

    #return xx/(xy**2)
    correlation = torch.sum(target * output, dim=1)
    energies = torch.norm(target, p=2, dim=1) * torch.norm(output, p=2, dim=1)

    return torch.mean(-(correlation / (energies + eps)))

def wSDRLoss(output,noisy,target,alpha=0.01,inSTFT=True,eps=2e-7):
        noise = noisy - target
        noise_est = noisy - output

        if alpha == -1 : 
            alpha = torch.sum(torch.pow(target,2)) / (torch.sum(torch.pow(target,2)) + torch.sum(torch.pow(noise,2))+ eps)
        wSDR = alpha * mSDRLoss(output,target,eps=eps) + (1-alpha)*mSDRLoss(noise_est,noise,eps=eps)
        return wSDR

def SDR(output,target, eps=2e-7):
    xy = torch.diag(output @ target.t())
    yy = torch.diag(target @ target.t())
    xx = torch.diag(output @ output.t())

    SDR = xy**2/ (yy*xx - xy**2 )
    return torch.mean(SDR)

def iSDRLoss(output,target, eps=2e-7):
    sdr = SDR(output,target,eps)
    return 1/sdr

def logSDRLoss(output,target, eps=2e-7):
    return SDR(output,target,eps)

def wMSELoss(output,target,alpha=0.9,eps=1e-13):
    s_mag = torch.abs(target)
    s_hat_mag = torch.abs(output)

    # scale
    s_mag= torch.log10(1+s_mag)
    s_hat_mag= torch.log10(1+s_hat_mag)
 
    s_mag = s_mag/torch.max(s_mag)
    s_hat_mag = s_hat_mag/torch.max(s_hat_mag)

    d = s_mag - s_hat_mag

    return torch.mean(alpha*(d + d.abs())/2 + (1-alpha) * (d-d.abs()).abs()/2)


"""
    Mel-domain Weighted Error
    output : STFT [B,F,T]
    target : STFT [B,F,T]
"""

mel_basis = None

def mwMSELoss(output,target,alpha=0.99,eps=1e-7,sr=16000,n_fft=512,device="cuda:0"):
    global mel_basis

    if mel_basis is None :
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft,n_mels=40)
        mel_basis = torch.from_numpy(mel_basis)
        mel_basis = mel_basis.to(device)

    # ERROR : weight becomes NAN
    # --> torch backpropagation weight clipping on 'sqrt'
    #    --> omitted 'sqrt'
    #s_mag = torch.sqrt(target[:,:,:,0]**2 + target[:,:,:,1]**2)
    #s_hat_mag = torch.sqrt(output[:,:,:,0]**2 + output[:,:,:,1]**2)

    # add eps due to 'MmBackward nan' error in gradient
    s_hat_mag = torch.abs(output) + eps
    s_mag = torch.abs(target) + eps

    # scale
    s_mag= torch.log10(1+s_mag)
    s_hat_mag= torch.log10(1+s_hat_mag)

    # mel
    s = torch.matmul(mel_basis,s_mag)
    s_hat = torch.matmul(mel_basis,s_hat_mag)

    # Batch Norm
    s_mag = s_mag/torch.max(s_mag)
    s_hat_mag = s_hat_mag/torch.max(s_hat_mag)

    d = s - s_hat

    mwMSE = torch.mean(alpha *(d + d.abs())/2 + (1-alpha) * (d-d.abs()).abs()/2)

    return  mwMSE

class LossBundle:
    def __init__(self,hp,device):
        self.hp = hp
        self.n_fft = hp.audio.frame
        self.n_mels = hp.audio.n_mels
        self.device = device
        self.alpha_wMSE = hp.loss.wMSE.alpha
        self.alpha_mwMSE = hp.loss.mwMSE.alpha
        self.beta_mwMSE_iSDR = hp.loss.mwMSE_iSDR.beta

        self.window = torch.hann_window(window_length=self.n_fft,periodic=True, dtype=None, 
                           layout=torch.strided, device=device, requires_grad=False)

        self.mel_basis = librosa.filters.mel(sr=16000, n_fft=self.n_fft,n_mels=self.n_mels)
        self.mel_basis = torch.from_numpy(self.mel_basis)
        self.mel_basis = self.mel_basis.to(self.device)

        self.eps = 1e-7

    ## Aux

    def stft(self,x):
        return torch.stft(x, self.n_fft, hop_length=None, win_length=None, window=self.window, center=True, normalized=False, onesided=None, length=None, return_complex=False)

    def istft(self,x,inReal=True):
        if inReal : 
            # real,imag to complex
            # [n_batch, n_fft, n_frame, real imag]
            x = torch.complex(x[:,:,:,0],x[:,:,:,1])
    # input (Tensor) –
    #
    # The input tensor. Expected to be output of stft(), 
    # can either be complex (channel, fft_size, n_frame), 
    # or real (channel, fft_size, n_frame, 2) where the channel dimension is optional.
    # 
    # Deprecated since version 1.8.0: Real input is deprecated, use complex inputs as returned by stft(..., return_complex=True) instead.
        return torch.istft(x, self.n_fft, hop_length=None, win_length=None, window=self.window, center=True, normalized=False, onesided=None, length=None, return_complex=False)

    ## LOSS

    # SI-SDR(Scale Invariant Source-to-Distortion Ratio)

    # (2019,ICASSP)SDR – Half-baked or Well Done?
    # https://ieeexplore.ieee.org/abstract/document/8683855

    # Based on https://github.com/sigsep/bsseval/issues/3
    def SISDR(self,output, target, inSTFT=True):
        if inSTFT : 
            output = self.istft(output)
            target = self.istft(target)

        # scaling actor 
        alpha =  torch.dot(output,target)/torch.sum(target**2)

        numer =  torch.sum((alpha*target)**2)
        denom =  torch.sum((alpha*target-output)**2)

        loss = 10*torch.log10(numer/denom)

        return loss

    # to compare
    def SDRLoss(self,output,target, inSTFT=True, eps=2e-7):
        if inSTFT : 
            if output.shape[-1] == 2 :
                output = self.istft(output,inReal=True)
            else :
                output = self.istft(output,inReal=False)
            if target.shape[-1] == 2 :
                target = self.istft(target,inReal=True)
            else :
                target = self.istft(target,inReal=False)

        xy = torch.diag(output @ target.t())
        yy = torch.diag(target @ target.t())
        xx = torch.diag(output @ output.t())

        SDR = xy**2/ (yy*xx - xy**2 )
        return torch.mean(SDR)
    
    def iSDRLoss(self,output,target, inSTFT=True, eps=2e-7):
        sdr = self.SDRLoss(output,target,inSTFT,eps)
        return 1/sdr

    # mSDR == CosineSimilarity
    # mSDR is not Scale Invariant
    def mSDRLoss(self,output,target, inSTFT=True, eps=2e-7):
        if inSTFT : 
            if output.shape[-1] == 2 :
                output = self.istft(output,inReal=True)
            else :
                output = self.istft(output,inReal=False)
            if target.shape[-1] == 2 :
                target = self.istft(target,inReal=True)
            else :
                target = self.istft(target,inReal=False)

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

    def wMSE(self,output,target,inSTFT=True):
        if not inSTFT : 
            #s_abs     = target.abs()
            #s_hat_abs = output.abs()
            d = target - output
        else : 
            # if not in magnitude
            # add eps due to 'MmBackward nan' error in gradient
            if target.shape[-1] == 2 :
                s_mag = target[:,:,:,0]**2 + target[:,:,:,1]**2 + self.eps
            else :
                s_mag = target
            if output.shape[-1] == 2 :
                s_hat_mag = output[:,:,:,0]**2 + output[:,:,:,1]**2 + self.eps
            else :
                s_hat_mag = output

        # scale
        if self.hp.loss.wMSE.scale == 'log' : 
            s_mag= torch.log10(s_mag)
            s_hat_mag= torch.log10(s_hat_mag)
        elif self.hp.loss.wMSE.scale == 'log+1' : 
            s_mag= torch.log10(1+s_mag)
            s_hat_mag= torch.log10(1+s_hat_mag)
        elif self.hp.loss.wMSE.scale == 'dB' : 
            s_mag= 10*torch.log10(s_mag)
            s_hat_mag= 10*torch.log10(s_hat_mag)
        elif self.hp.loss.wMSE.scale == 'none' : 
            pass
        else :
            raise Exception('Unknown scale type : '+ str(self.hp.loss.wMSE.scale))

        # norm
        if self.hp.loss.wMSE.norm == 'none' : 
            pass
        elif self.hp.loss.wMSE.norm == 'norm_freq_max':
            # norm  for each freq bin
            s_mag = s_mag /torch.max(s_mag,dim=(1))[0].view(s_mag.shape[0],s_mag.shape[1],1)
            s_hat_mag = s_hat_mag/torch.max(s_hat_mag,dim=(1))[0].view(s_hat_mag.shape[0],s_hat_mag.shape[1],1)
        elif self.hp.loss.wMSE.norm == 'norm_max_batch':
            s_mag = s_mag/torch.max(s_mag)
            s_hat_mag = s_hat_mag/torch.max(s_hat_mag)

        d = s_mag - s_hat_mag

        return torch.mean(self.alpha_wMSE *(d + d.abs())/2 + (1-self.alpha_wMSE) * (d-d.abs()).abs()/2)
    
    # Mel-domain Weighted Error
    def mwMSE(self,output,target,inSTFT=True):
        if not inSTFT : 
            output = self.stft(output)
            target = self.stft(target)

        # ERROR : weight becomes NAN
        # --> torch backpropagation weight clipping on 'sqrt'
        #    --> omitted 'sqrt'
        #s_mag = torch.sqrt(target[:,:,:,0]**2 + target[:,:,:,1]**2)
        #s_hat_mag = torch.sqrt(output[:,:,:,0]**2 + output[:,:,:,1]**2)

        # if not in magnitude
        # add eps due to 'MmBackward nan' error in gradient
        if target.shape[-1] == 2 :
            s_mag = target[:,:,:,0]**2 + target[:,:,:,1]**2 + self.eps
        else :
            s_mag = target
        if output.shape[-1] == 2 :
            s_hat_mag = output[:,:,:,0]**2 + output[:,:,:,1]**2 + self.eps
        else :
            s_hat_mag = output

        # scale
        if self.hp.loss.mwMSE.scale == 'log' : 
            s_mag= torch.log10(s_mag)
            s_hat_mag= torch.log10(s_hat_mag)
        elif self.hp.loss.mwMSE.scale == 'log+1' : 
            s_mag= torch.log10(1+s_mag)
            s_hat_mag= torch.log10(1+s_hat_mag)
        elif self.hp.loss.mwMSE.scale == 'dB' : 
            s_mag= 10*torch.log10(s_mag)
            s_hat_mag= 10*torch.log10(s_hat_mag)
        elif self.hp.loss.mwMSE.scale == 'none' : 
            pass
        else :
            raise Exception('Unknown scale type : '+ str(self.hp.loss.mwMSE.scale))

        # mel
        s = torch.matmul(self.mel_basis,s_mag)
        s_hat = torch.matmul(self.mel_basis,s_hat_mag)

        if self.hp.loss.mwMSE.norm == 'none' : 
            pass
        elif self.hp.loss.mwMSE.norm == 'norm_freq_max':
              # norm  for each freq bin
            s_mag = s_mag /torch.max(s_mag,dim=(1))[0].view(s_mag.shape[0],s_mag.shape[1],1)
            s_hat_mag = s_hat_mag/torch.max(s_hat_mag,dim=(1))[0].view(s_hat_mag.shape[0],s_hat_mag.shape[1],1)
        elif self.hp.loss.mwMSE.norm == 'norm_max_batch':
            s_mag = s_mag/torch.max(s_mag)
            s_hat_mag = s_hat_mag/torch.max(s_hat_mag)


        d = s - s_hat

        mwMSE = torch.mean(self.alpha_mwMSE *(d + d.abs())/2 + (1-self.alpha_mwMSE) * (d-d.abs()).abs()/2)

        #print('-----')
        #print(output)
        #print('mwMSE : '+ str(mwMSE))

        return  mwMSE

    def mwMSE_iSDR(self,output,target,inSTFT=True, ):
        if not inSTFT :
            wav_output = output
            wav_target = target
            spec_output = self.stft(output) 
            spec_target =  self.stft(target)
        else :
            spec_output = output
            spec_target = target
            wav_output =  self.istft(output)
            wav_target =  self.istft(target)

        beta = self.beta_mwMSE_iSDR
        
        l = beta*self.mwMSE(spec_output,spec_target) + (1-beta)*self.iSDRLoss(wav_output,wav_target,inSTFT=False)

        return l
