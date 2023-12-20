import torch
import torch.nn as nn
import librosa

EPS=1e-7
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

"""

"""
class CosSDRLossSegment(nn.Module):
    """
    It's a cosine similarity between predicted and clean signal
        loss = - <y_true, y_pred> / (||y_true|| * ||y_pred||)
    This loss function is always bounded between -1 and 1
    Ref: https://openreview.net/pdf?id=SkeRTsAcYm
    Hyeong-Seok Choi et al., Phase-aware Speech Enhancement with Deep Complex U-Net,
    """
    def __init__(self, reduction=torch.mean):
        super(CosSDRLossSegment, self).__init__()
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):
        num = torch.sum(target * output, dim=-1)
        den = torch.norm(target, dim=-1) * torch.norm(output, dim=-1)
        loss_per_element = -num / (den + EPS)
        loss = self.reduction(loss_per_element)
        return {"CosSDRLossSegment": loss} if out_dict else loss

class CosSDRLoss(nn.Module):
    def __init__(self, reduction=torch.mean):
        super(CosSDRLoss, self).__init__()
        self.segment_loss = CosSDRLossSegment(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target, chunk_size=1024, out_dict=True):
        out_chunks = torch.reshape(output, [output.shape[0], -1, chunk_size])
        trg_chunks = torch.reshape(target, [target.shape[0], -1, chunk_size])
        loss_per_element = torch.mean(
            self.segment_loss(out_chunks, trg_chunks, False), dim=-1
        )
        loss = self.reduction(loss_per_element)
        return {"CosSDRLoss": loss} if out_dict else loss

class MultiscaleCosSDRLoss(nn.Module):
    def __init__(self, chunk_sizes, reduction=torch.mean):
        super(MultiscaleCosSDRLoss, self).__init__()
        self.chunk_sizes = chunk_sizes
        self.loss = CosSDRLoss(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):
        loss_per_scale = [
            self.loss(output, target, cs, False) for cs in self.chunk_sizes
        ]
        loss_per_element = torch.mean(torch.stack(loss_per_scale), dim=0)
        loss = self.reduction(loss_per_element)
        return {"MultiscaleCosSDRLoss": loss} if out_dict else loss


class SpectrogramLoss(nn.Module):
    def __init__(self, reduction=torch.mean):
        super(SpectrogramLoss, self).__init__()
        self.gamma = 0.3
        self.reduction = reduction

    def forward(self, output, target, chunk_size=1024, out_dict=True):
        # stft.shape == (batch_size, fft_size, num_chunks)
        stft_output = torch.stft(output, chunk_size, hop_length=chunk_size//4, return_complex=True, center=False)
        stft_target = torch.stft(target, chunk_size, hop_length=chunk_size//4, return_complex=True, center=False)

        # clip is needed to avoid nan gradients in the backprop
        mag_output = torch.clip(torch.abs(stft_output), min=EPS)
        mag_target = torch.clip(torch.abs(stft_target), min=EPS)
        distance = mag_target**self.gamma - mag_output**self.gamma

        # average out
        loss_per_chunk = torch.mean(distance**2, dim=1)
        loss_per_element = torch.mean(loss_per_chunk, dim=-1)
        loss = self.reduction(loss_per_element)
        return {"SpectrogramLoss": loss} if out_dict else loss

class MultiscaleSpectrogramLoss(nn.Module):
    def __init__(self, chunk_sizes, reduction=torch.mean):
        super(MultiscaleSpectrogramLoss, self).__init__()
        self.chunk_sizes = chunk_sizes
        self.loss = SpectrogramLoss(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):
        loss_per_scale = [
            self.loss(output, target, cs, False) for cs in self.chunk_sizes
        ]
        loss_per_element = torch.mean(torch.stack(loss_per_scale), dim=0)
        loss = 15*self.reduction(loss_per_element)
        return {"MultiscaleSpectrogramLoss": loss} if out_dict else loss

class TrunetLoss(nn.Module):
    def __init__(self, frame_size_sdr=[4096, 2048, 1024, 512], frame_size_spec=[1024, 512, 256]):
        super(TrunetLoss, self).__init__()

        
        self.max_size = max(max(frame_size_sdr), max(frame_size_spec))

        self.sdr_loss = MultiscaleCosSDRLoss(frame_size_sdr)
        self.spc_loss = MultiscaleSpectrogramLoss(frame_size_spec)

    def forward(self, outputs, targets, out_dict=False):
        # shape: (batch_size, direct_or_reverberant, num_samples)
        yd = outputs
        td = targets

        if yd.shape[1]% self.max_size != 0:
            yd = yd[..., : -(yd.shape[-1] % self.max_size)]
            td = td[..., : -(td.shape[-1] % self.max_size)]

        # d=direct, r=reverberant; reverb = reverberant - direct
        # fmt: off
        losses = {
            "MultiscaleSpectrogramLoss_Direct":      self.spc_loss(yd, td, out_dict=False),
            "MultiscaleCosSDRWavLoss_Direct":        self.sdr_loss(yd, td, out_dict=False),
        }
        # fmt: on
        return (
            losses if out_dict else torch.sum(torch.stack([v for v in losses.values()]))
        )
    
class LevelInvariantNormalizedLoss(nn.Module) : 
    """
    Braun, Sebastian, and Ivan Tashev. "Data augmentation and loss normalization for deep noise suppression." 
    Speech and Computer: 22nd International Conference, SPECOM 2020, 
    St. Petersburg, Russia, October 7–9, 2020, Proceedings 22. Springer International Publishing, 2020.
    """
    def __init__(self,alpha = 0.3, c = 0.3, n_fft = 512):
        super(LevelInvariantNormalizedLoss, self).__init__()
        self.n_fft = n_fft
        self.window = torch.hann_window(n_fft)
        self.alpha = alpha
        self.c = c
        self.MSE = torch.nn.MSELoss()
        
    def forward(self,output,target):
        # Normalize
        
        # STFT
        Y = torch.stft(output,self.n_fft,return_complex=True,center=False,window=self.window.to(output.device))
        S = torch.stft(target,self.n_fft,return_complex=True,center=False,window=self.window.to(output.device))
        
        # Compress
        mag_Y = torch.pow(torch.abs(Y),self.c)
        mag_S = torch.pow(torch.abs(S),self.c)
        
        # Complex
        phase_Y = torch.exp(1j*torch.angle(Y))
        phase_S = torch.exp(1j*torch.angle(S))
        
        # Complex MSE is not implemented
        cL = ((torch.abs(mag_Y*phase_Y - mag_S*phase_S))**2).mean()
        
        mL = self.MSE(mag_Y,mag_S)
        # Loss
        L = self.alpha*cL + (1-self.alpha)*mL


        
        return L