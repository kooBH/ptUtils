import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch import Tensor
 

EPS=1e-7

from torch.autograd import Function

def as_module(func):
    """
    Decorator to convert a function into an nn.Module factory.
    """
    class Wrapped(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
        def forward(self, *args, **kwargs):
            # Combine init-time kwargs and forward-time kwargs
            merged = {**self.kwargs, **kwargs}
            return func(*args, **merged)
    return Wrapped

def calulate_window(window_size, hop_length, norm=True):
    window = torch.zeros(window_size)
    denom = window_size // hop_length
    if norm: 
        if denom == 4 : 
            for i in range(window_size):
                window[i] = torch.sin(torch.tensor(3.14159265358979323846 * (i + 0.5) / window_size))
            tmp = 0
            for i in range(window_size) :
                tmp += window[i] * window[i]
            tmp /= window_size/4
            tmp = torch.sqrt(tmp)
            for i in range(window_size) :
                window[i] /= tmp
        elif denom == 2 :
        # sine window for 50% overlap
            for i in range(window_size) :
                window[i] = torch.sin(torch.pi * torch.tensor((i+0.5) / window_size))
    else :
        window = torch.hann_window(window_size)
    return window

class SpecLoss(nn.Module):
    def __init__(self, n_fft,n_hop):
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.window = calulate_window(n_fft,n_hop,True)

    def Analysis(self, x):
        return torch.stft(x, self.n_fft, hop_length=self.n_hop, window=self.window.to(x.device), return_complex=True, center=False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))

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

@as_module
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

class wMSELoss(nn.Module) : 
    def __init__(self, alpha=0.99):
        super(wMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        s_mag = torch.abs(target)
        s_hat_mag = torch.abs(output)

        # scale
        s_mag= torch.log10(1+s_mag)
        s_hat_mag= torch.log10(1+s_hat_mag)
    
        s_mag = s_mag/torch.max(s_mag)
        s_hat_mag = s_hat_mag/torch.max(s_hat_mag)

        d = s_mag - s_hat_mag

        return torch.mean(self.alpha*(d + d.abs())/2 + (1-self.alpha) * (d-d.abs()).abs()/2)


class mwMSELoss(nn.Module):
    def __init__(self, alpha=0.99, eps=1e-7, sr=16000, n_fft=512, n_mels=40, device="cuda:0"):
        super(mwMSELoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.sr = sr
        self.n_fft = n_fft
        self.device = device

        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft,n_mels=n_mels)
        mel_basis = torch.from_numpy(mel_basis)
        mel_basis = mel_basis.to(device)
        self.register_buffer("mel_basis", mel_basis)

        hann = torch.hann_window(n_fft,device=device)
        self.register_buffer("hann", hann)

    def forward(self, output, target):
        """
            Mel-domain Weighted Error
            output : STFT [B,L]
            target : STFT [B,L]
        """
        output = torch.stft(output, self.n_fft, window=self.hann,return_complex=True)
        target = torch.stft(target, self.n_fft, window=self.hann,return_complex=True)

        # ERROR : weight becomes NAN
        # --> torch backpropagation weight clipping on 'sqrt'
        #    --> omitted 'sqrt'
        #s_mag = torch.sqrt(target[:,:,:,0]**2 + target[:,:,:,1]**2)
        #s_hat_mag = torch.sqrt(output[:,:,:,0]**2 + output[:,:,:,1]**2)

        # add eps due to 'MmBackward nan' error in gradient
        s_hat_mag = torch.abs(output) + self.eps
        s_mag = torch.abs(target) + self.eps

        # scale
        s_mag= torch.log10(1+s_mag)
        s_hat_mag= torch.log10(1+s_hat_mag)

        # mel
        s = torch.matmul(self.mel_basis,s_mag)
        s_hat = torch.matmul(self.mel_basis,s_hat_mag)

        # Batch Norm
        s_mag = s_mag/torch.max(s_mag)
        s_hat_mag = s_hat_mag/torch.max(s_hat_mag)

        d = s - s_hat

        # s >= s_Hat
        # -> mwMSE = torch.mean(alpha*d)
        # else 
        # -> mwMSE = torch.mean((1-alpha)*d)

        mwMSE = torch.mean(self.alpha *(d + d.abs())/2 + (1-self.alpha) * (d-d.abs()).abs()/2)

        return  mwMSE

"""
weighted-SDR loss
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

    def forward(self, output, target):
        num = torch.sum(target * output, dim=-1)
        den = torch.norm(target, dim=-1) * torch.norm(output, dim=-1)
        loss_per_element = -num / (den + EPS)
        loss = self.reduction(loss_per_element)
        return loss

class CosSDRLoss(nn.Module):
    def __init__(self, reduction=torch.mean):
        super(CosSDRLoss, self).__init__()
        self.segment_loss = CosSDRLossSegment(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target, chunk_size=1024):

        if output.shape[1]% chunk_size != 0:
            output = output[..., : -(output.shape[-1] % chunk_size)]
            target = target[..., : -(target.shape[-1] % chunk_size)]

        out_chunks = torch.reshape(output, [output.shape[0], -1, chunk_size])
        trg_chunks = torch.reshape(target, [target.shape[0], -1, chunk_size])
        loss_per_element = torch.mean(
            self.segment_loss(out_chunks, trg_chunks), dim=-1
        )
        loss = self.reduction(loss_per_element)
        return loss

class MultiscaleCosSDRLoss(nn.Module):
    def __init__(self, chunk_sizes, reduction=torch.mean):
        super(MultiscaleCosSDRLoss, self).__init__()
        self.chunk_sizes = chunk_sizes
        self.loss = CosSDRLoss(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target):
        loss_per_scale = [
            self.loss(output, target, cs) for cs in self.chunk_sizes
        ]
        loss_per_element = torch.mean(torch.stack(loss_per_scale), dim=0)
        loss = self.reduction(loss_per_element)
        return loss

class SpectrogramLoss(nn.Module):
    def __init__(self, reduction=torch.mean,overlap=0.25,type_weight = 0, norm_window=False, weight_mag = 1.0, weight_cplx = 1.0, alpha_mag = None):
        super(SpectrogramLoss, self).__init__()
        self.gamma = 0.3
        self.reduction = reduction
        self.type_weight = type_weight
        self.norm_window = norm_window

        self.weight_mag = weight_mag
        self.weight_cplx = weight_cplx

        if overlap == 0.5 : 
            self.denom = 2
        else :
            self.denom = 4

        self.alpha_mag = alpha_mag
        self.windows = {}

    def apply_weight(self, value, nfft):
        # value : [B,F,T]
        if self.type_weight == 0 :
            return value

        nhfft = nfft//2 + 1
        # double weight on high freq
        if self.type_weight == 1 : 
            mid = int(nhfft/2)
            value[:,mid:,:] *= 2
            return value
        
    def forward(self, output, target, chunk_size=1024, out_dict=True):
        if "{}".format(chunk_size) in self.windows.keys() : 
            window = self.windows["{}".format(chunk_size)]
        # Create Window
        else :
            if self.norm_window : 
                window = torch.zeros(chunk_size)
                if self.denom == 4 : 
                    for i in range(chunk_size):
                        window[i] = torch.sin(torch.tensor(3.14159265358979323846 * (i + 0.5) / chunk_size))
                    tmp = 0
                    for i in range(chunk_size) :
                        tmp += window[i] * window[i];
                    tmp /= chunk_size/4;
                    tmp = torch.sqrt(tmp);
                    for i in range(chunk_size) :
                        window[i] /= tmp;
                elif self.denom == 2 :
                    # sine window for 50%overlap
                    for i in range(chunk_size) :
                        window[i] = torch.sin(torch.pi * torch.tensor((i+0.5) / chunk_size));
            else :
                window = torch.hann_window(chunk_size)
                    
            window = window.to(output.device)
            self.windows["{}".format(chunk_size)] = window

        # stft.shape == (batch_size, fft_size, num_chunks)
        stft_output = torch.stft(output, chunk_size, hop_length=chunk_size//self.denom, return_complex=True, center=False,window = window)
        stft_target = torch.stft(target, chunk_size, hop_length=chunk_size//self.denom, return_complex=True, center=False, window = window)

        # clip is needed to avoid nan gradients in the backprop
        mag_output = torch.clip(torch.abs(stft_output), min=EPS)**self.gamma
        mag_target = torch.clip(torch.abs(stft_target), min=EPS)**self.gamma
        if self.alpha_mag is not None :
            d = mag_target - mag_output
            # if mag_target >= mag_output
            #   dist_mag = alpha_mag * d
            # else 
            #   dist_mag = (1-alpha_mag) * d
            dist_mag = self.alpha_mag * (d + d.abs())/2 + (1-self.alpha_mag) * (d-d.abs()).abs()/2
        else : 
            dist_mag = (mag_target - mag_output).abs()

        cplx_output = torch.view_as_real(mag_output*torch.exp(1j*angle.apply(stft_output)))
        cplx_target = torch.view_as_real(mag_target*torch.exp(1j*angle.apply(stft_target)))

        dist_cplx = F.mse_loss(cplx_output, cplx_target)

        distance = self.weight_mag * dist_mag + self.weight_cplx * dist_cplx

        self.apply_weight(distance,chunk_size)

        # average out
        loss_per_chunk = torch.mean(distance**2, dim=1)
        loss_per_element = torch.mean(loss_per_chunk, dim=-1)
        loss = self.reduction(loss_per_element)
        return {"SpectrogramLoss": loss} if out_dict else loss

class MultiscaleSpectrogramLoss(nn.Module):
    def __init__(self, chunk_sizes, reduction=torch.mean,overlap=0.25,type_weight=None,norm_window=False,weight_mag = 1.0, weight_cplx = 1.0, alpha_mag = None):
        super(MultiscaleSpectrogramLoss, self).__init__()
        self.chunk_sizes = chunk_sizes
        self.loss = SpectrogramLoss(nn.Identity(),overlap=overlap,type_weight=type_weight,norm_window=norm_window,weight_mag=weight_mag, weight_cplx=weight_cplx, alpha_mag=alpha_mag)
        self.reduction = reduction
    def forward(self, output, target):
        loss_per_scale = [
            self.loss(output, target, cs, False) for cs in self.chunk_sizes
        ]
        loss_per_element = torch.mean(torch.stack(loss_per_scale), dim=0)
        loss = self.reduction(loss_per_element)
        return  loss

"""
Y. Ai and Z. -H. Ling, "Neural Speech Phase Prediction Based on Parallel Estimation Architecture and Anti-Wrapping Losses," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10096553.

{f_{AW}}(x) = \left| {x - 2\pi \cdot round\left( {\frac{x}{{2\pi }}} \right)} \right|,x \in \mathbb{R}.

{\mathcal{L}_{IP}} = {\mathbb{E}_{({\mathbf{\hat P}},{\mathbf{P}})}}\overline {{f_{AW}}({\mathbf{\hat P}} - {\mathbf{P}})} 

ERROR::This makes weird behavior for data longer than training data
"""
class AntiWrappingLoss(nn.Module):
    def __init__(self):
        super(AntiWrappingLoss, self).__init__()
    
    def forward(self, output, target, chunk_size=1024, hop_length=None):
        if hop_length is None : 
            hop_length = chunk_size//4

        # stft.shape == (batch_size, fft_size, num_chunks)
        stft_output = torch.stft(output, chunk_size, hop_length=hop_length, return_complex=True, center=False)
        stft_target = torch.stft(target, chunk_size, hop_length=hop_length, return_complex=True, center=False)

        p_hat = torch.angle(stft_output)
        p = torch.angle(stft_target)
    
        # (7)
        error = torch.abs(p_hat - p - 2*torch.pi*torch.round((p_hat-p)/(2*torch.pi)))

        # (8)
        return torch.mean(error)        

class MultiscaleAntiWrappingLoss(nn.Module):
    def __init__(self, chunk_sizes) : 
        super(MultiscaleAntiWrappingLoss, self).__init__()
        self.chunk_sizes = chunk_sizes
        self.loss = AntiWrappingLoss()

    def forward(self, output, target, out_dict=True):
        loss_per_scale = [
            self.loss(output, target, cs) for cs in self.chunk_sizes
        ]
        loss = torch.mean(torch.stack(loss_per_scale), dim=0)
        return loss

class MultiMagnitudeLoss(nn.Module):
    def __init__(self,
                 n_fft=512,
                 weight = [1.0,1.0,1.0],
                 PC_factor=0.3,
                 reduction = torch.mean,
                 **kwargs) :
        super(MultiMagnitudeLoss,self).__init__()
        self.gamma = PC_factor
        self.reduction = reduction
        self.weight = weight
        self.n_fft = n_fft

    def stft_to_db(self, stft_data):
        # STFT 도메인 데이터를 dB로 변환
        magnitude = torch.abs(stft_data)
        db_data = 20 * torch.log10(magnitude + 1e-10)  # 안정성을 위해 작은 값 추가
        return db_data

    # estim : (B,L) time domain audio
    # target :(B,L) time domain audio
    def forward(self, estim, target):
        spec_estim = torch.stft(estim, n_fft = self.n_fft, return_complex=True)
        spec_target = torch.stft(target, n_fft = self.n_fft, return_complex=True)

        mag_estim = torch.abs(spec_estim)
        mag_target = torch.abs(spec_target)

        
        db_target = self.stft_to_db(spec_target)
        max_target = self.stft_to_db(torch.max(mag_target))
        
        m10_mask = db_target >= (max_target - 10)
        m20_mask = (db_target >= (max_target - 20)) & (db_target < (max_target - 10))
        m30_mask = (db_target < (max_target - 20))

        zero_tensor = torch.tensor(1e-13, device=mag_estim.device, dtype=mag_estim.dtype, requires_grad=True)

        m10_estim = torch.where(m10_mask, mag_estim, zero_tensor)
        m20_estim = torch.where(m20_mask, mag_estim, zero_tensor)
        m30_estim = torch.where(m30_mask, mag_estim, zero_tensor)

        m10_target = torch.where(m10_mask, mag_target, zero_tensor)
        m20_target = torch.where(m20_mask, mag_target, zero_tensor)
        m30_target = torch.where(m30_mask, mag_target, zero_tensor)

        m10_dist =  m10_target**self.gamma - m10_estim**self.gamma
        m20_dist =  m20_target**self.gamma - m20_estim**self.gamma
        m30_dist =  m30_target**self.gamma - m30_estim**self.gamma

        m10_loss = self.reduction(torch.mean(m10_dist**2, dim=1)) + 1e-13
        m20_loss = self.reduction(torch.mean(m20_dist**2, dim=1)) + 1e-13
        m30_loss = self.reduction(torch.mean(m30_dist**2, dim=1)) + 1e-13
                             
        #print(f"{m10_loss:.8f} {m20_loss:.8f} {m30_loss:.8f}")
        
        loss = self.weight[0]*m10_loss + self.weight[1]*m20_loss + self.weight[2]*m30_loss
        return loss

"""
SA-SNR
Li, Yihao, Meng Sun, and Xiongwei Zhang. "Scale-aware dual-branch complex convolutional recurrent network for monaural speech enhancement." Computer Speech & Language 86 (2024): 101618.
"""
class SASNRLossSegment(nn.Module):
    def __init__(self, reduction=torch.mean):
        super(SASNRLossSegment, self).__init__()
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):

        # (9)
        denom_s = torch.sum(output*target,dim=-1)
        numer_s = torch.norm(target, dim=-1) * torch.norm(target, dim=-1)
        #scale = (numer_s/denom_s)
        scale = denom_s/(numer_s+EPS)
        s_target = torch.unsqueeze(scale,-1) * target

        # (10)
        e_noise = output - s_target
 
        # (14)
        # Compute norms
        norm_s_target_squared = torch.sum(s_target ** 2, dim=-1)  
        norm_e_noise_squared = torch.sum(e_noise ** 2, dim=-1)  
        norm_predicted = torch.norm(output, dim=-1)  
        norm_target = torch.norm(target, dim=-1) 

        # Scaling factors
        scaling_factor = norm_predicted / (norm_target + EPS)
        min_factor = torch.minimum(scaling_factor, torch.ones_like(scaling_factor))
        max_factor = torch.maximum(scaling_factor, torch.ones_like(scaling_factor))
        scale_term = min_factor / max_factor

        # Compute SA-SNR loss
        ratio = norm_s_target_squared / (norm_e_noise_squared + EPS)  
        loss_per_element = -10 * torch.log10(ratio * scale_term + EPS)  

        loss = self.reduction(loss_per_element)
        return {"SASNRLossSegment": loss} if out_dict else loss

class SASNRLoss(nn.Module):
    def __init__(self, reduction=torch.mean):
        super(SASNRLoss, self).__init__()
        self.segment_loss = SASNRLossSegment(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target, chunk_size=1024, out_dict=True):
        out_chunks = torch.reshape(output, [output.shape[0], -1, chunk_size])
        trg_chunks = torch.reshape(target, [target.shape[0], -1, chunk_size])
        loss_per_element = torch.mean(
            self.segment_loss(out_chunks, trg_chunks, False), dim=-1
        )
        loss = self.reduction(loss_per_element)
        return {"SASNRLoss": loss} if out_dict else loss

class MultiscaleSASNRLoss(nn.Module):
    def __init__(self, chunk_sizes, reduction=torch.mean):
        super(MultiscaleSASNRLoss, self).__init__()
        self.chunk_sizes = chunk_sizes
        self.loss = SASNRLoss(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):
        loss_per_scale = [
            self.loss(output, target, cs, False) for cs in self.chunk_sizes
        ]
        loss_per_element = torch.mean(torch.stack(loss_per_scale), dim=0)
        loss = self.reduction(loss_per_element)
        return {"MultiscaleSASNRLoss": loss} if out_dict else loss

################# Multi-Losses

# Mutliscale Loss
class TrunetLoss(nn.Module):
    def __init__(self, frame_size_sdr=[4096, 2048, 1024, 512], frame_size_spec=[1024, 512, 256], overlap=0.25):
        super(TrunetLoss, self).__init__()
        
        self.max_size = max(max(frame_size_sdr), max(frame_size_spec))

        self.sdr_loss = MultiscaleCosSDRLoss(frame_size_sdr)
        self.spc_loss = MultiscaleSpectrogramLoss(frame_size_spec,overlap=overlap)

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
            "MultiscaleSpectrogramLoss_Direct":      self.spc_loss(yd, td),
            "MultiscaleCosSDRWavLoss_Direct":        self.sdr_loss(yd, td),
        }
        # fmt: on
        return (
            losses if out_dict else torch.sum(torch.stack([v for v in losses.values()]))
        )

## MutiLoss
class MultiLoss1(nn.Module):
    def __init__(self, 
    frame_size_sdr=[4096, 2048, 1024, 512], 
    frame_size_spec=[1024, 512, 256],
    frame_size_aw=[1024, 512, 256],
    weight_spec = 15,
    weight_sdr = 1,
    weight_aw = 1):
        super(MultiLoss1, self).__init__()
        
        self.max_size = max(max(frame_size_sdr), max(frame_size_spec),max(frame_size_aw))

        self.sdr_loss = MultiscaleCosSDRLoss(frame_size_sdr)
        self.spc_loss = MultiscaleSpectrogramLoss(frame_size_spec)
        self.aw_loss = MultiscaleAntiWrappingLoss(frame_size_aw)

        self.weight_spec = weight_spec
        self.weight_sdr = weight_sdr
        self.weight_aw = weight_aw

    def forward(self, outputs, targets, out_dict=False):
        yd = outputs
        td = targets

        if yd.shape[1]% self.max_size != 0:
            yd = yd[..., : -(yd.shape[-1] % self.max_size)]
            td = td[..., : -(td.shape[-1] % self.max_size)]

        self.losses = {
            "MultiscaleSpectrogramLoss_Direct":      self.weight_spec*self.spc_loss(yd, td),
            "MultiscaleCosSDRWavLoss_Direct":        self.weight_sdr*self.sdr_loss(yd, td, out_dict=False),
            "MultiscaleAntiWrappingLoss_Direct":     self.weight_aw*self.aw_loss(yd, td),
        }

        # fmt: on
        return (
            self.losses if out_dict else torch.sum(torch.stack([v for v in self.losses.values()]))
        )

# Multi-loss 2
class MultiLoss2(nn.Module):
    def __init__(self, frame_size_sdr=[4096, 2048, 1024, 512], frame_size_spec=[1024, 512, 256], overlap=0.25,weight_spec=15,weight_sdr = 1):
        super(MultiLoss2, self).__init__()

        self.weight_spec = weight_spec
        self.weight_sdr = weight_sdr
        
        self.max_size = max(max(frame_size_sdr), max(frame_size_spec))

        self.sdr_loss = MultiscaleCosSDRLoss(frame_size_sdr)
        self.spc_loss = MultiscaleSpectrogramLoss(frame_size_spec,overlap=overlap)

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
            "MultiscaleSpectrogramLoss_Direct":      self.weight_spec * self.spc_loss(yd, td),
            "MultiscaleCosSDRWavLoss_Direct":        self.weight_sdr * self.sdr_loss(yd, td, out_dict=False),
        }
        # fmt: on
        return (
            losses if out_dict else torch.sum(torch.stack([v for v in losses.values()]))
        )

# Multi-loss3
class MultiDecibelLoss(nn.Module):
    def __init__(self, 
    frame_size_sdr=[4096, 2048, 1024, 512],
    weight=[1,1,1]
    ):
        super(MultiDecibelLoss, self).__init__()
        
        self.max_size = max(frame_size_sdr)

        self.sdr_loss = MultiscaleCosSDRLoss(frame_size_sdr)
        self.mdb_loss = MultiMagnitudeLoss(weight=weight)

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
            "MultiDecibelLoss":      self.mdb_loss(yd, td),
            "MultiscaleCosSDRWavLoss_Direct":        self.sdr_loss(yd, td, out_dict=False),
        }
        # fmt: on
        return (
            losses if out_dict else torch.sum(torch.stack([v for v in losses.values()]))
        )


# Multi-loss 4
class MultiLoss4(nn.Module):
    def __init__(self, frame_size_sdr=[4096, 2048, 1024, 512], frame_size_spec=[1024, 512, 256], overlap=0.25,weight_spec=1, weight_sdr = 1):
        super(MultiLoss4, self).__init__()

        self.weight_spec = weight_spec
        self.weight_sdr = weight_sdr
        
        self.max_size = max(max(frame_size_sdr), max(frame_size_spec))

        self.sdr_loss = MultiscaleSASNRLoss(frame_size_sdr)
        self.spc_loss = MultiscaleSpectrogramLoss(frame_size_spec,overlap=overlap)

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
            "MultiscaleSpectrogramLoss_Direct":      self.weight_spec * self.spc_loss(yd, td),
            "MultiscaleSASNRLoss_Direct":        self.weight_sdr * self.sdr_loss(yd, td, out_dict=False),
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


    
### None Speech Loss
class LSNRLoss(nn.Module):
    def __init__(self,n_fft=512,hop_length=256):
        super(LSNRLoss, self).__init__()
        self. n_fft = n_fft
        self.hop_length = hop_length
        self.criterion = nn.MSELoss()

    def forward(self, estim, clean, noisy):
        # estim : [B,T,1]
        # clean : [B,L]
        # noisy : [B,L]
        
        # Cunfold
        clean_seg = clean.unfold(1, self.n_fft, self.hop_length)
        noise = noisy - clean
        noise_seg = noise.unfold(1, self.n_fft, self.hop_length)

        clean_mag = torch.sum(clean_seg**2, dim=2, keepdim=True)
        noise_mag = torch.sum(noise_seg**2,dim=2,keepdim=True)

        clean_dB = 10 * torch.log10(clean_mag + 1e-10)
        noise_dB = 10 * torch.log10(noise_mag + 1e-10)

        lsnr = clean_dB - noise_dB
        lsnr = nn.functional.pad(lsnr, (0,0,1,1), mode='constant', value=0.0)
        lsnr = torch.clamp(lsnr, min=-20.0, max=20.0)
        lsnr = lsnr/20.0

        loss = self.criterion(estim, lsnr)
        return loss



# From https://github.com/dmlguq456/TF_Restormer
class UTMOSLoss(torch.nn.Module):
	def __init__(self, fs, device='cuda'):
		super().__init__()
		# Load UTMOS22 strong model via torch.hub
		self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
		self.model.to(device)
		self.model.train()
		self.fs = fs
	
    # target is not used in this loss
	def forward(self, estim, target):
		# Ensure inputs are in the correct shape (batch_size, samples)
		if estim.dim() == 1:
			estim = estim.unsqueeze(0)

		scores = self.model(estim, sr=self.fs)
		
		return -torch.mean(scores)
	
	def mos(self, x):
		# Ensure inputs are in the correct shape (batch_size, samples)
		if x.dim() == 1:
			x = x.unsqueeze(0)
		
		with torch.no_grad():
			scores = self.model(x, sr=self.fs)
		
		return scores
    

"""
Based on https://github.com/aask1357/fastenhancer/

MIT License

Copyright (c) 2025 AHN Sung Hwan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class MagMSELoss(SpecLoss):
    def __init__(self, n_fft,n_hop,**kwargs): 
        super().__init__(n_fft,n_hop)
    def forward(self, s1: Tensor, s2: Tensor) -> Tensor:
        s1 = self.Analysis(s1)
        s2 = self.Analysis(s2)
        s1 = torch.abs(s1)
        s2 = torch.abs(s2)
        return F.mse_loss(s1, s2)

class ComplexMSELoss(SpecLoss):
    def __init__(self, n_fft, n_hop, **kwargs):
        super().__init__(n_fft,n_hop)
    
    def forward(self, s1: Tensor, s2: Tensor) -> Tensor:
        s1 = self.Analysis(s1)
        s2 = self.Analysis(s2)
        return F.mse_loss(torch.view_as_real(s1), torch.view_as_real(s2))

class PESQLoss(nn.Module):
    def __init__(self,**kwargs):
        from torch_pesq import PesqLoss
        super().__init__()
        self.initialized = False
        self.pesq_loss = PesqLoss(1.0, sample_rate=16_000)
    
    def forward(self, estim, target) -> Tensor:
        if not self.initialized:
            self.pesq_loss.to(estim.device)
            self.initialized = True
        with torch.amp.autocast("cuda", enabled=False):
            loss = self.pesq_loss(target.float(), estim.float())
        return loss.mean()
    
class WavL1Loss(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
    
    def forward(self, wav_hat: Tensor, wav: Tensor) -> Tensor:
        return F.l1_loss(wav_hat, wav)

class ConsistencyLoss(SpecLoss):
    def __init__(
        self, n_fft,n_hop, compression: float = 1.0, eps=1e-7,**kwargs
    ):
        super().__init__(n_fft,n_hop)
        self.compression = compression
        self.eps = eps
    
    def forward(self, estim: Tensor, target: Tensor) -> Tensor:
        estim = self.Analysis(estim)
        target = self.Analysis(target)

        mag_hat = torch.abs(estim).clamp(min=self.eps)
        mag = torch.abs(target).clamp(min=self.eps)

        estim = estim * mag_hat.pow(self.compression - 1.0)
        target = target * mag.pow(self.compression - 1.0)

        loss = F.mse_loss(torch.view_as_real(estim), torch.view_as_real(target))

        return loss
    
import itertools

# Loss wrapper for Permutation Invariant Training (PIT)
class PITLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, max_src: int):
        super().__init__()
        self.base_loss = base_loss
        self.max_src = max_src

        self.perms = list(itertools.permutations(range(self.max_src)))

    def forward(self, estim: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            estim: (Batch, Num_Sources, Features)
            target: (Batch, Num_Sources, Features)
        Returns:
            min_loss: Scalar tensor represented the average PIT loss over the batch.
        """
        batch_size = estim.size(0)
        
        # Holder for losses of all permutations per sample
        # Expected final shape after loop: (Batch, Num_Permutations)
        batch_perm_losses = []
        
        for perm in self.perms:
            permuted_estim = estim[:, perm, :]
            
            source_losses = []
            for i in range(self.max_src):
                # Compute loss for the current source pair
                loss_per_sample = self.base_loss(permuted_estim[:, i], target[:, i])
                
                # Dynamic shape handling for safety (especially when batch_size = 1 or reduction='mean')
                # Force the loss to have at least 1 dimension representing the batch
                if loss_per_sample.dim() == 0:
                    # If base_loss returned a scalar (due to reduction='mean' and batch_size=1),
                    # expand it back to shape (1,) to represent the batch dimension.
                    loss_per_sample = loss_per_sample.unsqueeze(0)
                elif loss_per_sample.dim() > 1:
                    # If features remain, reduce them to sound-level loss per sample
                    loss_per_sample = torch.mean(loss_per_sample, dim=list(range(1, loss_per_sample.dim())))
                
                source_losses.append(loss_per_sample)  # List of (Batch,)
            
            # Average over sources for the current permutation
            # Shape: (Batch,)
            perm_loss = torch.mean(torch.stack(source_losses, dim=0), dim=0)
            batch_perm_losses.append(perm_loss)
            
        # Shape: (Batch, Num_Permutations)
        batch_perm_losses = torch.stack(batch_perm_losses, dim=1)
        
        # Find the minimum loss for EACH sample independently
        # min_losses_per_sample shape: (Batch,)
        min_losses_per_sample, _ = torch.min(batch_perm_losses, dim=1)
        
        # Final objective: average over the batch (Works perfectly for batch_size=1 as well)
        return torch.mean(min_losses_per_sample)


####### ACCDOA #####
def ACCDOA_label(vad,azimuth,elevation):
    """Generates ACCDOA (Activity-coupled Cartesian Direction of Arrival) labels.
    Args:
        vad (torch.Tensor): Voice Activity Detection format with shape [B, N, T]
        azimuth (torch.Tensor): Azimuth angle in radians with shape [B, N]
        elevation (torch.Tensor): Elevation angle in radians with shape [B, N]
    Returns:
        torch.Tensor: ACCDOA labels with shape [B, N, T, 3]
                      The last dimension contains [x, y, z] coordinates.
    """
    # 1. Unsqueeze to align with the time dimension (T) of VAD [B, N] -> [B, N, 1]
    azimuth_expanded = azimuth.unsqueeze(-1)
    elevation_expanded = elevation.unsqueeze(-1)

    # 2. Convert Spherical Coordinates to Cartesian Coordinates (Unit Vector)
    # Target shape for each component: [B, N, 1] (will broadcast to [B, N, T] during multiplication)
    x = torch.cos(elevation_expanded) * torch.cos(azimuth_expanded)
    y = torch.cos(elevation_expanded) * torch.sin(azimuth_expanded)
    z = torch.sin(elevation_expanded)

    # 3. Couple Cartesian coordinates with voice activity (VAD)
    # Shape of each component after broadcasting: [B, N, T]
    acc_x = vad * x
    acc_y = vad * y
    acc_z = vad * z

    # 4. Stack along the last dimension to get [B, N, T, 3]
    # This aligns the [x, y, z] vectors at the final axis for each time frame
    acc_doa = torch.stack([acc_x, acc_y, acc_z], dim=-1)
    acc_doa = acc_doa.float() 
    acc_doa.to(vad.device)
    return acc_doa

class PIT_ACCDOALoss(nn.Module):
    def __init__(self, max_src):
        super(PIT_ACCDOALoss, self).__init__()
        self.max_src = max_src
        self.criterion = nn.MSELoss(reduction='none')  # Compute loss per element for PIT
        self.perms = list(itertools.permutations(range(self.max_src)))

    def forward(self, estim, target_vad, target_azimuth, target_elevation):
        """
        Args:
            estim: (Batch, Num_Sources, Time, 3) - Predicted ACCDOA vectors
            target_vad: (Batch, Num_Sources, Time`) - Ground truth VAD. Time dimension may differ from estim, but should be interpolated
            target_azimuth: (Batch, Num_Sources) - Ground truth azimuth angles in radians
            target_elevation: (Batch, Num_Sources) - Ground truth elevation angles in radians
        Returns:
            loss: Scalar tensor representing the PIT loss over ACCDOA predictions.
        """
        # Generate target ACCDOA labels from VAD and DOA angles
        target_accdoa = ACCDOA_label(target_vad, target_azimuth, target_elevation) # [B,N,T,3]

        # Interpolate estimated ACCDOA to match the time dimension of target_vad if necessary
        if estim.shape[2] != target_accdoa.shape[2]:
            target_time = target_accdoa.shape[2]
            target_3 = target_accdoa.shape[3]
            estim = F.interpolate(estim, size=(target_time, target_3), mode='bilinear', align_corners=False)

        b_losses = []
        for perm in self.perms:
            permuted_estim = estim[:, perm, :, :]  # Permute the source dimension
            
            loss_per_source = self.criterion(permuted_estim, target_accdoa)  # Element-wise MSE
            
            loss_per_source = loss_per_source.mean(dim=[1, 2])  # Average over Time and Coordinate
            
            b_losses.append(loss_per_source)  # List of (Batch,) for each permutation
        # Stack losses for all permutations: Shape (Batch, Num_Permutations)
        b_losses = torch.stack(b_losses, dim=1)
        # Find the minimum loss for each sample across permutations: Shape (Batch,)
        min_loss_per_sample, _ = torch.min(b_losses, dim=1)
        # Average over the batch to get the final scalar loss
        return torch.mean(min_loss_per_sample)

from itertools import combinations
class PairwisePearsonCorrelationLoss(nn.Module):
    def __init__(self, N: int, eps: float = 1e-8):
        """
        Loss function to penalize pairwise Pearson correlation between N sources.
        """
        super(PairwisePearsonCorrelationLoss, self).__init__()
        self.comb = list(combinations(range(N), 2))  # All unique pairs of sources
        self.eps = eps

    def forward(self, x: torch.Tensor, dummy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Estimated sources tensor of shape [B, N, T]
            dummy (torch.Tensor): Dummy argument (not used)
        Returns:
            torch.Tensor: Scalar loss value
        """
        B, N, T = x.shape
        if N <= 1:
            # If there is only 1 source, correlation between different sources cannot be computed.
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        def Pearson(estim,target):
            # Compute Pearson Correlation Coefficient
            numerator = torch.sum(estim * target, dim=-1)
            denominator = torch.sqrt(torch.sum(estim ** 2, dim=-1) * torch.sum(target ** 2, dim=-1) + 1e-8)
            correlation = numerator / denominator
            return correlation
        
        loss = 0.0
        for i, j in self.comb:
            corr = Pearson(x[:, i, :], x[:, j, :])  # Shape: [B]
            loss += torch.mean(corr ** 2)  # Penalize squared correlation
      
        return loss
    


### Integrated Loss Module
class ListLoss(nn.Module):
    def __init__(self,hp, loss_list : list, weight_list = None):
        super(ListLoss, self).__init__()
        self.loss_list = loss_list
        self.hp = hp

        losses = []
        
        for loss in loss_list:
            cls = globals()[loss]
            m = cls(**hp[loss])
            self.add_module(loss,m)
            losses.append(m)

        self.weight = torch.ones(len(losses))

        if weight_list is not None:
            if len(weight_list) != len(losses):
                raise ValueError("weight_list must be the same length as loss_list")

            for i, weight in enumerate(weight_list):
                if weight is not None:
                    self.weight[i] = weight 

        self.n_modules = len(losses)
        self.losses = nn.ModuleList(losses)
        
    def forward(self, output, target,return_dict=False):
        self.weight = self.weight.to(output.device)
        loss = 0
        lost_dict = {}
        for i in range(self.n_modules):
            l = self.losses[i](output, target) * self.weight[i]
            loss += l
            if torch.isnan(l):
                print(f"Loss {self.loss_list[i]} is NaN")
            lost_dict[self.loss_list[i]] = l

        if return_dict:
            return loss, lost_dict
        return loss