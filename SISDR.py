import torch

# SI-SDR(Scale Invariant Source-to-Distortion Ratio)

# (2019,ICASSP)SDR – Half-baked or Well Done?
# https://ieeexplore.ieee.org/abstract/document/8683855

# NOTE ::  SDR == SI-SDR, 

defalut_n_fft = 1024
window = torch.hann_window(window_length=defalut_n_fft,periodic=True, dtype=None, 
                           layout=torch.strided, device=None, requires_grad=False)

window = window.to('cuda:1')

# Based on https://github.com/sigsep/bsseval/issues/3
def SISDR(output, target, inSTFT=True, n_fft=1024):
    global window
    if inSTFT : 
        if n_fft != defalut_n_fft : 
            window = torch.hann_window(window_length=fft_size,periodic=True, dtype=None, 
                           layout=torch.strided, device=None, requires_grad=False)
            
        output = torch.istft(output, n_fft, hop_length=None, win_length=None, window=window, center=True, normalized=False, onesided=None, length=None, return_complex=False)
        target= torch.istft(target, n_fft, hop_length=None, win_length=None, window=window, center=True, normalized=False, onesided=None, length=None, return_complex=False)

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
def SDRLoss(output,target, inSTFT=True, n_fft=1024,eps=2e-7):
    global window

    if inSTFT : 
        if n_fft != defalut_n_fft : 
            window = torch.hann_window(window_length=fft_size,periodic=True, dtype=None, 
                           layout=torch.strided, device=None, requires_grad=False)
            
        output = torch.istft(output, n_fft, hop_length=None, win_length=None, window=window, center=True, normalized=False, onesided=None, length=None, return_complex=False)
        target= torch.istft(target, n_fft, hop_length=None, win_length=None, window=window, center=True, normalized=False, onesided=None, length=None, return_complex=False)

    xy = torch.diag(output @ target.t())
    yy = torch.diag(target @ target.t())
    xx = torch.diag(output @ output.t())

    SDR = xy**2/ (yy*xx - xy**2 )
    return torch.mean(1/SDR)

# mSDR is not Signal Invariant
def mSDRLoss(output,target, inSTFT=True, n_fft=1024, eps=2e-7):
    global window
    if inSTFT : 
        if n_fft != defalut_n_fft : 
            window = torch.hann_window(window_length=fft_size,periodic=True, dtype=None, 
                           layout=torch.strided, device=None, requires_grad=False)
            
        output = torch.istft(output, n_fft, hop_length=None, win_length=None, window=window, center=True, normalized=False, onesided=None, length=None, return_complex=False)
        target= torch.istft(target, n_fft, hop_length=None, win_length=None, window=window, center=True, normalized=False, onesided=None, length=None, return_complex=False)
    # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
    # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
    #  > Maximize Correlation while producing minimum energy output.
    #xx = torch.dot(output,output)
    #xy = torch.dot(output,target)

    #return xx/(xy**2)
    correlation = torch.sum(target * output, dim=1)
    energies = torch.norm(target, p=2, dim=1) * torch.norm(output, p=2, dim=1)

    return torch.mean(-(correlation / (energies + eps)))

# test
if __name__ == '__main__' :
    root = '/home/data/kbh/MCSE/CGMM_RLS_MPDR/'
    clean_path = root + 'clean/011_011C0201.pt'
    SNRm5 = root + 'SNR-5/estimated_speech/011_011C0201.pt'
    SNR0 = root + 'SNR0/estimated_speech/011_011C0201.pt'
    SNRp5 = root + 'SNR5/estimated_speech/011_011C0201.pt'

    spec_clean = torch.load(clean_path)
    spec_SNRm5 = torch.load(SNRm5)
    spec_SNR0  = torch.load(SNR0)
    spec_SNRp5 = torch.load(SNRp5)

    print(spec_clean.shape)

    SDRm5_1 = SDRLoss(spec_SNRm5,spec_clean)
    SDR0_1 = SDRLoss(spec_SNR0,spec_clean)
    SDRp5_1 = SDRLoss(spec_SNRp5,spec_clean)

    SDRm5_2 = mSDRLoss(spec_SNRm5,spec_clean)
    SDR0_2 = mSDRLoss(spec_SNR0,spec_clean)
    SDRp5_2 = mSDRLoss(spec_SNRp5,spec_clean)

    spec_clean = spec_clean * 30

    SDRm5_3 = SDRLoss(spec_SNRm5,spec_clean)
    SDR0_3 = SDRLoss(spec_SNR0,spec_clean)
    SDRp5_3 = SDRLoss(spec_SNRp5,spec_clean)

    SDRm5_4 = mSDRLoss(spec_SNRm5,spec_clean)
    SDR0_4 = mSDRLoss(spec_SNR0,spec_clean)
    SDRp5_4 = mSDRLoss(spec_SNRp5,spec_clean)

    print('--- SDR ---')
    print(SDRm5_1)
    print(SDR0_1)
    print(SDRp5_1)
    print('--- mSDR ---')
    print(SDRm5_2)
    print(SDR0_2)
    print(SDRp5_2)
    print('--- SDR *30 ---')
    print(SDRm5_3)
    print(SDR0_3)
    print(SDRp5_3)
    print('--- mSDR *30 ---')
    print(SDRm5_4)
    print(SDR0_4)
    print(SDRp5_4)
    print('--- mSDR (x,x) ---')
    print(mSDRLoss(spec_clean,spec_clean))

    print('--------------')
    print(SDRLoss(spec_clean,spec_clean))

    # Same Output
