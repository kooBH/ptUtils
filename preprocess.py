import torch
import torch.nn as nn

hann_window = torch.hann_window(n_fft)

"""
    x : cpu, [n_sample]
    return : cpu, [1 F T]
"""
def wav2spec(x,n_fft=512,n_hop=128):

    X = torch.stft(
        x, 
        n_fft=n_fft, 
        hop_length=n_hop,
        window = hann_window,
        return_complex = True
    )

    return torch.unsqueeze(X,0)


"""
    x : [1 F T]
    return : [3 F T]
        magnitude, real, imag
"""
def spec2MRI(x):

    X = torch.cat((torch.abs(x),x.real,x.imag),0)

    return X