import pesq
import torch
import numpy as np

"""
    output : wav[n_target,n_sample]
    target : wav[n_target,n_sample]
"""
def SIR(estim,target, requires_grad=False,device="cuda:0") :
    if estim.shape != target.shape : 
        raise Exception("ERROR::metric.py::SIR:: output shape != target shape | {} != {}".format(output.shape,target.shape))

    if len(estim.shape) != 2 : 
        raise Exception("ERROR::metric.py::SIR:: output dim {} != 2".format(len(output.shape)))
    n_target  = estim.shape[0]
    
    s_target = []
    e_interf = []

    for i in range(n_target) : 
        s_target.append(torch.inner(estim[i],target[i])*target[i]/torch.inner(target[i],target[i]))

        tmp = None
        for j in range(n_target) : 
            if i == j :
                continue
            if tmp is None : 
                tmp = torch.inner(estim[i],target[j])*target[j]/torch.inner(target[j],target[j])
            else : 
                tmp += torch.inner(estim[i],target[j])*target[j]/torch.inner(target[j],target[j])
        e_interf.append(tmp)

    SIR =  torch.tensor(0.0, requires_grad=requires_grad).to(device)
    for i in range(n_target) : 
        SIR += (torch.inner(s_target[i],s_target[i]))/torch.inner(e_interf[i],e_interf[i])
    return 10*torch.log10(SIR)

"""

"""
def PESQ(estim,target,fs=16000,mode="both") :
    estim = estim.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    if mode =="wb" : 
        val_pesq = pesq.pesq(fs, target, estim, 'wb',on_error=pesq.PesqError.RETURN_VALUES)
    elif mode == "nb" :
        val_pesq = pesq.pesq(fs, target, estim, 'nb',on_error=pesq.PesqError.RETURN_VALUES)
    else :
        val_pesq = pesq.pesq(fs, target, estim, 'wb',on_error=pesq.PesqError.RETURN_VALUES)
        val_pesq += pesq.pesq(fs,target,estim,'nb',on_error=pesq.PesqError.RETURN_VALUES)
        val_pesq /= 2
    return val_pesq