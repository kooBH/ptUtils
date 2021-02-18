import pytorch

# post filter from 
#(2017,ICASSP)A SPEECH ENHANCEMENT ALGORITHM BY ITERATING SINGLE- AND MULTI-MICROPHONE PROCESSING AND ITS APPLICATION TO ROBUST ASR

# default parameters as the paper
def gain_base(s,m,alpha=-5,beta=2):
    # [F,T] ? [B,F,T] ? 
""" formula(9)
    cSNR(f) = 10\log_{10}\frac{\sum^T_{t=1}m(t,f)\tilde{s}(t,f)^2}{\sum^T_{t=1}(1-m(t,f))\tilde{s}(t,f)^2}   \tag{9}   
"""

    numer = m*torch.pow(s,2)
    denom = (1-m)*torch.pow(s,2)
    numer = torch.sum(numer,dim=1)
    denom = torch.sum(denom,dim=1)
    cSNR = 10*torch.log(numer/denom)

""" formula(10)
    cSNR(f) = 10\log_{10}\frac{\sum^T_{t=1}m(t,f)\tilde{s}(t,f)^2}{\sum^T_{t=1}(1-m(t,f))\tilde{s}(t,f)^2}   \tag{9}   
     \lambda(f) = \frac{1}{1+exp((cSNR(f)-\alpha)/\beta)}   \tag{10}
"""
    lambda_val = 1/(1+torch.exp((cSNR-alpha)/beta))

""" formula(11)
g(t,f) = m(t,f)^{\lambda(f)} \tag{11}
"""
    return torch.pow(m,lambda_val)


