import numpy as np
import torch
from tensorboardX import SummaryWriter

try : 
    from .plotting import *
except ImportError:
    from plotting import *

# https://pytorch.org/docs/stable/tensorboard.html

class MyWriter(SummaryWriter):
    def __init__(self, logdir, n_fft=512,n_hop=128):
        super(MyWriter, self).__init__(logdir)

        self.n_fft = n_fft
        self.n_hop = n_hop

        self.window = torch.hann_window(window_length=n_fft, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False)
    def log_value(self, train_loss, step,tag):
        self.add_scalar(tag, train_loss, step)

    def log_train(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_test(self,test_loss,step) : 
        self.add_scalar('test_loss', test_loss, step)

    def log_audio(self,wav,label='label',step=0,sr=16000) : 
        wav = wav.detach().cpu().numpy()
        wav = wav/np.max(np.abs(wav))
        self.add_audio(label, wav, step, sr)

    def log_MFCC(self,input,output,clean,step):
        input = input.to('cpu')
        output = output.to('cpu')
        clean= clean.to('cpu')

        noisy = input[0]
        estim = input[1]

        noisy = noisy.detach().numpy()
        estim = estim.detach().numpy()
        output = output.detach().numpy()
        clean= clean.detach().numpy()

        output = np.expand_dims(output,0)
        clean = np.expand_dims(clean,0)

        noisy = MFCC2plot(noisy)
        estim = MFCC2plot(estim)
        output = MFCC2plot(output)
        clean = MFCC2plot(clean)

        self.add_image('noisy',noisy,step,dataformats='HWC')
        self.add_image('estim',estim,step,dataformats='HWC')
        self.add_image('clean',clean,step,dataformats='HWC')
        self.add_image('output',output,step,dataformats='HWC')

        #self.add_image('noisy',noisy,step)
        #self.add_image('estim',estim,step)
        #self.add_image('output',output,step)

    # add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
    def log_spec(self,data,label,step) :
        self.add_image(label,
            spec2plot(data), step, dataformats='HWC')

    def log_mag(self,data,label,step):
        self.add_image(label,
            mag2plot(data), step, dataformats='HWC')
 
    def log_wav2spec(self,noisy,estim,clean,step) :
        noisy = torch.from_numpy(noisy)
        estim = torch.from_numpy(estim)
        clean = torch.from_numpy(clean)

        noisy = torch.stft(noisy,n_fft=self.n_fft, hop_length = self.n_hop, window = self.window, center = True, normalized=False, onesided=True)
        estim = torch.stft(estim,n_fft=self.n_fft, hop_length = self.n_hop, window = self.window, center = True, normalized=False, onesided=True)
        clean = torch.stft(clean,n_fft=self.n_fft, hop_length = self.n_hop, window = self.window, center = True, normalized=False, onesided=True)

        self.log_spec(noisy,estim,clean,step)
    
    """
    data : 
        (9, n_sample) == [noisy, target 0 ~ target 3, output 0 ~ output 3]
    """
    def log_DOA_wav(self,data,step,label="Output"):
        image = wav2plotDOA(data)
        self.add_image(label,image,step)

if __name__=='__main__':
    MyWriter()

