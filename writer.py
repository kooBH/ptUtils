import numpy as np
import torch
from tensorboardX import SummaryWriter

from .plotting import spec2plot


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp
        self.window = torch.hann_window(window_length=hp.audio.frame, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False)

    def log_train(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_test(self,test_loss,step) : 
        self.add_scalar('test_loss', test_loss, step)

    def log_audio(self,noisy,estim,clean,step) : 
        self.add_audio('noisy', noisy, step, self.hp.audio.samplerate)
        self.add_audio('estim', estim, step, self.hp.audio.samplerate)
        self.add_audio('clean', clean, step, self.hp.audio.samplerate)

    def log_spec(self,noisy,estim,clean,step) :
        self.add_image('noisy',
            spec2plot(noisy), step, dataformats='HWC')
        self.add_image('estim',
            spec2plot(estim), step, dataformats='HWC')
        self.add_image('clean',
            spec2plot(clean), step, dataformats='HWC')
 
    def log_wav2spec(self,noisy,estim,clean,step,num_frame) :
        noisy = torch.stft(noisy,n_fft=hp.audio.frame, hop_length = hp.aduio.shift, window = window, center = True, normalized=False, onesided=True,length=num_frame*hp.audio.shift)
        estim = torch.stft(estim,n_fft=hp.audio.frame, hop_length = hp.aduio.shift, window = window, center = True, normalized=False, onesided=True,length=num_frame*hp.audio.shift)
        noise = torch.stft(noise,n_fft=hp.audio.frame, hop_length = hp.aduio.shift, window = window, center = True, normalized=False, onesided=True,length=num_frame*hp.audio.shift)

        log_spec(noisy,estim,clean,step)
