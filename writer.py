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
 
    def log_wav2spec(self,noisy,estim,clean,step) :
        noisy = torch.from_numpy(noisy)
        estim = torch.from_numpy(estim)
        clean = torch.from_numpy(clean)

        noisy = torch.stft(noisy,n_fft=self.hp.audio.frame, hop_length = self.hp.audio.shift, window = self.window, center = True, normalized=False, onesided=True)
        estim = torch.stft(estim,n_fft=self.hp.audio.frame, hop_length = self.hp.audio.shift, window = self.window, center = True, normalized=False, onesided=True)
        clean = torch.stft(clean,n_fft=self.hp.audio.frame, hop_length = self.hp.audio.shift, window = self.window, center = True, normalized=False, onesided=True)

        self.log_spec(noisy,estim,clean,step)


if __name__=='__main__':
    data = np.load('input.npy')
    print("input shape : " + str(np.shape(data)))
    spec2plot(data)