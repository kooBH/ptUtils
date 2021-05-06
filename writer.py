import numpy as np
import torch
from tensorboardX import SummaryWriter

try : 
    from .plotting import spec2plot,MFCC2plot
except ImportError:
    from plotting import spec2plot,MFCC2plot

# https://pytorch.org/docs/stable/tensorboard.html

class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp
        self.window = torch.hann_window(window_length=hp.audio.frame, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False)
    def log_value(self, train_loss, step,tag):
        self.add_scalar(tag, train_loss, step)

    def log_train(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_test(self,test_loss,step) : 
        self.add_scalar('test_loss', test_loss, step)

    def log_audio(self,noisy,estim,output,clean,step) : 
        self.add_audio('noisy', noisy, step, self.hp.audio.samplerate)
        self.add_audio('estim', estim, step, self.hp.audio.samplerate)
        self.add_audio('clean', clean, step, self.hp.audio.samplerate)

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
 
    def log_wav2spec(self,noisy,estim,clean,step) :
        noisy = torch.from_numpy(noisy)
        estim = torch.from_numpy(estim)
        clean = torch.from_numpy(clean)

        noisy = torch.stft(noisy,n_fft=self.hp.audio.frame, hop_length = self.hp.audio.shift, window = self.window, center = True, normalized=False, onesided=True)
        estim = torch.stft(estim,n_fft=self.hp.audio.frame, hop_length = self.hp.audio.shift, window = self.window, center = True, normalized=False, onesided=True)
        clean = torch.stft(clean,n_fft=self.hp.audio.frame, hop_length = self.hp.audio.shift, window = self.window, center = True, normalized=False, onesided=True)

        self.log_spec(noisy,estim,clean,step)

def check_MFCC():
    from hparams import HParam
    ## log MFCC test
    hp = HParam("../../config/TEST.yaml")
    log_dir = '/home/nas/user/kbh/MCFE/'+'/'+'log'+'/'+'TEST'

    writer = MyWriter(hp, log_dir)

    input = torch.load('input.pt').to('cpu')
    clean = torch.load('clean.pt').to('cpu')
    output = torch.load('output.pt').to('cpu')

    print('input : ' + str(input.shape))
    print('output : ' + str(output.shape))

    noisy = input[0]
    estim = input[1]

    noisy = noisy.detach().numpy()
    estim = estim.detach().numpy()
    output = output.detach().numpy()
    clean= clean.detach().numpy()


    output = np.expand_dims(output,0)
    clean = np.expand_dims(clean,0)

    print(noisy.shape)
    print(estim.shape)
    print(output.shape)
    print(clean.shape)

    noisy = MFCC2plot(noisy)
    estim = MFCC2plot(estim)
    output = MFCC2plot(output)
    clean = MFCC2plot(clean)

    print('MFCC')
    print(noisy.shape)
    print(estim.shape)
    print(output.shape)
    print(clean.shape)

    writer.log_MFCC(noisy,estim,output,clean,0)

if __name__=='__main__':
    check_MFCC()

