import numpy as np
import torch
import wandb

try:
    from .plotting import *
except ImportError:
    from plotting import *

class MyWandbWriter:
    def __init__(self, project_name, run_name=None, config=None,id=None, n_fft=512, n_hop=128, sr=16000):
        """
        Initialize Weights & Biases Logger.
        
        Args:
            project_name (str): The name of the project where you are sending the new run.
            run_name (str, optional): An individual name for this run.
            config (dict, optional): Hyperparameters to save with the run.
        """
        # Initialize wandb session
        wandb.init(project=project_name, name=run_name, config=config, id=id, resume="allow")

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.sr = sr

        self.window = torch.hann_window(window_length=n_fft, periodic=True,
                                        dtype=None, layout=torch.strided, device=None,
                                        requires_grad=False)

    def log_value(self, train_loss, step, tag):
        # wandb.log tracks continuous metrics using 'step' as an entry key
        wandb.log({tag: train_loss}, step=step)

    def log_train(self, train_loss, step):
        wandb.log({'train_loss': train_loss}, step=step)

    def log_test(self, test_loss, step):
        wandb.log({'test_loss': test_loss}, step=step)

    def log_audio(self, wav, label='label', step=0):
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        
        # Audio normalization
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val
            
        # Log audio data to wandb
        wandb.log({label: wandb.Audio(wav, sample_rate=self.sr)}, step=step)

    def log_MFCC(self, input, output, clean, step):
        input = input.to('cpu')
        output = output.to('cpu')
        clean = clean.to('cpu')

        noisy = input[0]
        estim = input[1]

        noisy = noisy.detach().numpy()
        estim = estim.detach().numpy()
        output = output.detach().numpy()
        clean = clean.detach().numpy()

        output = np.expand_dims(output, 0)
        clean = np.expand_dims(clean, 0)

        noisy = MFCC2plot(noisy)
        estim = MFCC2plot(estim)
        output = MFCC2plot(output)
        clean = MFCC2plot(clean)

        # In wandb.Image, NumPy array shapes (H, W, C) are automatically handled
        wandb.log({
            'noisy': wandb.Image(noisy),
            'estim': wandb.Image(estim),
            'clean': wandb.Image(clean),
            'output': wandb.Image(output)
        }, step=step)

    def log_spec(self, data, label, step):
        # Convert plotted image to wandb.Image
        wandb.log({label: wandb.Image(spec2plot(data))}, step=step)

    def log_mag(self, data, label, step):
        wandb.log({label: wandb.Image(mag2plot(data))}, step=step)
 
    def log_wav2spec(self, noisy, estim, clean, step):
        # Note: The original TensorBoard implementation was missing an internal target function 
        # or had a bug where self.log_spec accepted 3 positional arguments instead of 2.
        # This implementation logs all three components under respective tags.
        noisy_t = torch.from_numpy(noisy)
        estim_t = torch.from_numpy(estim)
        clean_t = torch.from_numpy(clean)

        noisy_stft = torch.stft(noisy_t, n_fft=self.n_fft, hop_length=self.n_hop, window=self.window, center=True, normalized=False, onesided=True)
        estim_stft = torch.stft(estim_t, n_fft=self.n_fft, hop_length=self.n_hop, window=self.window, center=True, normalized=False, onesided=True)
        clean_stft = torch.stft(clean_t, n_fft=self.n_fft, hop_length=self.n_hop, window=self.window, center=True, normalized=False, onesided=True)

        self.log_spec(noisy_stft, 'stft_noisy', step)
        self.log_spec(estim_stft, 'stft_estim', step)
        self.log_spec(clean_stft, 'stft_clean', step)
    
    def log_DOA_wav(self, data, step, label="Output"):
        image = wav2plotDOA(data)
        wandb.log({label: wandb.Image(image)}, step=step)

    def finish(self):
        """Close the wandb run session explicitly."""
        wandb.finish()

if __name__ == '__main__':
    # Usage Example
    # writer = MyWandbWriter(project_name="my_audio_project", run_name="experiment_1")
    pass