import numpy as np
import torch
import wandb

try:
    from .plotting import *
except ImportError:
    from plotting import *

"""
W&B drop-in for writer.MyWriter : same log_* API, so train_SE.py / train_SS.py call sites are unchanged.

- x-axis is the custom metric 'global_step' (define_metric), not wandb's internal step,
  so a resumed run (same id, step restarting below the last logged one) is not dropped.
- calls with the same step are buffered and sent as one row; the buffer is flushed when
  the step changes and on close().
- wandb's telemetry import hooks are removed after init (ForkedError in forked DataLoader workers, see __init__).
- tags containing '/' (e.g. 'train/loss_x') are grouped into W&B panel sections.
"""

class MyWandbWriter:
    def __init__(self, project, name=None, config=None, id=None, dir=None, entity=None,
                 mode="online", tags=None, console="auto", n_fft=512, n_hop=128, sr=16000):
        self.run = wandb.init(project=project, name=name, config=config, id=id, dir=dir,
                              entity=entity, mode=mode, tags=tags, resume="allow",
                              settings=wandb.Settings(console=console))
        # DataLoader workers are forked after init. A first import inside a worker (librosa lazily
        # imports core.audio -> joblib on the first rs.load) fires wandb's telemetry import hook,
        # which needs the parent's asyncio thread -> ForkedError in DatasetRIR.__getitem__.
        # Telemetry only; drop the hooks (private API, checked on wandb 0.31.0).
        if hasattr(self.run, "_unregister_telemetry_import_hooks"):
            self.run._unregister_telemetry_import_hooks(self.run.id)
        else:
            print("WARNING::MyWandbWriter : wandb import hooks not removed, "
                  "DataLoader workers may raise ForkedError")
        self.run.define_metric("global_step")
        self.run.define_metric("*", step_metric="global_step")

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.sr = sr
        self.window = torch.hann_window(window_length=n_fft, periodic=True)

        self._buf = {}
        self._buf_step = None

    def _log(self, tag, value, step):
        if self._buf_step is not None and step != self._buf_step:
            self.flush()
        self._buf_step = step
        self._buf[tag] = value

    def flush(self):
        if self._buf:
            self._buf["global_step"] = self._buf_step
            self.run.log(self._buf)
        self._buf = {}
        self._buf_step = None

    def log_value(self, train_loss, step, tag):
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        self._log(tag, train_loss, step)

    def log_train(self, train_loss, step):
        self.log_value(train_loss, step, 'train_loss')

    def log_test(self, test_loss, step):
        self.log_value(test_loss, step, 'test_loss')

    def log_audio(self, wav, label='label', step=0):
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().float().numpy()
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val
        self._log(label, wandb.Audio(wav, sample_rate=self.sr), step)

    def log_MFCC(self, input, output, clean, step):
        noisy = input[0].detach().cpu().numpy()
        estim = input[1].detach().cpu().numpy()
        output = np.expand_dims(output.detach().cpu().numpy(), 0)
        clean = np.expand_dims(clean.detach().cpu().numpy(), 0)

        self._log('noisy', wandb.Image(MFCC2plot(noisy)), step)
        self._log('estim', wandb.Image(MFCC2plot(estim)), step)
        self._log('clean', wandb.Image(MFCC2plot(clean)), step)
        self._log('output', wandb.Image(MFCC2plot(output)), step)

    def log_spec(self, data, label, step):
        self._log(label, wandb.Image(spec2plot(data)), step)

    def log_mag(self, data, label, step):
        self._log(label, wandb.Image(mag2plot(data)), step)

    def log_wav2spec(self, noisy, estim, clean, step):
        # writer.MyWriter.log_wav2spec calls log_spec with a wrong signature; here each is logged separately
        for tag, x in (('stft_noisy', noisy), ('stft_estim', estim), ('stft_clean', clean)):
            X = torch.stft(torch.from_numpy(x), n_fft=self.n_fft, hop_length=self.n_hop,
                           window=self.window, center=True, normalized=False, onesided=True,
                           return_complex=True)
            self.log_spec(X, tag, step)

    def log_DOA_wav(self, data, step, label="Output"):
        self._log(label, wandb.Image(wav2plotDOA(data)), step)

    def close(self):
        self.flush()
        self.run.finish()

    finish = close
