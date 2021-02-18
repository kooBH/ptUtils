import torch

# (7) weighted-SDR loss
def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.

    # Batch preserving sum for convenience.
    # (6)
    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = torch.sum(orig * est, dim=1)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    alpha_clean = torch.sum(clean**2, dim=1)
    alpha_noise = torch.sum(noise**2, dim=1)

    # alpha : the energy ratio between clean speech and noise
    alpha = alpha_clean / (alpha_clean + alpha_noise + eps)
    wSDR = (alpha * mSDRLoss(clean, clean_est)
            + (1 - alpha) * mSDRLoss(noise, noise_est))
    return torch.mean(wSDR)

