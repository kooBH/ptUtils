# utils

## TL;DR

Training/eval helpers: `Loss.py` (multiscale spectral / CosSDR / PIT / ACCDOA /
Pearson losses, `ListLoss` combiner), `WER.py` / `WER_mult.py` (Whisper + jiwer
WER, single- and permutation-invariant multi-target), `metric.py` (PESQ, STOI,
SI-SDR), `writer.py` (TensorBoard; tags must be unique), `GradientStabilizer.py`
(norm-fixing gradient rescaling), `EMA.py` (weight EMA evaluated/saved per epoch
by `train_SS.py`), `schedule.py` (warm-up schedulers).

# NOTE

## WER.py / WER_mult.py (Whisper + jiwer WER)

Transcribe model outputs with Whisper and score WER against LibriSpeech `*.trans.txt` references. Both `eval(...)` methods return **WER in percent** (matching the `%` logging in `train_SE.py` / `train_SS.py`).

- **`WER.py` (`WERScorer`)** — single-target (1 speaker per file). Each audio is matched to its reference by utt_id (the file stem). Used by `train_SE.py`.
- **`WER_mult.py` (`MultiChannelWERScorer`)** — multi-target extraction. The scored audio is the model's **multi-channel output** (one channel per extracted speaker); each channel is transcribed and matched to the reference speakers **permutation-invariantly** (min total word errors over all channel↔reference assignments). Used by `train_SS.py`.

### `WER_mult` reference resolution (important)

`eval(dirs_ref, dir_audio)` takes a **list** of reference directories, one per target speaker:

- **`dirs_ref[0]` = base speaker** (e.g. `672`): matched to each mixture by **utt_id** (the audio file stem).
- **`dirs_ref[1:]` = other target speakers** (e.g. `1580`): matched by **base-order index**, not utt_id. The `i`-th audio file (in base ordering) takes the `i`-th transcript of that speaker (`other[i % len(other)]`).

This mirrors the mixing script `data/ECNR/scripts/mixing/Mix_2target_1interf.py` (`spk2_files[i % len(spk2_files)]`, indexed by the base speaker). **Each non-base speaker pool must have ≥ as many utterances as the base speaker**, and both sides are sorted by utt_id so the ordering reproduces the mix.

Config (`config/common/common_v7_lv2_W10.yaml`):
```yaml
eval:
  ref   : ["/home/data/Evaluation/LGCTO/672", "/home/data/Evaluation/LGCTO/1580"]  # [base, other]
  audio : "/home/data/Evaluation/LGCTO/W10_2target"
  tmp   : "/home/data/Evaluation/LGCTO/tmp_lv1"
  whisper_model : "small"
```

> The earlier version keyed *both* ref dirs by the same utt_id, which only worked for the old LGCTO set (spk1/spk2 shared ids). For LibriSpeech the two targets are different speakers, so every second-speaker lookup missed → `overall_wer = nan` → logged as WER=100. The index-pairing above fixes that.

## GradientStabilizer.py

Implementation of **GradientStabilizer: Fix the Norm, Not the Gradient** (ICML 2026, Huang et al. — `ref/(2026,ICML)29778_GradientStabilizer_Fix_t.pdf`, Algorithm 1).

Optimizer-agnostic, threshold-free alternative to gradient clipping: keeps the gradient's direction but replaces its global L2 magnitude with `rho_t = m_t / sqrt(v_t)`, where `m_t`/`v_t` are EMAs of the gradient norm and squared norm. Guarantees a uniform bound `rho_t <= (1-γ1)/sqrt(1-γ2) · 1/sqrt(1-γ1²/γ2)` regardless of spike size (Lemma 3.6), so raw gradient explosions cannot produce large parameter updates.

Paper defaults (fixed across all their experiments): `gamma1=0.6`, `gamma2=0.999`. Requires `gamma1² < gamma2`. Note: `m_0=v_0=0` with **no bias correction** (the boundedness proof relies on this recursion), so early-step `rho` can reach ~12–16; this is intentional per the paper.

```python
from utils.GradientStabilizer import GradientStabilizer

stabilizer = GradientStabilizer(model.parameters())  # or pass the Module

loss.backward()
rho, R = stabilizer.step()   # rescales p.grad in-place; R = raw ||g||, rho = stabilized norm
optimizer.step()
```

In `train_SS.py` it is enabled via the train config (see `config/train/train_v57_PIT2_SESS_SDR_gradstab.yaml`):

```yaml
train:
  GradientStabilizer :
    use : True
    gamma1 : 0.6
    gamma2 : 0.999
```

Configs without the block run unchanged. When enabled, raw and stabilized norms are logged to TensorBoard as `train/grad_norm_raw` / `train/grad_norm_stab`.

`step()` returns `(rho, R)` for logging (e.g. TensorBoard `train/grad_norm_raw` vs `train/grad_norm_stab`). Zero or non-finite gradient norms are a no-op and do not corrupt the EMA state. `state_dict()`/`load_state_dict()` are provided for checkpoint resume.

## Loss.py — ACCDOA losses

Two PIT losses consume an ACCDOA head (`aux['accdoa']`, `(B, N, T, 3)`; see `models/TF_CorrNet_MISO_v3`). Both take `(estim, vad, azimuth, elevation)`, build the target with `ACCDOA_label`, interpolate `estim` along time when the frame counts differ, and return a scalar — so they are interchangeable in `train_SE.py` (`loss.ACCDOA.type`).

| class | objective |
|-------|-----------|
| `PIT_ACCDOALoss` | `MSE(P_hat, P)` — one term, activity and direction entangled (`type: joint`, default) |
| `PIT_SplitACCDOALoss` | `alpha * MSE(‖P_hat‖, ‖P‖) + beta * I_active * MSE(P_hat/‖P_hat‖, P/‖P‖)` (`type: split`) |

**Why split.** In the joint form the VAD label scales the DoA target: an inactive frame drags the whole vector to 0, so the direction gradient on an active frame is scaled by that frame's activity and — when VAD is sparse — silent frames dominate the objective. The split form gives the magnitude branch the VAD job and the *unit-vector* branch the DoA job, with the direction term evaluated only where the target is active (`‖P‖ > active_thr`) **and averaged over those frames alone**, so the localization gradient does not shrink as the utterance gets sparser.

```yaml
loss:
  ACCDOA :
    use     : True
    type    : "split"   # "joint" -> PIT_ACCDOALoss | "split" -> PIT_SplitACCDOALoss
    weight  : 0.005     # weight of the whole ACCDOA term in the total loss
    max_src : 1         # PIT permutations; match model.num_spks
    deg     : True      # DatasetRIR emits DEGREES
    alpha   : 1.0       # activity (magnitude) term
    beta    : 1.0       # localization (direction) term
    active_thr : 0.5    # active when ||P|| > active_thr (VAD is 0/1)
```

Notes:
- PIT picks **one** permutation on the combined `alpha*norm + beta*dir` objective; both reported terms come from that permutation.
- Unit vectors use `F.normalize(..., eps=1e-3)`, i.e. divide by `max(‖P‖, eps)`. A true `1e-8` guard makes the direction gradient of a near-zero estimate blow up like `1/‖P_hat‖` (~1e5 at init); the floor bounds it (~4 measured).
- A sample with no active frame contributes no direction gradient (the term is exactly 0, not NaN).
- `last_loss_norm` / `last_loss_dir` hold the unweighted terms of the winning permutation (detached) — `train_SE.py` logs them as `train_loss_accdoa_vad` / `train_loss_accdoa_doa` and the `valid_*` counterparts.
- `train_SS.py` still constructs `PIT_ACCDOALoss` unconditionally (it is not wired for `TF_CorrNet_MISO_v3` anyway — see that model's README).

## Loss.py — `MultiscalemwMSELoss`

Multi-scale version of `mwMSELoss` (mel-domain asymmetric weighted error), following the `MultiscaleCosSDRLoss` / `MultiscaleSpectrogramLoss` pattern: the base loss runs at several window sizes and the per-scale scalars are averaged (optionally weighted).

A single `n_fft` fixes one time/frequency trade-off — short windows resolve onsets/transients, long windows resolve pitch harmonics. Unlike the SDR/spectrogram multiscale losses, the mel filterbank itself depends on `n_fft`, so **each scale owns a `mwMSELoss` submodule with its own basis**; they are not shared.

```yaml
loss:
  ListLoss :
    list   : ["MultiscalemwMSELoss", "MultiscaleCosSDRLoss"]
    weight : [1.0, 1.0]
  MultiscalemwMSELoss :
    chunk_sizes : [1024, 512, 256]   # n_fft per scale
    alpha       : 0.9                # asymmetry, as in mwMSELoss
    n_mels      : 80                 # int (shared) or one value per scale, e.g. [80, 40, 40]
    overlap     : 0.25               # hop = n_fft * overlap (SpectrogramLoss convention)
    # weight    : [1.0, 1.0, 0.5]    # optional per-scale weight, default uniform
```

Notes:
- `n_mels > n_fft//2 + 1` raises `ValueError` at construction. A short window cannot support that many narrow low-frequency bands and librosa would silently return all-zero (dead) filters, so the scale would contribute nothing. With `chunk_sizes: [1024, 512, 256]` and a large `n_mels`, pass a list.
- `overlap: 0.25` reproduces the single-scale `mwMSELoss` hop exactly (`torch.stft`'s default `n_fft//4`) — `mwMSELoss` gained an explicit `n_hop` argument (default `None` → `n_fft//4`), so existing configs are unaffected.
- Takes time-domain `[B, L]` like `mwMSELoss` and returns a scalar, so it drops into `ListLoss`. Under `PITLoss` the scalar means the permutation is picked batch-globally rather than per sample — same as the other scalar-returning losses here (`MultiscaleSpectrogramLoss`, …), not specific to this one.

## TensorBoard
Tensorboard 사용시, 클래스가 다르더라도 label이 달라야 한다 


+ BAD
```python
writer.log_audio(log_noisy,"noisy",step)
writer.log_audio(log_estim,"estim",step)
writer.log_audio(log_clean,"clean",step)

writer.log_spec(log_noisy,"noisy",step)
writer.log_spec(log_estim,"estim",step)
writer.log_spec(log_clean,"clean",step)
```
=> audio에서도 "noisy" 라벨 붙은걸 다 띄우려하고, image에서도 "noisy" 붙은 거를 다 띄우려한다. 

+ GOOD
```python
writer.log_audio(log_noisy,"noisy_audio",step)
writer.log_audio(log_estim,"estim_audio",step)
writer.log_audio(log_clean,"clean_audio",step)

writer.log_spec(log_noisy,"noisy_spec",step)
writer.log_spec(log_estim,"estim_spec",step)
writer.log_spec(log_clean,"clean_spec",step)
```

## EMA.py (weight EMA for training)

`ModelEMA(model, decay=0.999, warmup_steps=0)` keeps an exponential moving
average of the model `state_dict` (`update(model)` after every optimizer step,
`swap(model)` context manager to evaluate with the averaged weights,
`state_dict()` returns a plain loadable model state_dict).

Why: with batch size 1 the per-epoch eval WER of the lv2 separators swings by
~20 %-abs between consecutive epochs, so `bestmodel_wer.pt` is the minimum of a
noisy sequence rather than a stable operating point
(`logs/20260915_lv2_WER_cap.md`). `train_SS.py` enables it from the train config:

```yaml
train:
  EMA :
    use : True
    decay : 0.9995      # ~2000-step horizon (2/3 of a 3000-item epoch at batch 1)
    warmup_steps : 0
```

The EMA weights are evaluated every epoch with the same protocol as the raw
weights (TensorBoard `WER_<whisper>_ema`) and saved as `bestmodel_wer_ema.pt` /
`lastmodel_ema.pt` next to the raw checkpoints.
