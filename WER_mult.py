"""
+ Requirements
Whisper
jiwer

---

$ pip install -U openai-whisper
$ pip install jiwer
https://www.openslr.org/12

You may need to update ffmpeg

$ sudo apt update && sudo apt install ffmpeg

+ Whisper
Size	Parameters	English-only model	Multilingual model	Required VRAM	Relative speed
tiny	39 M	tiny.en	tiny	~1 GB	~32x
base	74 M	base.en	base	~1 GB	~16x
small	244 M	small.en	small	~2 GB	~6x
medium	769 M	medium.en	medium	~5 GB	~2x
large	1550 M	N/A	large	~10 GB	1x

"""
from pathlib import Path
from itertools import permutations
import re
import sys
from typing import Any, Optional, Union

import numpy as np
import soundfile as sf
import torch
import whisper
import jiwer
from tqdm.auto import tqdm


def list_audio(fmt: set[str], root: Union[str, Path]) -> list[Path]:
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in fmt
    )


class MultiChannelWERScorer:
    AUDIO_EXTENSIONS = {".flac", ".wav"}
    _NORM_RE = re.compile(r"[^a-z0-9\s']+")

    def __init__(
        self,
        model_name: str = "large-v2",
        device: str = "cuda:1",
        language: str = "en",
        fp16: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self._model: Optional[whisper.Whisper] = None

        self.options = {
            "language": language,
            "temperature": 0,
            "beam_size": 5,
            "best_of": 1,
            "condition_on_previous_text": False,
            "fp16": fp16,
        }

    @property
    def model(self) -> whisper.Whisper:
        if self._model is None:
            self._model = self.load_model()
        return self._model

    def load_model(self) -> whisper.Whisper:
        try:
            return whisper.load_model(self.model_name, device=self.device)
        except Exception as e:
            print(
                f"ERROR: could not load model '{self.model_name}': {e}",
                file=sys.stderr,
            )
            raise

    @classmethod
    def normalize(cls, text: str) -> str:
        t = text.lower()
        t = cls._NORM_RE.sub(" ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @classmethod
    def load_single_ref(cls, trans_root: Union[str, Path]) -> dict[str, str]:
        """Load transcript mapping (utt_id -> text) from a single directory."""
        trans_root = Path(trans_root)
        refs: dict[str, str] = {}
        for trans_path in sorted(trans_root.rglob("*.trans.txt")):
            for line in trans_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                refs[utt_id.strip()] = text.strip()
        return refs

    @classmethod
    def load_ordered_ref(cls, trans_root: Union[str, Path]) -> list[str]:
        """Load transcripts as a list ordered by utt_id.

        Used for the non-base target speakers, which are paired to each mixture
        by index (position in the base speaker ordering) rather than by utt_id
        -- mirroring the mixing script (``spk2_files[i % len(spk2_files)]``).
        """
        refs = cls.load_single_ref(trans_root)
        return [refs[utt_id] for utt_id in sorted(refs.keys())]

    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Resample to 16kHz if needed and run Whisper transcription."""
        if sample_rate != 16000:
            import torchaudio.functional as F
            tensor_audio = torch.from_numpy(audio_data).float().unsqueeze(0)
            resampled = F.resample(tensor_audio, sample_rate, 16000)
            audio_data = resampled.squeeze(0).numpy()

        audio_data = audio_data.astype(np.float32)

        result = self.model.transcribe(
            audio_data,
            verbose=None,
            **self.options,
        )
        return (result.get("text") or "").strip()

    def transcribe_channels(self, audio_path: Union[str, Path]) -> list[str]:
        """Split audio into channels and transcribe each channel."""
        data, samplerate = sf.read(str(audio_path), dtype="float32")
        
        if data.ndim == 1:
            channels = [data]
        else:
            channels = [data[:, ch] for ch in range(data.shape[1])]

        hypotheses = []
        for ch_data in channels:
            hyp = self.transcribe_audio_data(ch_data, samplerate)
            hypotheses.append(hyp)

        return hypotheses

    @staticmethod
    def _min_permutation_error(refs: list[str], hyps: list[str]) -> float:
        """Minimum total word errors over all channel<->reference assignments.

        ``refs`` and ``hyps`` are normalized strings of equal length. Word errors
        are weighted by each reference's word count so the dataset-level WER is a
        proper micro-average.
        """
        n = len(refs)
        # err[r][h] = word errors when reference r is matched to hypothesis h
        err = [
            [float(jiwer.wer(refs[r], hyps[h])) * len(refs[r].split()) for h in range(n)]
            for r in range(n)
        ]
        best = None
        for perm in permutations(range(n)):
            total = sum(err[r][perm[r]] for r in range(n))
            best = total if best is None else min(best, total)
        return best if best is not None else 0.0

    def score_dataset(
        self,
        dataset_name: str,
        base_ref: dict[str, str],
        other_refs: list[list[str]],
        audio_files: list[Path],
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Permutation-invariant multi-speaker WER.

        The base speaker (``base_ref``, e.g. 672) is matched to each mixture by
        utt_id (the audio file stem). Every other target speaker (``other_refs``)
        is paired by index: the ``i``-th audio file in base ordering takes the
        ``i``-th transcript of that speaker (wrapping with ``% len`` to mirror the
        mixing script). The number of scored channels equals ``1 + len(other_refs)``.
        """
        n_spk = 1 + len(other_refs)
        if verbose:
            print(f"\n{'='*60}")
            print(f"   Dataset : {dataset_name}  ({len(audio_files)} files, {n_spk} spk)")
            print(f"{'='*60}")

        total_ref_words = 0
        total_err_words = 0.0
        scored_count = 0
        missing_ref = 0

        pbar = tqdm(audio_files, desc="Scoring", unit="file") if verbose else audio_files


        for idx, audio_path in enumerate(pbar):
            utt_id = audio_path.stem

            # Base speaker reference by utt_id; other speakers by base-order index.
            base_text = base_ref.get(utt_id)
            if base_text is None or any(len(o) == 0 for o in other_refs):
                missing_ref += 1
                continue

            utt_refs = [base_text] + [o[idx % len(o)] for o in other_refs]

            # Transcribe each output channel -> one hypothesis per speaker.
            hyps_raw = self.transcribe_channels(audio_path)
            if len(hyps_raw) < n_spk:
                missing_ref += 1
                continue

            refs = [self.normalize(r) for r in utt_refs]
            hyps = [self.normalize(h) for h in hyps_raw[:n_spk]]

            min_err_words = self._min_permutation_error(refs, hyps)
            cur_ref_words = sum(len(r.split()) for r in refs)

            total_err_words += min_err_words
            total_ref_words += cur_ref_words
            scored_count += 1

            if verbose:
                file_wer = (min_err_words / cur_ref_words) if cur_ref_words > 0 else 0.0
                print(f"  [{idx:>4}] {utt_id}  Min-Permutation WER={file_wer:.3f}")
                if isinstance(pbar, tqdm):
                    pbar.set_description(
                        f"WER={total_err_words/total_ref_words*100:.2f}"
                        if total_ref_words > 0 else "WER=nan"
                    )

        overall_wer = (
            total_err_words / total_ref_words
            if total_ref_words > 0 else float("nan")
        )

        return {
            "dataset": dataset_name,
            "scored": scored_count,
            "missing_ref": missing_ref,
            "total_ref_words": total_ref_words,
            "overall_wer": overall_wer,
        }

    def eval(
        self,
        dirs_ref: list[Union[str, Path]],
        dir_audio: Union[str, Path],
        verbose: bool = False,
    ) -> float:
        """Evaluate a dataset given reference directories, returning WER in percent.

        ``dirs_ref[0]`` is the base speaker (matched by utt_id); the remaining
        directories are the other target speakers (paired by index). Returns the
        overall WER as a percentage to match ``utils/WER.py`` and the ``%``
        logging in ``train_SS.py``.
        """
        base_ref = self.load_single_ref(dirs_ref[0])
        other_refs = [self.load_ordered_ref(path) for path in dirs_ref[1:]]
        audio_root = Path(dir_audio)
        audio_files = list_audio(self.AUDIO_EXTENSIONS, audio_root)

        summary = self.score_dataset(
            dataset_name=audio_root.name,
            base_ref=base_ref,
            other_refs=other_refs,
            audio_files=audio_files,
            verbose=verbose,
        )
        return summary["overall_wer"] * 100.0


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # dirs_ref[0] = base speaker (matched by utt_id),
    # dirs_ref[1:] = other target speakers (paired by base-order index).
    dirs_ref = [
        "/home/data/Evaluation/LGCTO/672",   # base speaker
        "/home/data/Evaluation/LGCTO/1580",  # other target speaker
    ]

    dir_audios = ["/home/data/Evaluation/LGCTO/W10_2target"]

    for dir_audio in dir_audios:
        for model_name in ["small", "large-v2"]:
            scorer = MultiChannelWERScorer(model_name=model_name, device=device)
            wer_pct = scorer.eval(dirs_ref=dirs_ref, dir_audio=dir_audio, verbose=True)
            print(f"{model_name} | {wer_pct:.2f} % | {dir_audio}")