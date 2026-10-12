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
import re
import sys
from typing import Any, Optional, Union

import whisper
import jiwer
from tqdm.auto import tqdm

def list_audio(fmt, root: Union[str, Path]) -> list[Path]:
    if root is None:
        return []
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in fmt
    )


class WERScorer:
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
            "language": language,                # Specify language to avoid auto-detection variance
            "temperature": 0,                # Disable random sampling
            "beam_size": 5,                  # Fixed beam size
            "best_of": 1,                    # Use only one sample set
            "condition_on_previous_text": False, # Prevent error propagation between segments
            "fp16": fp16                     # Ensure device-level consistency (CPU vs GPU)
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
    def load_refs(cls, trans_root: Union[str, Path]) -> dict[str, str]:
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

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        result = self.model.transcribe(
            str(audio_path),
            verbose=None,
            **self.options,
        )
        return (result.get("text") or "").strip()

    def score_dataset(
        self,
        dataset_name: str,
        refs: dict[str, str],
        audio_files: list[Path],
        verbose: bool = False,
    ) -> dict[str, Any]:
        if verbose : 
            print(f"\n{'='*60}")
            print(f"  Dataset : {dataset_name}  ({len(audio_files)} files)")
            print(f"{'='*60}")

        total_ref_words = 0
        total_err_words = 0.0
        scored_count = 0
        missing_ref = 0

        if verbose:
            pbar = tqdm(audio_files, desc="Scoring", unit="file")
        else :
            pbar = audio_files

        for idx, audio_path in enumerate(pbar):
            utt_id = audio_path.stem
            ref_raw = refs.get(utt_id)
            if ref_raw is None:
                missing_ref += 1
                continue

            hyp_raw = self.transcribe(audio_path)

            ref = self.normalize(ref_raw)
            hyp = self.normalize(hyp_raw)

            wer_val = float(jiwer.wer(ref, hyp))
            ref_words = len(ref.split()) if ref else 0
            total_ref_words += ref_words
            total_err_words += wer_val * ref_words
            scored_count += 1

            if verbose:
                print(f"  [{idx:>4}] {utt_id}  WER={wer_val:.3f}")
                print(f"         REF: {ref}")
                print(f"         HYP: {hyp}")
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
        dir_ref: Union[str, Path],
        dir_audio: Union[str, Path],
        verbose: bool = False,
    ) -> dict[str, Any]:
        refs = self.load_refs(dir_ref)
        audio_root = Path(dir_audio)
        audio_files = list_audio(self.AUDIO_EXTENSIONS, audio_root)

        summary = self.score_dataset(
            dataset_name=audio_root.name,
            refs=refs,
            audio_files=audio_files,
            verbose=verbose,
        )
        wer = summary["overall_wer"]*100
        return wer


if __name__ == "__main__":
    device = "cpu"
    device = "cuda:0"
    model_name = "small" # ~2000MB VRAM
    #model_name = "large-v2" #  ~9000MB VRAM
    dir_audio = "/home/data/Evaluation/LGCTO/672-clean"
    dir_ref = "/home/data/Evaluation/LGCTO/672"

    #for dir_audio in ["/home/data/Evaluation/LGCTO/672-clean", "/home/data/Evaluation/LGCTO/672-babble-16kHz", "/home/data/Evaluation/LGCTO/672-babble"]: 
    for dir_audio in ["/home/data/Evaluation/LGCTO/672-babble-16kHz_avg"]: 
        for model_name in ["small", "large-v2"] : 
            WER_scorer = WERScorer(model_name=model_name, device=device)
            summary = WER_scorer.eval(dir_ref=dir_ref, dir_audio=dir_audio, verbose=False)
            #print(summary)
            print(f"{model_name} | {summary['overall_wer']*100:.2f} % | {dir_audio}")
