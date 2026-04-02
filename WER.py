"""
+ Requirements
Whisper
whisper_normalizer
editdistance
LibriSpeech Corpus

---

$ pip install -U openai-whisper
$ pip install editdistance
$ pip install whisper_normalizer
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

import whisper
from whisper_normalizer.basic import BasicTextNormalizer
import jiwer
import re
import editdistance
import os,glob
import numpy as np
from pathlib import Path
from typing import Any

def WER_librispeech(dir_clean, dir_noisy, model_size = "large") :
    model = whisper.load_model(model_size)
    normalizer = BasicTextNormalizer()

    distance = 0
    total = 0
    val_WER = 0

    list_trans = glob.glob(os.path.join(dir_clean,"**","*.txt"),recursive=True)

    for item in list_trans :
        with open(item,"r") as f:
            txt = f.readlines()
            list_id = [t.split(" ")[0] for t in txt]
            list_tsc = [t.split(list_id[idx])[1].replace("\n","") for idx,t in enumerate(txt)]
            
            # iteration for directory
            for idx in range(len(list_id)):
                result = model.transcribe(os.path.join(dir_noisy,list_id[idx]+".wav"))
                estim = result["text"].upper()
                GT = list_tsc[idx]
                estim = normalizer(estim)
                GT = normalizer(GT)

                t_d = editdistance.eval(estim, GT)
                t_total = len(GT.split(" "))

                distance += t_d
                total += t_total
    val_WER = (distance/total)*100
    return val_WER


glboal_model = None
global_refs = None

_norm_re = re.compile(r"[^a-z0-9\s']+")

def load_refs_librispeech(trans_root: Path) -> dict[str, str]:
    """Load all *.trans.txt files under *trans_root* into a utt_id→text dict."""
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

def normalize(text: str) -> str:
    t = text.lower()
    t = _norm_re.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def WER_librispeech_per_audio(audio, file_path, dir_root,device="cpu"):

    global glboal_model
    global global_refs
    if glboal_model is None:
        model_kwargs: dict[str, Any] = {}
        model_kwargs["device"] = device
        print(f"Loading Whisper model large-v2 {model_kwargs}")
        glboal_model = whisper.load_model("large-v2", **model_kwargs)
        global_refs = load_refs_librispeech(Path(dir_root))
    ids = re.findall(r'\d+', os.path.basename(file_path))
    utt_id = "-".join(ids)
    ref_raw = global_refs.get(utt_id)

    result  = glboal_model.transcribe(
        audio.astype(np.float32),
        language="en",
        fp16=False,
        verbose=None,
    )
    hyp_raw = (result.get("text") or "").strip()

    ref = normalize(ref_raw)
    hyp = normalize(hyp_raw)

    wer_val   = float(jiwer.wer(ref, hyp))
    return wer_val


if __name__ == "__main__" : 

    dir_nosiy = "/home/data/kbh/LibriSpeech_noisy/SNR10"