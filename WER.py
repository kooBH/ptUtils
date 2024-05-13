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
import editdistance
import os,glob

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


if __name__ == "__main__" : 

    dir_nosiy = "/home/data/kbh/LibriSpeech_noisy/SNR10"