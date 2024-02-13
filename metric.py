import os
import pesq
import torch
import numpy as np
import librosa as rs
from pystoi.stoi import stoi
import scipy

"""
    output : wav[n_target,n_sample]
    target : wav[n_target,n_sample]
"""
def SIR(estim,target, requires_grad=False,device="cuda:0") :
    if estim.shape != target.shape : 
        raise Exception("ERROR::metric.py::SIR:: output shape != target shape | {} != {}".format(output.shape,target.shape))

    if len(estim.shape) != 2 : 
        raise Exception("ERROR::metric.py::SIR:: output dim {} != 2".format(len(output.shape)))
    n_target  = estim.shape[0]
    
    s_target = []
    e_interf = []

    for i in range(n_target) : 
        s_target.append(torch.inner(estim[i],target[i])*target[i]/torch.inner(target[i],target[i]))

        tmp = None
        for j in range(n_target) : 
            if i == j :
                continue
            if tmp is None : 
                tmp = torch.inner(estim[i],target[j])*target[j]/torch.inner(target[j],target[j])
            else : 
                tmp += torch.inner(estim[i],target[j])*target[j]/torch.inner(target[j],target[j])
        e_interf.append(tmp)

    SIR =  torch.tensor(0.0, requires_grad=requires_grad).to(device)
    for i in range(n_target) : 
        SIR += (torch.inner(s_target[i],s_target[i]))/torch.inner(e_interf[i],e_interf[i])
    return 10*torch.log10(SIR)
"""

"""
def PESQ(estim,target,fs=16000,mode="both") :
    if torch.is_tensor(estim) : 
        estim = estim.cpu().detach().numpy()
    if torch.is_tensor(target) : 
        target = target.cpu().detach().numpy()

    if mode =="wb" : 
        val_pesq = pesq.pesq(fs, target, estim, 'wb',on_error=pesq.PesqError.RETURN_VALUES)
    elif mode == "nb" :
        val_pesq = pesq.pesq(fs, target, estim, 'nb',on_error=pesq.PesqError.RETURN_VALUES)
    else :
        val_pesq = pesq.pesq(fs, target, estim, 'wb',on_error=pesq.PesqError.RETURN_VALUES)
        val_pesq += pesq.pesq(fs,target,estim,'nb',on_error=pesq.PesqError.RETURN_VALUES)
        val_pesq /= 2
    return val_pesq

def PESQ_WB(estim,target,fs=16000) :
    return PESQ(estim,target,fs,"wb")

def PESQ_NB(estim,target,fs=16000) :
    return PESQ(estim,target,fs,"nb")

def STOI(estim,target,fs=16000,mode="both") :

    if torch.is_tensor(estim) : 
        estim = estim.cpu().detach().numpy()
    if torch.is_tensor(target) : 
        target = target.cpu().detach().numpy()

    return stoi(target, estim, fs, extended=False)

def SNR(estim,target, requires_grad=False,device="cuda:0") :
    if estim.shape != target.shape : 
        raise Exception("ERROR::metric.py::SIR:: output shape != target shape | {} != {}".format(estim.shape,target.shape))
    estim = torch.Tensor(estim)
    target = torch.Tensor(target)

    s_target = (torch.inner(estim,target)*target/torch.inner(target,target))

    tmp = estim - s_target 
    e_noise = (tmp)

    SNR = (torch.inner(s_target,s_target))/torch.inner(e_noise,e_noise)
    return 10*torch.log10(SNR)

"""
Equivalent to SDR
"""
def SDR(estim,target,requires_grad=False,device="cuda:0"):
    return SNR(estim,target,requires_grad,device)

def SISDR(estim,target,requires_grad=False,device="cuda:0"):
    return SNR(estim,target,requires_grad,device)


class DNSMOS_singleton(object):
    def __new__(cls, primary_model_path,sr=16000):
        if not hasattr(cls,'instance'):
            import onnxruntime as ort
            cls.onnx_sess = ort.InferenceSession(primary_model_path,providers=["CPUExecutionProvider"])
            cls.sr = sr
            cls.instance = super(DNSMOS_singleton, cls).__new__(cls)
            cls.INPUT_LENGTH = 9.01
            print("metric.py::DNSMOS initialized")
        # recycle
        else :
            pass
        return cls.instance
    
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = rs.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (rs.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly
    
    def __call__(self, aud, input_fs):
        fs = self.sr
        if input_fs != fs:
            audio = rs.resample(aud, orig_sr=input_fs, target_sr=self.sr)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(self.INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - self.INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+self.INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            oi = {'input_1': input_features}
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,False)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        clip_dict = {'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        return clip_dict
    
def DNSMOS(estim,target,fs=16000, ret_all = False):
    model = DNSMOS_singleton(os.path.dirname(os.path.abspath(__file__))+"/sig_bak_ovr.onnx",fs)
    clip_dict = model(estim, input_fs=fs)
    if ret_all : 
        return [clip_dict['OVRL'],clip_dict['SIG'],clip_dict['BAK']]
    else :
        return clip_dict["OVRL"]




class SigMOS_singleton(object):
    '''
    MOS Estimator for the P.804 standard.
    See https://arxiv.org/pdf/2309.07385.pdf
    '''
    def __new__(self):
        if not hasattr(self,'instance'):
            import onnxruntime as ort
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model-sigmos_1697718653_41d092e8-epo-200.onnx')

            self.sampling_rate = 48_000
            self.resample_type = 'fft'

            # STFT params
            self.dft_size = 960
            self.frame_size = 480
            self.window_length = 960
            self.window = np.sqrt(np.hanning(int(self.window_length) + 1)[:-1]).astype(np.float32)

            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            self.session = ort.InferenceSession(model_path, options,providers=['CPUExecutionProvider'])
            self.instance = super(SigMOS_singleton, self).__new__(self)
            print("metric.py::SigMOS initialized")

        else :
            pass
        return self.instance

    def stft(self, signal):
        last_frame = len(signal) % self.frame_size
        if last_frame == 0:
            last_frame = self.frame_size

        padded_signal = np.pad(signal, ((self.window_length - self.frame_size, self.window_length - last_frame),))
        frames = rs.util.frame(padded_signal, frame_length=len(self.window), hop_length=self.frame_size, axis=0)
        spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
        return spec.astype(np.complex64)

    @staticmethod
    def compressed_mag_complex(x: np.ndarray, compress_factor=0.3):
        x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)
        x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
        if compress_factor == 1:
            mag = np.sqrt(x2)
        else:
            x = np.power(x2, (compress_factor - 1) / 2) * x
            mag = np.power(x2, compress_factor / 2)

        features = np.concatenate((mag, x), axis=-2)
        features = np.transpose(features, (1, 0, 2))
        return np.expand_dims(features, 0)

    def run(self, audio: np.ndarray, sr=None):

        if sr is not None and sr != self.sampling_rate:
            audio = rs.resample(audio, orig_sr=sr, target_sr=self.sampling_rate, res_type=self.resample_type)
            #print(f"Audio file resampled from {sr} to {self.sampling_rate}!")

        features = self.stft(audio)
        features = self.compressed_mag_complex(features)

        onnx_inputs = {inp.name: features for inp in self.session.get_inputs()}
        output = self.session.run(None, onnx_inputs)[0][0]

        result = {
                'MOS_COL': float(output[0]), 'MOS_DISC': float(output[1]), 'MOS_LOUD': float(output[2]),
                'MOS_NOISE': float(output[3]), 'MOS_REVERB': float(output[4]), 'MOS_SIG': float(output[5]),
                'MOS_OVRL': float(output[6])
            }
        return result
    
def SigMOS(estim,target,fs=16000, ret_all = False): 
    model = SigMOS_singleton()
    if torch.is_tensor(estim) : 
        estim = estim.cpu().detach().numpy()
    clip_dict = model.run(estim, sr=fs)

    if ret_all : 
        return [clip_dict['MOS_OVRL'],clip_dict['MOS_SIG'],clip_dict['MOS_NOISE'],clip_dict['MOS_REVERB'],clip_dict['MOS_DISC'],clip_dict['MOS_LOUD'],clip_dict['MOS_COL']]
    else :
        return clip_dict["MOS_OVRL"]


def run_metric(estim,target,method,fs=16000):

    val = globals()[method](estim,target,fs)
    return val


# Test 
if __name__ == "__main__" :
    x = torch.rand(32000)
    y = torch.rand(32000)
    sr = 16000

    for method in ["PESQ", "STOI", "SNR", "DNSMOS", "SigMOS"] : 
        print(method)
        val = run_metric(x,y, method,fs=sr)
        print(val)

    for method in ["PESQ", "STOI", "SNR", "DNSMOS", "SigMOS"] : 
        print(method)
        val = run_metric(x,y, method,fs=sr)
        print(val)
