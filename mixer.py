import array
import math
import numpy as np
import wave

import scipy.signal as ss
import rir_generator as rir

def setSNR(clean,noise,ratio=0,normalized=False):
    if not normalized : 
        clean = clean/np.max(np.abs(clean))
        noise = noise/np.max(np.abs(noise))

    mean_energy_clean = np.sum(np.power(clean,2))
    mean_energy_noise = np.sum(np.power(noise,2))
    energy_normal = np.sqrt(mean_energy_clean)/np.sqrt(mean_energy_noise)
    SNR_weight = energy_normal/np.sqrt(np.power(10,SNR/10))

    if ratio >= 0 :
        # decrease erergy of noise
        noise = noise*SNR_weight
    else :
        # decrease erergy of clean
        clean = clean / SNR_weight
    
    return clean, noise

# Read clean wav file & noise wav file and mix
# must be len(noise) > len(clean) : random mixing in noise
# return mixed wav
def mix(clean,noise,output,snr=0,normalized=False,h=None):
    if len(noise) < len(clean) : 
        raise ValueError("Noise must be longer than Clean")
    ## random sampling from noise
    start = np.random.randint(0,len(noise)-len(clean))
    noise = noise[start:start+len(clean)]


    ## reverberation
    if h != None : 
        speech = ss.convolve(h,speech)
        noise  = ss.convolve(h_noise,noise)

    ## SNR
    clean, noise = setSNR(clean,noise,snr,normalized=normalized)

    ## mix
    mixed = speech + noise

if __name__ == '__main__' : 

    ## SNR test
    '''
    import librosa
    import soundfile
    SNR =  5

    clean,_ = librosa.load('clean.wav',sr=16000)
    noise,_ = librosa.load('noise.wav',sr=16000)

    clean, noise = setSNR(clean,noise,ratio=SNR)

    soundfile.write('clean_'+str(SNR)+'.wav',clean,16000)
    soundfile.write('noise_'+str(SNR)+'.wav',noise,16000)
    '''

    ## mixing test
    c = 340
    fs = 16000

    L = [5, 4, 3] # Room Dimensions [x y z] (m)
    centerSensors = [1.5 2, 1]
    sensorSpec = [  [0, 0.10, 0],
                    [0, 0.06, 0],
                    [0, 0.02, 0],
                    [0, -0.02, 0],
                    [0, -0.06, 0],
                    [0, -0.10, 0]]
    r = np.add(centerSensors,sensorSpec) # receiver position(s)
    s_s = [4, 3, 1.7] # source position of speech

    s_np = [5, 4, 3] # source position of point noise 

    s_n1 = [5, 4, 3] # source position of diffuse noise
    s_n2 = [0, 4, 3] # source position of diffuse noise
    s_n3 = [5, 0, 3] # source position of diffuse noise
    s_n4 = [5, 0, 3] # source position of diffuse noise
    s_n5 = [5, 4, 0] # source position of diffuse noise
    s_n6 = [0, 4, 0] # source position of diffuse noise
    s_n7 = [5, 0, 0] # source position of diffuse noise
    s_n8 = [0, 0, 0] # source position of diffuse noise

    reverb = 0.4

    
