import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import TextProcess

class SpecAugment(nn.Module):
    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35, overriding_rate=0.9):
        super().__init__()
        self.rate = rate
        self.overriding_rate = overriding_rate
        self.time_stretch = torchaudio.transforms.TimeStretch()

        self.specAug1 = nn.Sequential(torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
                                     torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
                                     )
        self.specAug2 = nn.Sequential(torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
                                     torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
                                     torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
                                     torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
                                     )
        
        policies = {
                        1: self.policy1,
                        2: self.policy2,
                        3: self.policy3
                    }
        
        self._forward = policies[policy]
    
    def forward(self,x):
        return self._forward(x)
    
    def apply_time_stretch(self, x):
        stft_transform = torchaudio.transforms.Spectrogram(n_fft=400, power=None) # converting to complex spectrogram
        complex_specgram = stft_transform(x)

        transform =  self.time_stretch(complex_specgram, 0.9)
        transform =  self.time_stretch(complex_specgram, 1.2)

        inverse_stft_transform  = torchaudio.transforms.InverseSpectrogram(n_fft=400)  # converting complex spectrogram to non-complex spectrogram 
        non_complex_spectrogram = inverse_stft_transform(transform)

        return non_complex_spectrogram
    
    
    def policy1(self,x):
        probability = torch.rand(1,1).item()
        x = self.apply_time_stretch(x)
        if self.rate > probability:
            return self.specAug1(x)
        else: 
            return x
        
    def policy2(self,x):
        probability = torch.rand(1,1).item()
        x = self.apply_time_stretch(x)
        if self.rate > probability:
            return self.specAug2(x)
        else: 
            return x
    
    def policy3(self,x):
        probability = torch.rand(1,1).item()
        x = self.apply_time_stretch(x)
        if probability > 0.5:
            return self.policy1(x)
        else: 
            return self.policy2(x)
    
    



class LogMelSpec(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.LogMel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=81)

    def forward(self, x):
        x = self.LogMel(x)
        x - np.log(x + 1e-14)
        return x
    
def createMelSpec(sample_rate):
    return LogMelSpec(sample_rate=sample_rate,n_mels=81)


class Data(torch.utils.data.Dataset):
    parameter = {
        "sample_rate" : 16000, "specAug_rate": 0.5,
        "specAug_policy":3, "freq_mask":15, 
        "time_mask":35, "overriding_rate":0.9
    }

    def __init__(self,json_path,sample_rate,specAug_rate,
                 specAug_policy, freq_mask, time_mask, 
                 overriding_rate, log_ex = True):
        self.log_ex = log_ex
        self.text_process = TextProcess()

        print("loading the json data file... \npath : ",json_path)
        self.data = pd.read_json(json_path)

        self.audio_transform = torch.nn.Sequential(
                                LogMelSpec(sample_rate=sample_rate),
                                SpecAugment(rate=specAug_rate,policy=specAug_policy,freq_mask=freq_mask,time_mask=time_mask,overriding_rate=overriding_rate)
                                )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        try:
            file_path = self.data.key.iloc[index]
            waveform, _ = torchaudio.load(file_path)
            label = self.text_process.text_to_int(self.data.text.iloc[index])

            spectrogram = self.audio_transform(waveform) # shape waveform = (channel, feature, time)
            spec_len = spectrogram.shape[-1] //2
            label_len = len(label)

            if spec_len < label_len:
                raise Exception("sepectrogram len is smaller than lable len")
            if spectrogram.shape[0] > 1:
                raise Exception("Dual channel, so skipping the file ",file_path)
            if label_len == 0:
                raise Exception("label len is zero, so skipping the file", file_path)
                        
        except Exception as e:
            if self.log_ex:
                print(str(e),file_path)
            return self.__getitem__(index=index-1 if index!=0 else index +1)
        
        return spectrogram, label,spec_len,label_len
    

    def describe(self):
        return self.data.describe()
        




