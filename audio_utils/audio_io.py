# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:30:38 2020

@author: david
"""

from scipy.io import wavfile
import resampy
import numpy as np


#%%
def read_audio_data(file):
    '''read audio, only support 16-bit depth'''
    rate, wav_data = wavfile.read(file)
    assert wav_data.dtype == np.int16, 'Not support: %r' % wav_data.dtype  # check input audio rate(int16)
    scaled_data = wav_data / 32768.0   # 16bit standardization
    return rate, scaled_data

#%%
def audio_pre_processing(data, sr, sr_new):
    # Convert to mono.
    try:
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)
    except:
        pass
    # Resampling the data to specified rate
    if sr != sr_new:
      data = resampy.resample(data, sr, sr_new)
    return data

#%%
def write_audio_data(filename, rate, wav_data):
    '''write normalized audio signals with 16 bit depth to a wave file'''
    wav_data = wav_data * 32768.0   # 16bit
    wav_data = wav_data.astype(np.int16)
    wavfile.write(filename, rate, wav_data)
    print(filename + ' Saved')