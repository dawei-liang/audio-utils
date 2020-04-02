# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:33:21 2020

@author: david
"""

import numpy as np


#%%
"""
Alternative: librosa.util.frame. It could be faster in some implementation.
"""
def framing(data, window_length, hop_length):
    """
    Convert 1D time series signals or N-Dimensional frames into a (N+1)-Dimensional array of frames.
    No zero padding, rounding at the end.
    Args:
        data: Input signals.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.
    Returns:
        np.array with as many rows as there are complete frames that can be extracted.
    """
    
    num_samples = data.shape[0]
    frame_array = data[0:window_length]
    # create a new axis as # of frames
    frame_array = frame_array[np.newaxis]  
    start = hop_length
    for _ in range(num_samples):
        end = start + window_length
        if end <= num_samples:
            # framing at the 1st axis
            frame_temp = data[start:end]
            frame_temp = frame_temp[np.newaxis]
            frame_array = np.concatenate((frame_array, frame_temp), axis=0)
        start += hop_length
    return frame_array


#%%

def reconstruct_time_series(frames, hop_length_samples):
    """
    Reconstruct N-Dimensional framed array back to (N-1)-Dimensional frames or 1D time series signals
    Args:
        frames = [# of frames, window length1 in samples, (window length2, ...)]
        hop_length_samples = # of samples skipped between two frames
    return:
        (N-1)-Dimensional frames or 1D time series signals
    """
    new_signal = []
    for i in range(len(frames)-1):
        for j in range(0, hop_length_samples):
            new_signal.append(frames[i, j])
    # Last frame
    for i in range(frames.shape[1]):
        new_signal.append(frames[-1,i])
        
    new_signal = np.asarray(new_signal)
    
    return new_signal