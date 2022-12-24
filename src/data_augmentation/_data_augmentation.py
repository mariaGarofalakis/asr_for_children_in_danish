import random
import os
from torch_audiomentations import SomeOf,AddBackgroundNoise, Gain, ApplyImpulseResponse,PitchShift
from torch_time_stretch import time_stretch

class Data_augmentation(object):
    
    def __init__(self, noise_dir,room_dir):
        
        transforms = [
            Gain(
                min_gain_in_db=-2.0,
                max_gain_in_db=5.0,
                p=0.8

            ),
            AddBackgroundNoise(background_paths= noise_dir,min_snr_in_db =  7.0, max_snr_in_db = 15.0,p=0.5),
            ApplyImpulseResponse(room_dir,p=0.5),
            PitchShift(sample_rate=16000,min_transpose_semitones=-2,max_transpose_semitones=2,p=0.9)
        ]
        self.augmentation = SomeOf((1, 3),transforms,p=0.8)

    def perform_data_augmentation(self,input, sampling_rate=16000):
        augmented_input = time_stretch(self.augmentation(input,sample_rate=sampling_rate),
                     random.uniform(1, 1.2),sample_rate=sampling_rate)
        return augmented_input