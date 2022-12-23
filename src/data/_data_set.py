import torch
from torch_audiomentations import SomeOf, AddBackgroundNoise, Gain, ApplyImpulseResponse, PitchShift
from torch_time_stretch import time_stretch
import soundfile as sf

class DanDataset(torch.utils.data.Dataset):
    def __init__(self, dset, do_augmentation=True, 
                noise_dir = '/zhome/2f/8/153764/Desktop/test/paradeigmata/noise', 
                room_dir='/zhome/2f/8/153764/Desktop/test/paradeigmata/room'):
        self.dset = dset

        self.do_augmentation = do_augmentation
        if self.do_augmentation:
            self._perform_augmentations(noise_dir, room_dir)



    def _perform_augmentations(self, noise_dir, room_dir):
 
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


    def get_augmentation(self, inputs):
        
         
        aug_audio = time_stretch(
                                self.augmentation(
                                inputs.unsqueeze(dim=1).to("cuda:0"),sample_rate=16000),
                                random.uniform(1, 1.2),sample_rate=16000).squeeze(dim=1)
        
        return aug_audio


    def __getitem__(self, idx):

        if self.do_augmentation:

            input = torch.squeeze(self.get_augmentation(torch.FloatTensor(self.dset['input_values'][idx])),0)
        else:
            input = torch.FloatTensor(self.dset['input_values'][idx])
        
        return {'input_values': input, 'labels': torch.FloatTensor(self.dset['labels'][idx])}

    def __len__(self):
        return len(self.dset)
