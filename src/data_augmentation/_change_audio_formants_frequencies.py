import parselmouth
from parselmouth.praat import call
import glob, os
import random
import librosa
import soundfile as sf
import glob, os
import random
import torch
import argparse

def change_chanel(directory,the_file):
    filename = directory+the_file
    y, sr = librosa.load(filename, mono=False)
    y_mono = librosa.to_mono(y)
    y_8k = librosa.resample(y_mono, orig_sr=sr, target_sr=16000)

    data =  torch.from_numpy(y_8k)
    sf.write('C:/Users/maria/Desktop/diplwmatikh/change_chanel/'+the_file, data.to(torch.float32).to("cpu"), 16000)



def change_formants(directory,the_file):
    filename = directory+the_file
    sound = parselmouth.Sound(filename)
    pitch = sound.to_pitch()
    medain_pitch = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")

    if medain_pitch > 115:
      time_shift=random.uniform(1.1, 1.25)
      if medain_pitch < 200 :
        factor = random.uniform(1.35, 1.4)
        medain_pitch = 265 + random.uniform(-5, 15 )
      else:
        factor = random.uniform(1.4, 1.45)
        medain_pitch = 285 + random.uniform(-5, 5 )
      manipulated_sound = call(sound, "Change gender", 100,
                          500, factor, medain_pitch, 1, time_shift)
      new_file = "C:/Users/maria/Desktop/AAAAAAAAAAAAAAAAAAAAa/change_formants_dom4kimh_allo/"+the_file
      manipulated_sound.save(new_file, "WAV")



def iterate_through_files():
    directory = args.directory
    os.chdir(directory)
    for file in glob.glob("*.wav"):
        print(file)       
        change_chanel(directory,file)
        change_formants(directory,file)

        

directory = 'C:/Users/maria/Desktop/diplwmatikh/change_chanel/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-directory", default="path to the directory where all the audio files are saved", type=str)
    args = parser.parse_args()
    iterate_through_files()


