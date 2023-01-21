# ASR_for_children_in_danish
I implement an Automatic Speech Recognition (ASR) system in order to transcribe children voices in danish. The whole implementation was based on the fine-tuning of  ["chcaa/xls-r-300m-danish-nst-cv9"](https://huggingface.co/chcaa/xls-r-300m-danish-nst-cv9)  which is a Wev2Vec2 model fine-tuned for adults speaking in danish. I recoment to follow all the steps above cerfully in order to be able to fine tune the base model.

#Datasets
There are two types of datasets as we can see from the project folders in projects base directory, **baseline_model_dataset** and **fine_tuning_dataset** each dataset has it's own purpose in the project. All datasets have two identical .csv files **train.csv** and **validation.csv**. Both files have the same fields **path** and **audio** : where you have to save the exact path of all your .wav files in the form /home/user/(your path)/wav_file_name.wav (for all .wav files) and **	sentence**: which containes the transcription of each audio file.

 - **baseline_model_dataset** : This dataset is used for the approximation of Fisher information matrix to implement the method of elastic weight consolidation. More specifically in this case we used the [common voice dataset](https://commonvoice.mozilla.org/da/datasets) on which it is known that the baseline model is trained on in order to calculate both the Fisher Information Matrix and the initial parameters of the baseline model. 

- **fine_tuning_dataset** : This dataset is used for the fine-tuning of the base-model, and containes children voices in Danish. We used [Plunkett dataset](https://sla.talkbank.org/TBB/childes/Scandinavian/Danish/Plunkett). Scince it does not contain time stamps you have to cut the audio by yourself and to match the audio with it's transcription. In my case I used one child for training (25min. of trancribed audio) and the other child for validation (4min. of trancribed audio). It is also important to use the audio files where the child is older.

# Build project enviroment
Firstly is recomented to create a python virtual enviroment and install all project's requirements. To install all project requirements cd to asr_for_children_in_danish (base directory) and execute:


```
pip install -r requirements.txt
```
This will download and install all the required python packages in order to execute this project. Moving on in the same directory (asr_for_children_in_danish) execute the commant:
```
python set_up.py install
```
# Data augmentation
To successfully implement data augmentations to make the dataset more generic you have to download and save audio files described bellow:
-  cd to **asr_for_children_in_danish/src/data_augmentation/noise/** and place all the audios from [audio files for background noise](https://www.openslr.org/17/)

- cd to **asr_for_children_in_danish/src/data_augmentation/room/** and place all the audios from  [environmental Impulse Responses ](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html)
