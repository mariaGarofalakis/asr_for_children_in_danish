# ASR_for_children_in_danish
I implement an Automatic Speech Recognition (ASR) system in order to transcribe children voices in danish.The whole implementation was based on the fine-tuning of  "chcaa/xls-r-300m-danish-nst-cv9"  which is a Wev2Vec2 model fine-tuned for adults speaking in danish.


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
-  cd to *asr_for_children_in_danish/src/data_augmentation/noise/* and place all the audios from  [audio files for background noise]         (http://www.openslr.org/17/)
- cd to *asr_for_children_in_danish/src/data_augmentation/room/* and place all the audios from  [environmental Impulse Responses ](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html)
