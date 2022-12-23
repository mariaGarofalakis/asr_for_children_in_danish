# Language model creation steps
In order to create a language model to boost the ASR model follow the steps from [link](https://huggingface.co/blog/wav2vec2-with-ngram)

Firstly we have to set up the appropriate enviroment and the folders for the kenlm:

**cd to asr_for_children_in_danish/src/language_model and execute**
```
pip install datasets transformers
pip install https://github.com/kpu/kenlm/archive/master.zip pyctcdecode
pip install https://github.com/kpu/kenlm/archive/master.zip
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
```
then execute the file **language_model_dataset.py** in order to extract the .txt  file which you are going to give as input to **kenlm/build/bin/lmplz**, in other words execute the shell comand ("/src/language_model/text.txt" is the path to the .txt file):
**cd ../../** go buck two -> language model directory
```
kenlm/build/bin/lmplz -o 5 <"text.txt" > "5gram.arpa"
```

Where text.txt is the file we extracted from the **language_model_dataset.py**, but it could be just any file of the same form.

Execute **create_language_model_processor.py** in order to get the language model processor
