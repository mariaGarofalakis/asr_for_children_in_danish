# ASR_for_children_in_danish
In order to create a language model to boost the ASR model follow the steps from [link](https://huggingface.co/blog/wav2vec2-with-ngram)

Firstly we have to set up the appropriate enviroment and the folders for the kenlm:

```
pip install datasets transformers
pip install https://github.com/kpu/kenlm/archive/master.zip pyctcdecode
pip install https://github.com/kpu/kenlm/archive/master.zip
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
ls kenlm/build/bin
```
then execute the file **language_model_dataset.py** in order to extract the .txt  file which you are going to give as input to **kenlm/build/bin/lmplz**, in other words execute the shell comand:

```
kenlm/build/bin/lmplz -o 5 <"/content/gdrive/MyDrive/text.txt" > "5gram.arpa"
```

Where text.txt is the file we extracted from the **language_model_dataset.py**, but it could be just any file of the same form.

Execute **create_language_model_processor.py** in order to get the language model processor