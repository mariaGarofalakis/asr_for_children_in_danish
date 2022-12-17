import re
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor


class DatasetBuilding:
    def __init__(self, script_dir, dataset_scrit_path):
        self.the_dataset = load_dataset(name='testd',data_dir=script_dir,path=dataset_scrit_path)
        self.train_data = self.the_dataset["train"]
        self.evaluation_data = self.the_dataset["validation"]
        
        self.processor = None

    def remove_special_characters(self, batch):
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
        return batch

    def prepare_dataset(self, batch):       
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_values"]= batch["input_values"]
        batch["input_length"] = len(batch["input_values"])
        """
        Temporarily sets the tokenizer for processing the input. 
        Useful for encoding the labels when fine-tuning Wav2Vec2.
        """
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["sentence"]).input_ids
        return batch

    def make_dataset(self, model_checkpoint):
  
        self.train_data = self.train_data.cast_column("audio", Audio(sampling_rate=16_000))
        self.evaluation_data = self.evaluation_data.cast_column("audio", Audio(sampling_rate=16_000))

        self.train_data = self.train_data.map(self.remove_special_characters)
        self.evaluation_data = self.evaluation_data.map(self.remove_special_characters)


        self.processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)
        

        self.train_data.map = self.train_data.map(self.prepare_dataset).remove_columns(["path", "audio", "sentence"])
        self.evaluation_data = self.evaluation_data.map(self.prepare_dataset).remove_columns(["path", "audio", "sentence"])

        return self.train_data.map, self.evaluation_data

    