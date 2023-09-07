import copy
import torch
import numpy as np
import os
from typing import Dict, List, Union
from dataclasses import dataclass
from transformers import Wav2Vec2Processor, TrainerCallback, Trainer
from datasets import load_metric



def save_model_info(model, processor, trainer, absolute_path):
    model.save_pretrained(os.path.join(absolute_path,'../../final_model'))
    processor.save_pretrained(os.path.join(absolute_path,'../../save_processor'))
    
    the_loss_file = os.path.join(absolute_path,'../../the_loss_file.txt')

    with open(the_loss_file, 'w') as f:
        for log_history in trainer.state.log_history:
            print(log_history, file=f)



def get_parameter_names(model, forbidden_layer_types):
            """
            Returns the names of the model parameters that are not inside a forbidden layer.
            """
            result = []
            for name, child in model.named_children():
                result += [
                    f"{name}.{n}"
                    for n in get_parameter_names(child, forbidden_layer_types)
                    if not isinstance(child, tuple(forbidden_layer_types))
                ]
            # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
            result += list(model._parameters.keys())
            return result
class Metrics():
    def __init__(self, dataset):
        self.dataset = dataset

        
    def compute_metrics(self, pred):

        wer_metric = load_metric("wer")
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.dataset.processor.tokenizer.pad_token_id

        pred_str = self.dataset.processor.batch_decode(pred_ids)
        """
        we transform the encoded labels back to the original string 
        by replacing -100 with the pad_token_id and decoding the 
        ids while making sure that consecutive tokens are not 
        grouped to the same token in CTC style
        """
        label_str = self.dataset.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_features = [{"input_values": torch.from_numpy(np.array(feature["input_values"])).to(device)} for feature in features]
        label_features = [{"input_ids": torch.from_numpy(np.array(feature["labels"])).to(device)} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
