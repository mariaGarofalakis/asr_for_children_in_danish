from transformers import Trainer
from typing import  Any,  Dict,  Union
import torch


class CustomTrainer(Trainer):
    
    def set_penalty(self, penalty):
        self.penalty = penalty
    def set_augmentation(self,  augmentation):
        self.augmentation = augmentation

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        if self.augmentation:
            inputs['input_values'] = self.augmentation.perform_data_augmentation(inputs['input_values'].unsqueeze(dim=1)).squeeze(dim=1)
                    
   
        return inputs
        
    def compute_loss(self, model, inputs, return_outputs=False):
       
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.penalty:
            importance = 1000
            loss = loss + importance * self.penalty(self.model)  
        return (loss, outputs) if return_outputs else loss