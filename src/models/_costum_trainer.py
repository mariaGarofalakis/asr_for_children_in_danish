from transformers import Trainer

class CustomTrainer(Trainer):
    
    def set_penalty(self, penalty, ewc=True):
        self.penalty = penalty
        
    def compute_loss(self, model, inputs, return_outputs=False):
       
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.penalty:
            importance = 1000
            loss = loss + importance * self.penalty(self.model)  
        return (loss, outputs) if return_outputs else loss