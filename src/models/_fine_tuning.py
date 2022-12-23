from torch.utils.data import DataLoader
#from torch.optim import AdamW
from transformers.optimization import AdamW
import torch
import random
from tqdm.auto import tqdm
from evaluate import load
from transformers import get_scheduler
from src.ewc.ewc_penalty import EWC_Pemalty
from src.data_augmentation._data_augmentation import Data_augmentation
from src.models._utils import get_parameter_names

class Fine_tuner():
    def __init__(self, model, train_dataset, eval_dataset, data_collator,data_augmentation = True, ewc =True ,batch_size = 8):
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size,collate_fn=data_collator)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size,collate_fn=data_collator)
        self.model = model
        if data_augmentation:
            self.augmentation = Data_augmentation()
        if ewc:
            self.the_penaldy = EWC_Pemalty(self.model)

        

    def create_optimizer(self,learning_rate,weight_decay):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model
        ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]

        
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
    
        optimizer_cls = AdamW
        optimizer_kwargs = {'lr': learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-08}
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                
        return self.optimizer


    def compute_metrics(self, pred, reference_labels, processor):
        wer_metric = load("wer")
        pred_logits = pred.logits
        pred_ids = torch.argmax(pred_logits.cpu(), axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        wer = wer_metric.compute(predictions=pred_str, references=reference_labels)

        return wer

    def evaluate_model(self, processor):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        evaluation_wer = []
        for batch in self.eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            label_str = processor.batch_decode(batch['labels'], group_tokens=False)
            evaluation_wer.append(self.compute_metrics(outputs, label_str, processor))
        return sum(evaluation_wer)/len(evaluation_wer)
        

    def fine_tuning_process(self, processor, num_epochs, learning_rate,weight_decay):
        self.create_optimizer(learning_rate,weight_decay)
   #     optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(name="linear", 
                                     optimizer=self.optimizer, 
                                     num_warmup_steps=0, 
                                     num_training_steps=num_training_steps)

        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        the_losses = []
        for epoch in range(num_epochs):
            train_wer = []
            for batch in self.train_dataloader:
                        
                batch = {k: v.to(device) for k, v in batch.items()}
                
                if self.augmentation:
                    batch['input_values'] = self.augmentation.perform_data_augmentation(batch['input_values'].unsqueeze(dim=1)).squeeze(dim=1)
                    
                label_str = processor.batch_decode(batch['labels'], group_tokens=False)
                outputs = self.model(**batch)
                if self.the_penaldy:
                    importance = 1000
                else:
                    importance = 0

                loss = outputs.loss+ importance * self.the_penaldy.penalty(self.model)
                
            
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

                train_wer.append(self.compute_metrics(outputs,label_str,processor))         
           
            av_train_wer = sum(train_wer)/len(train_wer)
            evaluation_wer = self.evaluate_model(processor)


            print("\'train_wer\': " + str(av_train_wer) + ", \'eval_wer\': " + str(evaluation_wer) + "}")
            the_losses.append("\'train_wer\': " + str(av_train_wer) + ", \'eval_wer\': " + str(evaluation_wer) + "}")
        return the_losses