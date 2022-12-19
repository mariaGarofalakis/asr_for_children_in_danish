from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import random
from tqdm.auto import tqdm
from  evaluate import load
import numpy as np
from transformers import get_scheduler
from ewq_penalty import EWC_Pemalty
from torch_time_stretch import time_stretch
from torch_audiomentations import SomeOf,AddBackgroundNoise, Gain, ApplyImpulseResponse,PitchShift


class Fine_tuner():
    def __init__(self, model, train_dataset, eval_dataset, data_collator ,batch_size = 8):
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size,collate_fn=data_collator)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size,collate_fn=data_collator)
        self.model = model

        noise_dir = '/zhome/2f/8/153764/Desktop/test/paradeigmata/noise'
        room_dir = '/zhome/2f/8/153764/Desktop/test/paradeigmata/room'
        transforms = [
            Gain(
                min_gain_in_db=-2.0,
                max_gain_in_db=5.0,
                p=0.8

            ),
            AddBackgroundNoise(background_paths= noise_dir,min_snr_in_db =  7.0, max_snr_in_db = 15.0,p=0.5),
            ApplyImpulseResponse(room_dir,p=0.5),
            PitchShift(sample_rate=16000,min_transpose_semitones=-2,max_transpose_semitones=2,p=0.9)
        ]
        self.augmentation = SomeOf((1, 3),transforms,p=0.8)
        self.the_penaldy = EWC_Pemalty()

    def compute_metrics(self, pred):

        wer_metric = load("wer")
        pred_logits = pred.logits
        pred_ids = np.argmax(pred_logits, axis=-1)
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

    def evaluate_model(self):
        self.model.eval()
        evaluation_wer = []
        for batch in self.eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            evaluation_wer.append(self.compute_metrics(outputs, batch))
        return evaluation_wer/len(evaluation_wer)
        

    def fine_tuning_process(self,num_epochs, learning_rate):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(name="linear", 
                                     optimizer=optimizer, 
                                     num_warmup_steps=0, 
                                     num_training_steps=num_training_steps)

        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        the_losses = []
        for epoch in range(num_epochs):
            train_wer = []
            train_loss = []
            for batch in self.train_dataloader:
                        
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['input_values'] = time_stretch(
                                    self.augmentation(
                                    batch['input_values'].unsqueeze(dim=1),sample_rate=16000),
                                    random.uniform(1, 1.2),sample_rate=16000).squeeze(dim=1)
                outputs = self.model(**batch)
                importance = 1000
                loss = outputs.loss+ importance * self.the_penaldy.penalty(self.model)
                
            
                loss.backward()
                train_loss.append(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                train_wer.append(self.compute_metrics(outputs))         
           
            av_train_wer = train_wer/len(train_wer)
            av_train_loss = train_loss/len(train_loss)
            evaluation_wer = self.evaluate_model()


            print("\'train_loss\': {} \'train_wer\': {}, \'eval_wer\': {}}".format(av_train_loss, av_train_wer, evaluation_wer))
            the_losses.append("\'train_loss\': " + str(av_train_loss) + "\'train_wer\': " + str(av_train_wer) + ", \'eval_wer\': " + str(evaluation_wer) + "}")
        return the_losses