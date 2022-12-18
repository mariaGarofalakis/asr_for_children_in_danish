from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import evaluate


class Fine_tuner():
    def __init__(self, model, train_dataset, eval_dataset, batch_size = 8):
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size)
        self.model = model

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

    def evaluate_model():
        self.model.eval()
        evaluation_wer = []
        for batch in self.eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            evaluation_wer.append(compute_metrics(outputs))
        return evaluation_wer/len(evaluation_wer)
        

    def fine_tuning_process(num_epochs, learning_rate):
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        num_training_steps = num_epochs * len(train_dataloader)
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
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                train_loss.append(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                train_wer.append(compute_metrics(outputs))         
           
            av_train_wer = train_wer/len(train_wer)
            av_train_loss = train_loss/len(train_loss)
            evaluation_wer = evaluate_model()


            print("\'train_loss\': {} \'train_wer\': {}, \'eval_wer\': {}}".format(av_train_loss, av_train_wer, evaluation_wer))
            the_losses.append("\'train_loss\': " + str(av_train_loss) + "\'train_wer\': " + str(av_train_wer) + ", \'eval_wer\': " + str(evaluation_wer) + "}")
        return the_losses