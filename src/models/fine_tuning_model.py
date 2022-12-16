import argparse
from data.make_dataset import DatasetBuilding
import _utils
from transformers import AutoModelForCTC, TrainingArguments, Trainer

class Model_fine_tuning(object):
    def __init__(self):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.dataset_dir = args.dataset_dir
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.model_checkpoint = args.model_checkpoint


    def fine_tuning(self):

        dataset = DatasetBuilding(self.dataset_dir, self.dataset_dir + self.dataset_name)
        dataset.make_dataset(self.model_checkpoint)

        data_collator = _utils.DataCollatorCTCWithPadding(processor=dataset.processor, padding=True)
        metrics = _utils.Metrics(dataset)

        model = AutoModelForCTC.from_pretrained(
            self.model_checkpoint,
            ctc_loss_reduction="mean",
            pad_token_id=dataset.processor.tokenizer.pad_token_id,
        )

        training_args = TrainingArguments(
        output_dir='/zhome/2f/8/153764/Desktop/test/model_check_points',
        group_by_length=True,
        per_device_train_batch_size=self.batch_size,
        evaluation_strategy="steps",
        num_train_epochs=self.num_epocs,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=5000000,
        eval_steps=500,
        logging_steps=500,
        learning_rate=self.lr, 
        weight_decay=0.005029742789944035, 
        warmup_steps=1000,
        save_total_limit=2,
        push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=metrics.compute_metrics,
            train_dataset=dataset.train_data,
            eval_dataset=dataset.evaluation_data,
            tokenizer=dataset.processor.feature_extractor,
        )



        trainer.add_callback(_utils.CustomCallback(trainer))
        trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", default=0.0003950115925060478, type=float)
    parser.add_argument("-model_checkpoint", default="chcaa/xls-r-300m-danish-nst-cv9", type=str)
    parser.add_argument("-batch_size", default=14, type=int)
    parser.add_argument("-num_epochs", default=100, type=int)
    parser.add_argument("-dataset_dir", default="/zhome/2f/8/153764/Desktop/final/", type=str)
    parser.add_argument("-dataset_name", default="test_data_augm", type=str)

    
    args = parser.parse_args()