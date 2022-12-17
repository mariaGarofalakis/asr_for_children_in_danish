import argparse
from src.data.make_dataset import DatasetBuilding
from src.models._utils import Metrics, DataCollatorCTCWithPadding, CustomCallback
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

        dataset = DatasetBuilding(self.dataset_name, self.dataset_dir)
        train_data, evaluation_data = dataset.make_dataset(self.model_checkpoint)

        data_collator = DataCollatorCTCWithPadding(processor=dataset.processor, padding=True)
        metrics = Metrics(dataset)

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
        num_train_epochs=self.num_epochs,
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
            train_dataset=train_data,
            eval_dataset=evaluation_data,
            tokenizer=dataset.processor.feature_extractor,
        )



        trainer.add_callback(CustomCallback(trainer))
        trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", default=0.0003950115925060478, type=float)
    parser.add_argument("-model_checkpoint", default="chcaa/xls-r-300m-danish-nst-cv9", type=str)
    parser.add_argument("-batch_size", default=14, type=int)
    parser.add_argument("-num_epochs", default=100, type=int)
    parser.add_argument("-dataset_dir", default="/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/data/", type=str)
    parser.add_argument("-dataset_name", default="data", type=str)

    
    args = parser.parse_args()

    fine_tuning_obj = Model_fine_tuning()
    fine_tuning_obj.fine_tuning()
