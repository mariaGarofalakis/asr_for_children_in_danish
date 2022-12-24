import argparse
from transformers import TrainingArguments, AutoModelForCTC
from src.data.make_dataset import DatasetBuilding
from src.models._costum_trainer import CustomTrainer
from src.data._data_set import DanDataset
from src.models._utils import Metrics, DataCollatorCTCWithPadding, CustomCallback, save_model_info
from src.ewc.ewc_penalty import EWC_Pemalty
from src.data_augmentation._data_augmentation import Data_augmentation

class Model_fine_tuning(object):
    def __init__(self):
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.dataset_dir = args.dataset_dir
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.model_checkpoint = args.model_checkpoint


    def fine_tuning(self):
        data_augmentation = args.data_augm
        ewc = args.ewc
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
                                        output_dir='C:/Users/maria/Desktop/mathimata/diplwmatikh/test/model_check_points',
                                        group_by_length=True,
                                        per_device_train_batch_size=self.batch_size,
                                        gradient_accumulation_steps=500000,
                                        evaluation_strategy="steps",
                                        num_train_epochs=self.num_epochs,
                                        fp16=True,
                                        gradient_checkpointing=True,
                                        save_steps=500,
                                        eval_steps=500,
                                        logging_steps=500,
                                        learning_rate=self.lr,
                                        weight_decay=self.weight_decay,
                                        warmup_steps=1000,
                                        save_total_limit=2,
                                        push_to_hub=False,
                                        )

        trainer = CustomTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=metrics.compute_metrics,
            train_dataset=train_data,
            eval_dataset=evaluation_data,
            tokenizer=dataset.processor.feature_extractor,
        )
        if ewc:
            the_penaldy = EWC_Pemalty(model)
            trainer.set_penalty(the_penaldy)
        else:
            trainer.set_penalty(None)

        if data_augmentation:
            the_augmentation = Data_augmentation()
            trainer.set_augmentation(the_augmentation)
        else:
            trainer.set_augmentation(None)

        trainer.add_callback(CustomCallback(trainer))
        trainer.train()

        save_model_info(model, dataset.processor, trainer)


        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", default=0.0003950115925060478, type=float)
    parser.add_argument("-weight_decay", default=0.005029742789944035, type=float)
    parser.add_argument("-model_checkpoint", default="chcaa/xls-r-300m-danish-nst-cv9", type=str)
    parser.add_argument("-batch_size", default=10, type=int)
    parser.add_argument("-num_epochs", default=100, type=int)
    parser.add_argument("-dataset_dir", default="/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/fine_tuning_dataset/", type=str)
    parser.add_argument("-dataset_name", default="fine_tuning_dataset", type=str)
    parser.add_argument("-data_augm", default=True, type=bool)
    parser.add_argument("-ewc", default=True, type=bool)

    
    args = parser.parse_args()

    fine_tuning_obj = Model_fine_tuning()
    fine_tuning_obj.fine_tuning()
