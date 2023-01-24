from transformers import TrainerCallback
import copy
import os
import argparse
from transformers import TrainingArguments, AutoModelForCTC
from src.data.make_dataset import DatasetBuilding
from src.models._costum_trainer import CustomTrainer
from transformers import TrainerCallback
from src.models._utils import Metrics, DataCollatorCTCWithPadding, save_model_info
from src.ewc.ewc_penalty import EWC_Pemalty
from src.data_augmentation._data_augmentation import Data_augmentation



class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = copy.deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            
            return control_copy


class Model_fine_tuning(object):
    def __init__(self):
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.dataset_dir = args.dataset_dir
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.model_checkpoint = args.model_checkpoint


    def fine_tuning(self,absolute_path):
        data_augmentation = args.data_augm
        ewc = args.ewc


        dataset = DatasetBuilding(self.dataset_name, self.dataset_dir)
        train_data, evaluation_data = dataset.make_dataset(self.model_checkpoint)

        data_collator = DataCollatorCTCWithPadding(processor=dataset.processor, padding=True)

        model = AutoModelForCTC.from_pretrained(
        self.model_checkpoint,
        ctc_loss_reduction="mean",
        pad_token_id=dataset.processor.tokenizer.pad_token_id,
        )
        metrics = Metrics(dataset)
        training_args = TrainingArguments(
        output_dir=os.path.join(absolute_path, "../../model_check_points"),
        group_by_length=True,
        per_device_train_batch_size=self.batch_size,
        evaluation_strategy="steps",
        num_train_epochs=self.num_epochs,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=5000000,
        eval_steps=500,
        logging_steps=500,
        learning_rate= self.lr,
        weight_decay = self.weight_decay,
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
            the_penaldy = EWC_Pemalty(model = model,
                                      dataset_name = 'baseline_model_dataset', 
                                      dataset_dir = os.path.join(absolute_path,'../../baseline_model_dataset/'),
                                      absolute_path = absolute_path)
            trainer.set_penalty(the_penaldy)
        else:
            trainer.set_penalty(None)

        if data_augmentation:
            
            the_augmentation = Data_augmentation(noise_dir=os.path.join(absolute_path,"../data_augmentation/noise"),
                                                 room_dir=os.path.join(absolute_path,"../data_augmentation/room"))
            trainer.set_augmentation(the_augmentation)
        else:
            trainer.set_augmentation(None)

        trainer.add_callback(CustomCallback(trainer))
        trainer.train()


        save_model_info(model, dataset.processor, trainer, absolute_path)


if __name__ == "__main__":

    absolute_path = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", default=0.000302, type=float)
    parser.add_argument("-weight_decay", default=0.0056509, type=float)
    parser.add_argument("-model_checkpoint", default="chcaa/xls-r-300m-danish-nst-cv9", type=str)
    parser.add_argument("-batch_size", default=4, type=int)
    parser.add_argument("-num_epochs", default=100, type=int)
    parser.add_argument("-dataset_dir", default=os.path.join(absolute_path, '../../fine_tuning_dataset'), type=str)
    parser.add_argument("-dataset_name", default="fine_tuning_dataset", type=str)
    parser.add_argument("-data_augm", default=True, type=bool)
    parser.add_argument("-ewc", default=True, type=bool)
    args = parser.parse_args()

    fine_tuning_obj = Model_fine_tuning()
    fine_tuning_obj.fine_tuning(absolute_path)