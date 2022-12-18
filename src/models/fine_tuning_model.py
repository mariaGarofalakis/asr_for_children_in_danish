import argparse
from src.data.make_dataset import DatasetBuilding
from src.models._utils import Metrics, DataCollatorCTCWithPadding, CustomCallback
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from _fine_tuning import Fine_tuner

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
        perform_fine_tuning = Fine_tuner(model, train_data, evaluation_data, self.batch_size)

        log_losses = perform_fine_tuning.fine_tuning_process(self.num_epochs,self.lr)
        save_model_info(model, dataset.processor, log_losses)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", default=0.0003950115925060478, type=float)
    parser.add_argument("-model_checkpoint", default="chcaa/xls-r-300m-danish-nst-cv9", type=str)
    parser.add_argument("-batch_size", default=14, type=int)
    parser.add_argument("-num_epochs", default=1, type=int)
    parser.add_argument("-dataset_dir", default="/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/data/", type=str)
    parser.add_argument("-dataset_name", default="data", type=str)

    
    args = parser.parse_args()

    fine_tuning_obj = Model_fine_tuning()
    fine_tuning_obj.fine_tuning()
