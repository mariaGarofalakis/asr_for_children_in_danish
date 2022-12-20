import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC
from _calculate_fisher_info_matrix import EWC
from src.data.make_dataset import DatasetBuilding

class EWC_Pemalty(object):
    
    def __init__(self, 
                 dataset_name = 'baseline_model_dataset', 
                 dataset_dir = '/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/data/baseline_model_dataset/',
                 model_checkpoint = "chcaa/xls-r-300m-danish-nst-cv9"):
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        old_task_parameters_path = '/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/src/interpret_model/ewc/fisher_info_results/old_task_parameters.pth'
        fisher_matrix_path = '/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/src/interpret_model/ewc/fisher_info_results/fisher_matrix.pth'

        if not((os.path.exists(old_task_parameters_path)) & (os.path.exists(fisher_matrix_path))):
            self.calculate_fisher_info_matrix(dataset_name, dataset_dir,model_checkpoint)      
        self.old_task_parameters = torch.load(old_task_parameters_path,map_location=device)
        self.fisher_matrix = torch.load(fisher_matrix_path,map_location=device)

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self.fisher_matrix[n] * (p - self.old_task_parameters[n]) ** 2
            loss += _loss.sum()
        return loss

    def calculate_fisher_info_matrix(dataset_name, dataset_dir, model_checkpoint):

        # here we get a sample from the dataset of the pre-trained model (common voice)
        dataset = DatasetBuilding(dataset_name, dataset_dir)
        train_data, evaluation_data = dataset.make_dataset(model_checkpoint)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=1)
        model = AutoModelForCTC.from_pretrained(
            model_checkpoint,
            ctc_loss_reduction="mean",
            pad_token_id=dataset.processor.tokenizer.pad_token_id,
        )
        EWC(model, train_dataloader)