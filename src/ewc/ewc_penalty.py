import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC
from src.ewc._calculate_fisher_info_matrix import EWC
from src.data.make_dataset import DatasetBuilding

class EWC_Pemalty(object):
    
    def __init__(self, 
                 model,
                 dataset_name, 
                 dataset_dir,
                 absolute_path,
                 model_checkpoint = "chcaa/xls-r-300m-danish-nst-cv9"):
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        old_task_parameters_path = os.path.join(absolute_path,'../../ewc/fisher_info_results/old_task_parameters.pth')
        fisher_matrix_path = os.path.join(absolute_path,'../../ewc/fisher_info_results/fisher_matrix.pth')

        if not((os.path.exists(old_task_parameters_path)) & (os.path.exists(fisher_matrix_path))):
            self.calculate_fisher_info_matrix(model,dataset_name, dataset_dir,absolute_path, model_checkpoint)      
        self.old_task_parameters = torch.load(old_task_parameters_path,map_location=device)
        self.fisher_matrix = torch.load(fisher_matrix_path,map_location=device)

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self.fisher_matrix[n] * (p - self.old_task_parameters[n]) ** 2
            loss += _loss.sum()
        return loss

    def calculate_fisher_info_matrix(self,model,dataset_name, dataset_dir,the_absolute_path,model_checkpoint):

        # here we get a sample from the dataset of the pre-trained model (common voice)
        dataset = DatasetBuilding(dataset_name, dataset_dir)
        train_data, evaluation_data = dataset.make_dataset(model_checkpoint)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=1)
        EWC(model, train_dataloader, the_absolute_path)