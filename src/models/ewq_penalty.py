import torch
from torch import nn

class EWC_Pemalty(object):
    
    def __init__(self):
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.old_task_parameters = torch.load('/zhome/2f/8/153764/Desktop/final/old_task_parameters.pth',map_location=device)
        self.fisher_matrix = torch.load('/zhome/2f/8/153764/Desktop/final/fisher_matrix.pth',map_location=device)

    def penalty(self, model: nn.Module):

        loss = 0
        for n, p in model.named_parameters():
            _loss = self.fisher_matrix[n] * (p - self.old_task_parameters[n]) ** 2
            loss += _loss.sum()
      #      print('the loss')
      #      print(loss)
        return loss