from copy import deepcopy
import torch
from torch import nn
import torch
from src.utils._utils import var2device
import os



class EWC(object):
    
    """
    Class to calculate the Fisher Information Matrix
    used in the Elastic Weight Consolidation portion
    of the loss function
    """
    
    def __init__(self, model, dataset,absolute_path):
        self.absolute_path = absolute_path
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(device) #pretrained model
        self.dataset = dataset #samples from the old task or tasks
        
        
        # n is the string name of the parameter matrix p, aka theta, aka weights
        # in self.params we reference all of those weights that are open to
        # being updated by the gradient
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        
        # make a copy of the old weights, ie theta_A,star, ie 𝜃∗A, in the loss equation
        # we need this to calculate (𝜃 - 𝜃∗A)^2 because self.params will be changing 
        # upon every backward pass and parameter update by the optimizer
        self._means = {}
        
        for n, p in deepcopy(self.params).items():
            self._means[n] = var2device(p.data)
        
        # calculate the fisher information matrix 
        self._precision_matrices = self._diag_fisher()


    def _write_results_to_files(self, the_matrix, file_name):
        path =  os.path.join(absolute_path, "../../ewc/fisher_info_results/")
        torch.save(the_matrix, path+file_name +".pth")



    def _diag_fisher(self):
        
        # save a copy of the zero'd out version of
        # each layer's parameters of the same shape
        # to precision_matrices[n]
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = var2device(p.data)

        # we need the model to calculate the gradient but
        # we have no intention in this step to actually update the model
        # that will have to wait for the combining of this EWC loss term
        # with the new task's loss term
        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            # remove channel dim, these are greyscale, not color rgb images
            # bs,1,h,w -> bs,h,w
            del input['input_length']
            input['input_values'] = var2device(torch.unsqueeze(torch.FloatTensor(input['input_values']), dim=0))
            input['labels'] = var2device(torch.FloatTensor(input['labels']))
       
            outputs = self.model(**input)
            loss = outputs.loss
           
            loss.backward()

            k=0
            for n, p in self.model.named_parameters():
                if (k==0):
                    precision_matrices[n].data=torch.zeros(p.shape)

                else:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

                k=k+1

        precision_matrices = {n: p for n, p in precision_matrices.items()}

        self._write_results_to_files(precision_matrices, 'fisher_matrix')
        self._write_results_to_files(self._means, 'old_task_parameters')









