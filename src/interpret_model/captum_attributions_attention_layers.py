from dataset_building import DatasetBuilding
from set_up_trainer import DataCollatorCTCWithPadding
from transformers import TrainerCallback
from datasets import load_metric
import numpy as np
from transformers import AutoModelForCTC
from datasets import load_metric
from transformers import TrainerCallback,Wav2Vec2Processor,Wav2Vec2ForCTC
from transformers import TrainingArguments, AutoTokenizer, AutoConfig
from transformers import Trainer
import torch

from datasets import load_dataset, Audio
from captum.attr import visualization as viz
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerActivation




batch_size = 1
num_epocs = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_checkpoint = "/zhome/2f/8/153764/Desktop/final/base_line_with_baby/all/"
processor = Wav2Vec2Processor.from_pretrained(
    "/zhome/2f/8/153764/Desktop/final/base_line_with_baby/all/")


model = Wav2Vec2ForCTC.from_pretrained(
      "/zhome/2f/8/153764/Desktop/final/base_line_with_baby/all/", output_attentions=True)


model.to(device)
model.eval()
model.zero_grad()


def var2device(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, requires_grad=True, **kwargs)


def predict(inputs):

    output = model(inputs)
    return output.logits,  output.attentions



def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred , _= predict(inputs)
    pred = pred[position]
    # ousiastika gurname gia kathe timestamp th timh tou megaluterou grammatos
    return pred.max(1).values

    

def construct_input_ref_pair(inputs):
    
    del inputs['input_length']
    inputs['input_values'] = var2device(torch.unsqueeze(torch.FloatTensor(inputs['input_values']), dim=0))
    inputs['labels'] = torch.FloatTensor(inputs['labels'])

    
    input_ids =  var2device(processor(
    inputs['input_values'], return_tensors="pt", padding="longest"
    ).input_values)

    ref_value = var2device(torch.zeros(inputs['input_values'].shape).type(torch.FloatTensor))
    ref_ids = var2device(processor(
    ref_value, return_tensors="pt", padding="longest"
    ).input_values)

    return input_ids, ref_ids, len(input_ids)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

if torch.__version__ >= '1.7.0':
    norm_fn = torch.linalg.norm
else:
    norm_fn = torch.norm

config = AutoConfig.from_pretrained(model_checkpoint)

tokenizer_type = config.model_type if config.tokenizer_class is None else None
config = config if config.tokenizer_class is not None else None

tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            config=config,
            tokenizer_type=tokenizer_type,
            bos_token="<s>",eos_token="</s>",unk_token="<unk>",pad_token="<pad>",word_delimiter_token="|"
        )

dataset = DatasetBuilding('test_dataset_baby','/zhome/2f/8/153764/Desktop/final/test_dataset_baby/')
maria_prepared_train, maria_prepared_test = dataset.auto_finalizing_dataset(model_checkpoint)




#Interpreting Layer Outputs and Self-Attention Matrices in each Layer
inputs = maria_prepared_test[20]
del inputs['input_length']
inputs['input_values'] = var2device(torch.unsqueeze(torch.FloatTensor(inputs['input_values']), dim=0))
inputs['labels'] = var2device(torch.FloatTensor(inputs['labels']))

ref_value = inputs.copy()
ref_value['input_values'] = var2device(torch.zeros(inputs['input_values'].shape).type(torch.FloatTensor))
ref_value['labels'] = var2device(torch.zeros(inputs['labels'].shape).type(torch.FloatTensor))




"""


lc = LayerConductance(squad_pos_forward_func, model.wav2vec2.encoder.layers[16])

layer_attributions, error = lc.attribute(inputs=inputs['input_values'],
                                baselines=ref_value['input_values'],
                                return_convergence_delta=True)

attention_layer_attrs = summarize_attributions(layer_attributions[1]).cpu().detach().tolist()
all_attention_atributions_output = layer_attributions[0].cpu().detach().tolist()
all_attention_atributions_attention = layer_attributions[1].cpu().detach().tolist()



with open(r'/zhome/2f/8/153764/Desktop/final/output_attention_layers16.txt', 'w') as fp:
    for item in all_attention_atributions_output:
        # write each item on a new line
        fp.write("%s\n" % all_attention_atributions_output)

with open(r'/zhome/2f/8/153764/Desktop/final/attention_attention_layers16.txt', 'w') as fp:
    for item in all_attention_atributions_attention:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/zhome/2f/8/153764/Desktop/final/all_layers_attribution16.txt', 'w') as fp:
    for item in attention_layer_attrs:
        # write each item on a new line
        fp.write("%s\n" % item)





lc = LayerConductance(squad_pos_forward_func, model.wav2vec2.encoder.layers[17])

layer_attributions, error = lc.attribute(inputs=inputs['input_values'],
                                baselines=ref_value['input_values'],
                                return_convergence_delta=True)

attention_layer_attrs = summarize_attributions(layer_attributions[1]).cpu().detach().tolist()
all_attention_atributions_output = layer_attributions[0].cpu().detach().tolist()
all_attention_atributions_attention = layer_attributions[1].cpu().detach().tolist()



with open(r'/zhome/2f/8/153764/Desktop/final/output_attention_layers17.txt', 'w') as fp:
    for item in all_attention_atributions_output:
        # write each item on a new line
        fp.write("%s\n" % all_attention_atributions_output)

with open(r'/zhome/2f/8/153764/Desktop/final/attention_attention_layers17.txt', 'w') as fp:
    for item in all_attention_atributions_attention:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/zhome/2f/8/153764/Desktop/final/all_layers_attribution17.txt', 'w') as fp:
    for item in attention_layer_attrs:
        # write each item on a new line
        fp.write("%s\n" % item)









lc = LayerConductance(squad_pos_forward_func, model.wav2vec2.encoder.layers[18])

layer_attributions, error = lc.attribute(inputs=inputs['input_values'],
                                baselines=ref_value['input_values'],
                                return_convergence_delta=True)
attention_layer_attrs = summarize_attributions(layer_attributions[1]).cpu().detach().tolist()
all_attention_atributions_output = layer_attributions[0].cpu().detach().tolist()
all_attention_atributions_attention = layer_attributions[1].cpu().detach().tolist()

with open(r'/zhome/2f/8/153764/Desktop/final/output_attention_layers18.txt', 'w') as fp:
    for item in all_attention_atributions_output:
        # write each item on a new line
        fp.write("%s\n" % all_attention_atributions_output)

with open(r'/zhome/2f/8/153764/Desktop/final/attention_attention_layers18.txt', 'w') as fp:
    for item in all_attention_atributions_attention:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/zhome/2f/8/153764/Desktop/final/all_layers_attribution18.txt', 'w') as fp:
    for item in attention_layer_attrs:
        # write each item on a new line
        fp.write("%s\n" % item)



lc = LayerConductance(squad_pos_forward_func, model.wav2vec2.encoder.layers[19])

layer_attributions, error = lc.attribute(inputs=inputs['input_values'],
                                baselines=ref_value['input_values'],
                                return_convergence_delta=True)
attention_layer_attrs = summarize_attributions(layer_attributions[1]).cpu().detach().tolist()
all_attention_atributions_output = layer_attributions[0].cpu().detach().tolist()
all_attention_atributions_attention = layer_attributions[1].cpu().detach().tolist()

with open(r'/zhome/2f/8/153764/Desktop/final/output_attention_layers19.txt', 'w') as fp:
    for item in all_attention_atributions_output:
        # write each item on a new line
        fp.write("%s\n" % all_attention_atributions_output)

with open(r'/zhome/2f/8/153764/Desktop/final/attention_attention_layers19.txt', 'w') as fp:
    for item in all_attention_atributions_attention:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/zhome/2f/8/153764/Desktop/final/all_layers_attribution19.txt', 'w') as fp:
    for item in attention_layer_attrs:
        # write each item on a new line
        fp.write("%s\n" % item)


lc = LayerConductance(squad_pos_forward_func, model.wav2vec2.encoder.layers[20])

layer_attributions, error = lc.attribute(inputs=inputs['input_values'],
                                baselines=ref_value['input_values'],
                                return_convergence_delta=True)
attention_layer_attrs = summarize_attributions(layer_attributions[1]).cpu().detach().tolist()
all_attention_atributions_output = layer_attributions[0].cpu().detach().tolist()
all_attention_atributions_attention = layer_attributions[1].cpu().detach().tolist()

with open(r'/zhome/2f/8/153764/Desktop/final/output_attention_layers20.txt', 'w') as fp:
    for item in all_attention_atributions_output:
        # write each item on a new line
        fp.write("%s\n" % all_attention_atributions_output)

with open(r'/zhome/2f/8/153764/Desktop/final/attention_attention_layers20.txt', 'w') as fp:
    for item in all_attention_atributions_attention:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/zhome/2f/8/153764/Desktop/final/all_layers_attribution20.txt', 'w') as fp:
    for item in attention_layer_attrs:
        # write each item on a new line
        fp.write("%s\n" % item)

lc = LayerConductance(squad_pos_forward_func, model.wav2vec2.encoder.layers[21])

layer_attributions, error = lc.attribute(inputs=inputs['input_values'],
                                baselines=ref_value['input_values'],
                                return_convergence_delta=True)
attention_layer_attrs = summarize_attributions(layer_attributions[1]).cpu().detach().tolist()
all_attention_atributions_output = layer_attributions[0].cpu().detach().tolist()
all_attention_atributions_attention = layer_attributions[1].cpu().detach().tolist()

with open(r'/zhome/2f/8/153764/Desktop/final/output_attention_layers21.txt', 'w') as fp:
    for item in all_attention_atributions_output:
        # write each item on a new line
        fp.write("%s\n" % all_attention_atributions_output)

with open(r'/zhome/2f/8/153764/Desktop/final/attention_attention_layers21.txt', 'w') as fp:
    for item in all_attention_atributions_attention:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/zhome/2f/8/153764/Desktop/final/all_layers_attribution21.txt', 'w') as fp:
    for item in attention_layer_attrs:
        # write each item on a new line
        fp.write("%s\n" % item)


lc = LayerConductance(squad_pos_forward_func, model.wav2vec2.encoder.layers[22])

layer_attributions, error = lc.attribute(inputs=inputs['input_values'],
                                baselines=ref_value['input_values'],
                                return_convergence_delta=True)
attention_layer_attrs = summarize_attributions(layer_attributions[1]).cpu().detach().tolist()
all_attention_atributions_output = layer_attributions[0].cpu().detach().tolist()
all_attention_atributions_attention = layer_attributions[1].cpu().detach().tolist()

with open(r'/zhome/2f/8/153764/Desktop/final/output_attention_layers22.txt', 'w') as fp:
    for item in all_attention_atributions_output:
        # write each item on a new line
        fp.write("%s\n" % all_attention_atributions_output)

with open(r'/zhome/2f/8/153764/Desktop/final/attention_attention_layers22.txt', 'w') as fp:
    for item in all_attention_atributions_attention:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/zhome/2f/8/153764/Desktop/final/all_layers_attribution22.txt', 'w') as fp:
    for item in attention_layer_attrs:
        # write each item on a new line
        fp.write("%s\n" % item)

"""
lc = LayerConductance(squad_pos_forward_func, model.wav2vec2.encoder.layers[23])

layer_attributions, error = lc.attribute(inputs=inputs['input_values'],
                                baselines=ref_value['input_values'],
                                return_convergence_delta=True)
attention_layer_attrs = summarize_attributions(layer_attributions[1]).cpu().detach().tolist()
all_attention_atributions_output = layer_attributions[0].cpu().detach().tolist()
all_attention_atributions_attention = layer_attributions[1].cpu().detach().tolist()

with open(r'/zhome/2f/8/153764/Desktop/final/output_attention_layers23.txt', 'w') as fp:
    for item in all_attention_atributions_output:
        # write each item on a new line
        fp.write("%s\n" % all_attention_atributions_output)

with open(r'/zhome/2f/8/153764/Desktop/final/attention_attention_layers23.txt', 'w') as fp:
    for item in all_attention_atributions_attention:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/zhome/2f/8/153764/Desktop/final/all_layers_attribution23.txt', 'w') as fp:
    for item in attention_layer_attrs:
        # write each item on a new line
        fp.write("%s\n" % item)





print('helllo')




