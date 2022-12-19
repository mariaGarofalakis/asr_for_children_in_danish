import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt



def var2device(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, requires_grad=True, **kwargs)


def predict(model, inputs):

    output = model(inputs)
    return output.logits,  output.attentions


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred , _= predict(inputs)
    pred = pred[position]
    return pred.max(1).values


def input_reference_pair(input_data):
    inputs = input_data
    del inputs['input_length']
    inputs['input_values'] = var2device(torch.unsqueeze(torch.FloatTensor(inputs['input_values']), dim=0))
    inputs['labels'] = var2device(torch.FloatTensor(inputs['labels']))

    ref_value = inputs.copy()
    ref_value['input_values'] = var2device(torch.zeros(inputs['input_values'].shape).type(torch.FloatTensor))
    ref_value['labels'] = var2device(torch.zeros(inputs['labels'].shape).type(torch.FloatTensor))

    return inputs, ref_value

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def save_atributions_to_file(atributions, file_name):
    with open(file_name, 'w') as fp:
        for item in atributions:
            # write each item on a new line
            fp.write("%s\n" % item)

def visualize_token2head_scores(scores_mat,all_tokens):
    fig = plt.figure(figsize=(30, 50))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(4, 2, idx+1)
        im = ax.matshow(scores_np, cmap='viridis')
        fontdict_x = {'fontsize': 7}
        fontdict_y = {'fontsize': 5}
        ax.set_xticks(range(scores_np.shape[1]))
        ax.set_yticks(range(scores_np.shape[0]))
        ax.set_xticklabels(all_tokens, fontdict=fontdict_x, rotation=0)   
        ax.set_yticklabels(range(scores_np.shape[0]), fontdict=fontdict_y)
        ax.set_xlabel('Layer {}'.format(idx+17))
        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def visualize_token2token_scores(scores_mat,all_tokens, x_label_name='Head'):
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Attention weights')

    for idx, scores in enumerate(scores_mat):
        ax = fig.add_subplot(1, 2, idx+1)

        im = ax.imshow(scores, cmap='viridis')        
        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(all_tokens)))
        ax.set_xticklabels(all_tokens, rotation = 360, ha="left")
        ax.tick_params(axis='both', which='major', labelsize=7, pad=7)
        fontdict_y = {'fontsize': 7}  
        ax.set_yticklabels(all_tokens, rotation = 270, ha="left", fontdict=fontdict_y)
        fig.colorbar(im, fraction=0.046)
    plt.tight_layout(pad=5)
    plt.show()

def calculate_prediction_tokens(input, tokenizer):
    the_scores, output_attentions = predict(input)
    output_attentions_all = torch.stack(output_attentions).squeeze()
    predicted_ids = torch.argmax(the_scores, dim=-1).squeeze()
    all_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)
    the_tokens = [sub.replace('<pad>','') for sub in all_tokens]

    return the_tokens