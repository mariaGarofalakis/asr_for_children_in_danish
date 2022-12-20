from src.data.make_dataset import DatasetBuilding
from transformers import AutoTokenizer, AutoConfig,Wav2Vec2ForCTC
import argparse
import interpret_model_utilis
import torch
from captum.attr import LayerActivation
import numpy as np
import matplotlib.pyplot as plt

if torch.__version__ >= '1.7.0':
    norm_fn = torch.linalg.norm
else:
    norm_fn = torch.norm

def predict(inputs,model):

    del inputs['input_length']

    inputs['input_values'] = interpret_model_utilis.var2device(torch.unsqueeze(torch.FloatTensor(inputs['input_values']), dim=0))
    inputs['labels'] = torch.FloatTensor(inputs['labels'])
    output = model(**inputs)
    return output.logits,  output.attentions

def visualize_token2head_scores(scores_mat,all_tokens):
    fig = plt.figure(figsize=(30, 50))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(3, 8, idx+1)
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
    plt.savefig('/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/src/interpret_model/results//attention_token_to_head.png')

def visualize_token2token_scores(scores_mat,all_tokens, x_label_name='Head'):
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Attention weights')

    for idx, scores in enumerate(scores_mat):
        ax = fig.add_subplot(4, 4, idx+1)

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
    plt.savefig('/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/src/interpret_model/results/attention_token_to_token.png')



def attention_layers_attributions():
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_checkpoint = "chcaa/xls-r-300m-danish-nst-cv9"
    model = Wav2Vec2ForCTC.from_pretrained(
        "chcaa/xls-r-300m-danish-nst-cv9", output_attentions=True)
    model.to(device)
    model.eval()
    model.zero_grad()

    config = AutoConfig.from_pretrained(model_checkpoint)

    tokenizer_type = config.model_type if config.tokenizer_class is None else None
    config = config if config.tokenizer_class is not None else None

    tokenizer = AutoTokenizer.from_pretrained(
                model_checkpoint,
                config=config,
                tokenizer_type=tokenizer_type,
                bos_token="<s>",eos_token="</s>",unk_token="<unk>",pad_token="<pad>",word_delimiter_token="|"
            )


    dataset = DatasetBuilding(dataset_name, dataset_dir)
    _, evaluation_data = dataset.make_dataset(model_checkpoint)

    #Interpreting Layer Outputs and Self-Attention Matrices in each Layer
    inputs, ref_value =  interpret_model_utilis.input_reference_pair(evaluation_data[20])
    the_scores, output_attentions = predict(evaluation_data[20],model)

    output_attentions_all = torch.stack(output_attentions).squeeze()
    the_tokens = interpret_model_utilis.calculate_prediction_tokens(model,evaluation_data[20],tokenizer)


    num_heads = 16
    head_size = 64

    layers = [model.wav2vec2.encoder.layers[layer].attention.v_proj for layer in range(len( model.wav2vec2.encoder.layers))]


    la = LayerActivation(interpret_model_utilis.squad_pos_forward_func, layers)

    value_layer_acts = la.attribute(inputs=inputs['input_values'],additional_forward_args=(model, 0))
    value_layer_acts = torch.stack(value_layer_acts)
    new_x_shape = value_layer_acts.size()[:-1] + (num_heads, head_size)
    value_layer_acts = value_layer_acts.view(*new_x_shape)

    value_layer_acts = value_layer_acts.permute(0, 1, 3, 2, 4)
    value_layer_acts_shape = value_layer_acts.size()

    # layer x batch x seq_length x num_heads x 1 x head_size
    value_layer_acts = value_layer_acts.view(value_layer_acts_shape[:-1] + (1, value_layer_acts_shape[-1],))
    value_layer_acts = value_layer_acts.swapaxes(2,3)
    print('value_layer_acts: ', value_layer_acts.shape)

    dense_acts = torch.stack([model.wav2vec2.encoder.layers[layer].attention.out_proj.weight for layer in range(len( model.wav2vec2.encoder.layers))])

    all_head_size = 1024

    dense_acts = dense_acts.view(len(layers), all_head_size, num_heads, head_size)

    # layer x num_heads x head_size x all_head_size
    dense_acts = dense_acts.permute(0, 2, 3, 1).contiguous()

    f_x = torch.stack([value_layer_acts_i.matmul(dense_acts_i) for value_layer_acts_i, dense_acts_i in zip(value_layer_acts, dense_acts)])
    f_x.shape

    # layer x batch x seq_length x num_heads x 1 x all_head_size)
    f_x_shape = f_x.size() 
    f_x = f_x.view(f_x_shape[:-2] + (f_x_shape[-1],))
    f_x = f_x.permute(0, 1, 3, 2, 4).contiguous() 

    #(layers x batch, num_heads, seq_length, all_head_size)
    f_x_shape = f_x.size() 

    #(layers x batch, num_heads, seq_length)
    f_x_norm = norm_fn(f_x, dim=-1)




    visualize_token2head_scores(f_x_norm.squeeze().detach().cpu().numpy(),the_tokens)

    # layer x batch x num_heads x seq_length x seq_length x all_head_size
    output_attentions_all = output_attentions_all.unsqueeze(1)
    alpha_f_x = torch.einsum('lbhks,lbhsd->lbhksd', output_attentions_all.cpu(), f_x.cpu())

    # layer x batch x num_heads x seq_length x seq_length
    alpha_f_x_norm = norm_fn(alpha_f_x, dim=-1)

    layer = 23
    visualize_token2token_scores(alpha_f_x_norm[layer].squeeze().detach().cpu().numpy(),the_tokens)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_dir", default="/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/data/", type=str)
    parser.add_argument("-dataset_name", default="data", type=str) 
    args = parser.parse_args()
    attention_layers_attributions()

