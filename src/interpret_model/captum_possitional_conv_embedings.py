
from transformers import AutoTokenizer, AutoConfig,Wav2Vec2ForCTC
import torch
from captum.attr import LayerIntegratedGradients
from src.data.make_dataset import DatasetBuilding
import argparse
import interpret_model_utilis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pos_conv_embendings(scores, the_tokens):
    attribitons = np.array(scores).squeeze()
    xticklabels= the_tokens

    ax = sns.heatmap(np.swapaxes(attribitons,0,1), xticklabels=xticklabels, linewidth=0.00005)
    ax.set_xticklabels(xticklabels, rotation = 360, ha="left",fontsize=8)
    plt.show()
    plt.savefig('/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/src/interpret_model/results/possitional_conv_emb.png')




def calculate_conv_embedings():
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_checkpoint = "chcaa/xls-r-300m-danish-nst-cv9"
    model = Wav2Vec2ForCTC.from_pretrained(model_checkpoint, output_attentions=True)
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
    the_tokens = interpret_model_utilis.calculate_prediction_tokens(model,evaluation_data[20],tokenizer)



    lig = LayerIntegratedGradients(interpret_model_utilis.squad_pos_forward_func, model.wav2vec2.encoder.pos_conv_embed)

    attributions, delta_start = lig.attribute(inputs=inputs['input_values'],
                                            baselines=ref_value['input_values'],
                                            additional_forward_args=(model, 0),
                                            return_convergence_delta=True)
    embedings_conv = attributions.squeeze().cpu().detach().tolist()

    interpret_model_utilis.save_atributions_to_file(embedings_conv, '/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/src/interpret_model/results/convolutional_embendings.txt')
    return embedings_conv, the_tokens



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_dir", default="/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/data/", type=str)
    parser.add_argument("-dataset_name", default="data", type=str)

    
    args = parser.parse_args()

    embedings_conv , the_tokens = calculate_conv_embedings()
    plot_pos_conv_embendings(embedings_conv , the_tokens)
    