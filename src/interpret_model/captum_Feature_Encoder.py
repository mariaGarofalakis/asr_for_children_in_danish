from src.data.make_dataset import DatasetBuilding
import argparse
import interpret_model_utilis
from transformers import Wav2Vec2ForCTC
import torch
from captum.attr import LayerConductance
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_layers_heatmap(scores):
  
    fig, axs = plt.subplots(3,3)
    fig.suptitle('Feature Encoder Layer Attributions')
    fig.tight_layout(pad=1.7)


    axs[0,0].title.set_text('1st Temp. Conv Layer')
    axs[0,1].title.set_text('2nd Temp. Conv Layer')
    axs[0,2].title.set_text('3nd Temp. Conv Layer')
    axs[1,0].title.set_text('4th Temp. Conv Layer')
    axs[1,1].title.set_text('5th Temp. Conv Layer')
    axs[1,2].title.set_text('6th Temp. Conv Layer')
    axs[2,1].title.set_text('7th Temp. Conv Layer')


    sns.heatmap(scores[0], linewidth=0.00005, ax = axs[0,0])
    sns.heatmap(scores[1], linewidth=0.00005, ax = axs[0,1])
    sns.heatmap(scores[2], linewidth=0.00005, ax = axs[0,2])
    sns.heatmap(scores[3], linewidth=0.00005, ax = axs[1,0])
    sns.heatmap(scores[4], linewidth=0.00005, ax = axs[1,1])
    sns.heatmap(scores[5], linewidth=0.00005, ax = axs[1,2])
    sns.heatmap(scores[6], linewidth=0.00005, ax = axs[2,1])


    plt.show()
    plt.savefig('/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/src/interpret_model/results/Feature_enco.png')



def calculate_feature_encoder():
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_checkpoint = "chcaa/xls-r-300m-danish-nst-cv9"
    model = Wav2Vec2ForCTC.from_pretrained(
        "chcaa/xls-r-300m-danish-nst-cv9", output_attentions=True)
    model.to(device)
    model.eval()
    model.zero_grad()


    dataset = DatasetBuilding(dataset_name, dataset_dir)
    _, evaluation_data = dataset.make_dataset(model_checkpoint)

    #Interpreting Layer Outputs and Self-Attention Matrices in each Layer
    inputs, ref_value =  interpret_model_utilis.input_reference_pair(evaluation_data[20])
    scores = []

    for itr,the_layer in enumerate(model.wav2vec2.feature_extractor.conv_layers):
    
        lc = LayerConductance(interpret_model_utilis.squad_pos_forward_func, the_layer)

        feature_encoder, error = lc.attribute(inputs=inputs['input_values'],
                                        baselines=ref_value['input_values'],
                                        additional_forward_args=(model, 0),
                                        return_convergence_delta=True)
        feature_encoder = feature_encoder.cpu().detach().tolist()

        scores.append(np.squeeze(np.array(feature_encoder)))

    interpret_model_utilis.save_atributions_to_file(feature_encoder,'/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/src/interpret_model/results/feature_encoder_layer'+str(itr)+'.txt')
    return scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_dir", default="/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/fine_tuning_dataset/", type=str)
    parser.add_argument("-dataset_name", default="fine_tuning_dataset", type=str)

    
    args = parser.parse_args()

    scores = calculate_feature_encoder()
    plot_layers_heatmap(scores)

    