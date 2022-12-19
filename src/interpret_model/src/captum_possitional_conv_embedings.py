
from transformers import Wav2Vec2ForCTC
import torch
from captum.attr import LayerIntegratedGradients
from src.data.make_dataset import DatasetBuilding
import argparse
import interpret_model_utilis


def calculate_conv_embedings():
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_checkpoint = "/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/final_model/"
    model = Wav2Vec2ForCTC.from_pretrained(
        "/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danishs/save_processor/", output_attentions=True)
    model.to(device)
    model.eval()
    model.zero_grad()


    dataset = DatasetBuilding(dataset_name, dataset_dir)
    _, evaluation_data = dataset.make_dataset(model_checkpoint)

    #Interpreting Layer Outputs and Self-Attention Matrices in each Layer
    inputs, ref_value =  interpret_model_utilis.input_reference_pair(evaluation_data[20])



    lig = LayerIntegratedGradients(interpret_model_utilis.squad_pos_forward_func, model.wav2vec2.encoder.pos_conv_embed)

    attributions, delta_start = lig.attribute(inputs=inputs['input_values'],
                                            baselines=ref_value['input_values'],
                                            return_convergence_delta=True)
    embedings_conv = attributions.squeeze().cpu().detach().tolist()

    interpret_model_utilis.save_atributions_to_file(embedings_conv, '/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/interpret_model/results/convolutional_embendings.txt')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_dir", default="/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/data/", type=str)
    parser.add_argument("-dataset_name", default="data", type=str)

    
    args = parser.parse_args()

    calculate_conv_embedings()
    