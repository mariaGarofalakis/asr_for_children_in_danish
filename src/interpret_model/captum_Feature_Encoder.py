from src.data.make_dataset import DatasetBuilding
import argparse
import interpret_model_utilis
from transformers import Wav2Vec2ForCTC
import torch
from captum.attr import LayerConductance




def calculate_feature_encoder():
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

    for itr,the_layer in enumerate(model.wav2vec2.feature_extractor.conv_layers):

        lc = LayerConductance(interpret_model_utilis.squad_pos_forward_func, the_layer)

        feature_encoder, error = lc.attribute(inputs=inputs['input_values'],
                                        baselines=ref_value['input_values'],
                                        return_convergence_delta=True)
        feature_encoder = feature_encoder.cpu().detach().tolist()

        interpret_model_utilis.save_atributions_to_file(feature_encoder,'/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/interpret_model/results/feature_encoder'+str(itr))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_dir", default="/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/data/", type=str)
    parser.add_argument("-dataset_name", default="data", type=str)

    
    args = parser.parse_args()

    calculate_feature_encoder()
    