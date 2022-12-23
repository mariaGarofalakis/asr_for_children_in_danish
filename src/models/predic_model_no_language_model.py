from evaluate import load
import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import evaluate



wer = evaluate.load("wer")
script_dir='test_test_dataset'
ds = load_dataset("Alvenir/alvenir_asr_da_eval", split="test")
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

processor = Wav2Vec2Processor.from_pretrained(
    "/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/save_processor/")

model = Wav2Vec2ForCTC.from_pretrained(
      "/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/final_model/")



def calc_model_wer(model,av_wer):
    input_values = processor(
        ds[itr]["audio"]["array"], return_tensors="pt", padding="longest"
    ).input_values
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    references = [ds[itr]['sentence']]
    wer_score = wer.compute(predictions=transcription, references=references)
    av_wer = av_wer + wer_score
    
    return av_wer
  

av_wer = []



for itr in range(len(ds)):
  av_wer_1 = calc_model_wer(model,av_wer)




print('The WER of the model evaluated on the Alvenir/alvenir_asr_da_eval with no language model is:')
print(av_wer_1/len(ds))

