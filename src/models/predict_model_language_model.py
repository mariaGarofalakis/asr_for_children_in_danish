from evaluate import load
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM




wer = load("wer")
script_dir='test_test_dataset'


ds = load_dataset("Alvenir/alvenir_asr_da_eval", split="test")
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))


processor = Wav2Vec2ProcessorWithLM.from_pretrained("/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/src/language_model/save_processor_lm")


model = Wav2Vec2ForCTC.from_pretrained(
      "/zhome/2f/8/153764/Desktop/the_project/ASR_for_children_in_danish/final_model/")


def calc_model_wer(model,av_wer):
  
  input_values = processor(
          ds[itr]["audio"]["array"], return_tensors="pt", padding="longest"
      ).input_values  # Batch size 1

  logits = model(input_values).logits
  transcription = processor.batch_decode(logits.detach().numpy()).text
  references = [ds[itr]['sentence']]
  wer_score = wer.compute(predictions=transcription, references=references)
  av_wer = av_wer + wer_score
  return av_wer
  

av_wer = 0
for itr in range(len(ds)):
  av_wer = calc_model_wer(model,av_wer)

print('The WER of the model evaluated on the Alvenir/alvenir_asr_da_eval with language model is:')
print(av_wer/len(ds))