from datasets import load_dataset
import re

target_lang="da"
chars_to_ignore_regex = '[,?.!\-\;\:"“%‘”�—’…–]'

def extract_text(batch):
  text = batch["translation"][target_lang]
  batch["text"] = re.sub(chars_to_ignore_regex, "", text.lower())
  return batch

dataset = load_dataset("europarl_bilingual", lang1="bg", lang2=target_lang, split="train")
dataset = dataset.map(extract_text, remove_columns=dataset.column_names)
with open("text.txt", "w") as file:
  file.write(" ".join(dataset["text"]))