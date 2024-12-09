from bert_score import score
import pandas as pd
import csv
import evaluate
from datetime import datetime

from nltk.translate import meteor
from nltk import word_tokenize, download

download('punkt_tab')

lang_dict = {
    "hi" : ["output/new_sentences_0_9_en_to_hi_2024_11_04_22_39_03_130182.tsv", "Chatgpt-Translated Hindi (en_hi_gpt_3_5_trans_las_gt_las_gt_0_9_gt_8_words)", "Ollama-Translated Hindi(en_hi_output_ollama_v3_trans_las_gt_las_gt_0_9_gt_8_words)"],
    "mr" : ["output/new_sentences_0_9_en_to_mr_2024_11_04_22_44_48_679230.tsv"],
    "te" : ["output/new_sentences_0_9_en_to_mr_2024_11_04_22_44_48_679230.tsv"]
}

def get_meteor_scores(predictions, references, model, lang):

   index = 0

   meteor_scores = []

   for pred, ref in zip(predictions, references):
      index += 1
      if(ref != ref or pred != pred): # checking if na
         continue
      else:
         meteor_scores.append(meteor([word_tokenize(ref)], word_tokenize(pred)))
        
   return meteor_scores

for tgt, arr in lang_dict.items():
    print("Run stats")
    print("TGT LANG", tgt)
    lang = tgt
    sents_path = arr[0] # path of the dataset containing translations
    df = pd.read_csv(sents_path, delimiter="\t", header=0, quoting=csv.QUOTE_NONE)
    
    print(df.columns)

   #  df['gpt_35_meteor_score'] = get_meteor_scores(df['gpt_3_5_trans'].to_list(), df['tgt'].to_list(), 'GPT 3.5', lang)
   #  df['gpt_4o_meteor_score'] = get_meteor_scores(df['gpt_4o_trans'].to_list(), df['tgt'].to_list(), 'GPT 3.5', lang)
   #  df['ollama_trans_meteor_score'] = get_meteor_scores(df['ollama_trans'].to_list(), df['tgt'].to_list(), 'GPT 3.5', lang)
    
    
    

   #  dataset_output = f'final_eval/{lang}_categorized_scores_w_meteor.tsv'
   #  df.to_csv(dataset_output, sep='\t', index=False)



