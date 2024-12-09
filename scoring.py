from bert_score import score
import pandas as pd
import csv
import evaluate
from datetime import datetime

from nltk.translate import meteor
from nltk import word_tokenize, download
import numpy as np

download('punkt_tab')
download('wordnet')


current_timestamp = datetime.now()
formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f").replace("-", "_").replace(":", "_").replace(".", "_")
bleu = evaluate.load("bleu")

# lang_dict = {
   #  "mr": ["Marathi", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_mr_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_07_12_46_36_970306.tsv"],
   #  "hi": ["Hindi", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_hi_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_07_11_41_03_172106.tsv"],
   #  "te": ["Telugu", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_te_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_10_23_03_13_293240.tsv"],
#    #  'as': ['Assamese',"/Users/madakhil/dataset-ops/output/1000_sents/translations/en_as_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_13_19_18_27_442868.tsv"],
#    #  'bn': ['Bangla', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_bn_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_13_19_48_05_069586.tsv"],
#    #  'gu': ['Gujrati', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_gu_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_13_20_22_01_151006.tsv"],
#    #  'kn': ['Kannada', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_kn_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_13_20_54_03_660808.tsv"],
#    #  'ml': ['Malayalam', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_ml_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_13_21_39_03_106961.tsv"],
#    #  'or': ["Oriya", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_or_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_13_22_22_55_696625.tsv"],
#    #  'pa': ["Punjabi", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_pa_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_13_23_09_06_109739.tsv"],
#    #  'ta': ['Tamil', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_ta_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_19_16_45_05_489066.tsv"],
# }


lang_dict = {
   #  "mr": ["Marathi", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_mr_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_19_22_25_15_343193.tsv"],
   #  "hi": ["Hindi", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_hi_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_19_19_31_20_463608.tsv"],
   #  "te": ["Telugu", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_te_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_19_22_44_47_518830.tsv"],
   #  'as': ['Assamese',"/Users/madakhil/dataset-ops/output/1000_sents/translations/en_as_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_19_23_27_54_352790.tsv"],
   #  'bn': ['Bangla', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_bn_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_19_23_57_34_236385.tsv"],
   #  'gu': ['Gujrati', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_gu_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_20_00_31_11_257639.tsv"],
   #  'kn': ['Kannada', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_kn_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_20_11_18_24_797445.tsv"],
   #  'ml': ['Malayalam', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_ml_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_20_12_28_11_334240.tsv"],
    'or': ["Oriya", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_or_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_20_13_44_53_648968.tsv"],
    'pa': ["Punjabi", "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_pa_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_20_15_08_21_171298.tsv"],
    'ta': ['Tamil', "/Users/madakhil/dataset-ops/output/1000_sents/translations/en_ta_with_prompt_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_20_16_01_13_985385.tsv"],
}

def get_bleu_scores(predictions, references, model, lang):
   """
   Takes the original sentence and the translations then gives the precision, recall and f1 scores for each sentences

   """
   print(f"Running BLEU Scores for {lang} {model}")
   op_bleu_scores = []
   index = 0
   for pred, ref in zip(predictions, references):
      index += 1
      try:
         op = bleu.compute(predictions=[pred], references=[ref])
         op_bleu_scores.append(round(op['bleu'], 2))
      except Exception as e:
         print(f"error at {index} {pred} -- {ref}")
         print(e)
         op_bleu_scores.append(0) #appending error score 0

   
   return op_bleu_scores


def get_bert_scores(predictions, references, model, lang):
   """
   Takes the original sentences and the translations then gives the precision, recall and f1 scores for each sentences

   """
   op_f1_scores = []
   index = 0
   for refs, hypo in zip(references.tolist(), predictions.tolist()):
    index += 1
    try:
          p, r, f1 = score([hypo], [refs], lang=lang)
          op_f1_scores.append(f1)
    except Exception as e:
        
       print(f"error at {index} {hypo} -- {refs}")
       print(e)
       op_f1_scores.append(0)
         
   return op_f1_scores


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
    language, sents_path = arr[0], arr[1] # path of the dataset containing translations
    df = pd.read_csv(sents_path, delimiter="\t", header=0, quoting=csv.QUOTE_NONE)[:1000]

   #  print(f"Running BLEU Scores for GPT 3.5")
   #  df['gpt_35_bleu_score'] = get_bleu_scores(df['gpt_3_5_trans'], df['tgt'], 'GPT 3.5', lang)
   #  print(f"Running BLEU Scores for GPT 4o")
   #  df['gpt_4o_bleu_score'] = get_bleu_scores(df['gpt_4o_trans'], df['tgt'], 'GPT 4o', lang)
    print(f"Running BLEU Scores for Ollama")
    df['ollama_bleu_score'] = get_bleu_scores(df['ollama_trans'], df['tgt'], 'OLLAMA', lang)

   #  print(f"Avg. gpt_35_f1_score {np.mean(df['gpt_35_bleu_score'])}")
   #  print(f"Avg. gpt_4o_f1_score {np.mean(df['gpt_4o_bleu_score'])}")
    print(f"Avg. ollama_bleu_score {np.mean(df['ollama_bleu_score'])}")

   
   #  print(f"Running BERT Scores for GPT 3.5")
   #  df['gpt_35_f1_score'] = get_bert_scores(df['gpt_3_5_trans'], df['tgt'], 'GPT 3.5', lang)
   #  print(f"Running BERT Scores for GPT 4o")
   #  df['gpt_4o_f1_score'] = get_bert_scores(df['gpt_4o_trans'], df['tgt'], 'GPT 4o', lang)
    print(f"Running BERT Scores for Ollama")
    df['ollama_f1_score'] = get_bert_scores(df['ollama_trans'], df['tgt'], 'OLLAMA', lang)
    print(f"Avg. ollama_f1_score {np.mean(df['ollama_f1_score'])}")

   #  print(f"Running METEOR Scores for GPT35")
   #  df['gpt_35_meteor_score'] = get_meteor_scores(df['gpt_3_5_trans'].to_list(), df['tgt'].to_list(), 'GPT 3.5', lang)
   #  print(f"Running METEOR Scores for GPT4o")
   #  df['gpt_4o_meteor_score'] = get_meteor_scores(df['gpt_4o_trans'].to_list(), df['tgt'].to_list(), 'GPT 4o', lang)
    print(f"Running METEOR Scores for Ollama")
    df['ollama_meteor_score'] = get_meteor_scores(df['ollama_trans'].to_list(), df['tgt'].to_list(), 'OLLAMA', lang)
    print(f"Avg. ollama_meteor_score {np.mean(df['ollama_meteor_score'])}")

    print(df.columns)



    dataset_output = f'./output/1000_sents/scores/en_{lang}_100_sents_scores_{formatted_timestamp}.tsv'
    df.to_csv(dataset_output, sep='\t', index=False)


