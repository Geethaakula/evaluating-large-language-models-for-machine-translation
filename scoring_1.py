from bert_score import score
import pandas as pd
import csv
import evaluate
from datetime import datetime

from nltk.translate import meteor
from nltk import word_tokenize, download

download('punkt_tab')


current_timestamp = datetime.now()
formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f").replace("-", "_").replace(":", "_").replace(".", "_")
bleu = evaluate.load("bleu")

lang_dict = {
   'as': ['Assamese', '/Users/madakhil/dataset-ops/output/en_as_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_01_15_26_52_577711.tsv'],
   'bn': ['Bangla', '/Users/madakhil/dataset-ops/output/en_bn_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_01_15_33_27_184055.tsv'],
#    'gu': ['Gujrati', '/Users/madakhil/dataset-ops/output/en_gu_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_01_15_40_54_109486.tsv'],
#    'kn': ['Kannada', '/Users/madakhil/dataset-ops/output/en_kn_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_01_15_48_37_835038.tsv'],
#    'ml': ['Malayalam', '/Users/madakhil/dataset-ops/output/en_ml_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_01_15_58_51_222649.tsv'],
#    'or': ['Oriya', '/Users/madakhil/dataset-ops/output/en_or_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_01_16_08_03_889506.tsv'],
#    'pa': ['Punjabi', '/Users/madakhil/dataset-ops/output/en_pa_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_01_16_18_56_689452.tsv'],
#    'ta': ['Tamil', '/Users/madakhil/dataset-ops/output/en_ta_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_01_16_27_26_438656.tsv']
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
   print(f"Running BERT Scores for {model}")
   op_p_scores, op_r_scores, op_f1_scores = [],[],[]
   index = 0
   for refs, hypo in zip(references, predictions):
    index += 1
    try:
          p, r, f1 = score([hypo], [refs], lang=lang)
          op_p_scores.append(p)
          op_r_scores.append(r)
          op_f1_scores.append(f1)
    except Exception as e:
        
       print(f"error at {index} {hypo} -- {refs}")
       print(e)
       op_p_scores.append(0)
       op_r_scores.append(0)
       op_f1_scores.append(0)
         
   return op_f1_scores


def get_meteor_scores(predictions, references, model, lang):

   meteor_scores = []
   index = 0
   
   for hypo,refs in zip(predictions, references):
        if pd.isna(hypo) or pd.isna(refs):
            meteor_scores.append(0.0)  # Assign 0.0 for NaN entries
            continue
        print("hypo",[word_tokenize(refs)])
        print("ref",word_tokenize(hypo))
        
        print(meteor([word_tokenize(refs)], word_tokenize(hypo)))
        print("======")
        index += 1
        
   
#    print(meteor_scores)
   return meteor_scores


for tgt, arr in lang_dict.items():
    print("Run stats")
    print("TGT LANG", tgt)
    lang = tgt
    language, sents_path = arr[0], arr[1] # path of the dataset containing translations
    df = pd.read_csv(sents_path, delimiter="\t", header=0, quoting=csv.QUOTE_NONE)
    

    print([i for i in df['gpt_3_5_trans'].to_list()])
    
    
    
    

    dataset_output = f'./output/en_{lang}_sents_final_scores_{formatted_timestamp}.tsv'
    # df.to_csv(dataset_output, sep='\t', index=False)


