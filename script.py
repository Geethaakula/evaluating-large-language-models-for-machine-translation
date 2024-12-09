"""
Original Hindi Gt 8 Words LABSE Scores 0.9 -> Trans. English 
Original English Gt 8 Words LABSE Scores 0.9 -> Trans. Hindi
"""
import pandas as pd
import csv
from datetime import datetime
import os
from dotenv import load_dotenv
import json



   

lang_dict = {
   'hi': ['Hindi','/Users/madakhil/dataset-ops/samanantar_v0.2_las/hi_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/hi_sents.tsv'],
   'mr': ['Marathi','/Users/madakhil/dataset-ops/samanantar_v0.2_las/mr_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/mr_sents.tsv'],
   'te': ['Telugu','/Users/madakhil/dataset-ops/samanantar_v0.2_las/te_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/te_sents.tsv'],
   'as': ['Assamese','/Users/madakhil/dataset-ops/samanantar_v0.2_las/as_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/as_sents.tsv'],
   'bn': ['Bangla','/Users/madakhil/dataset-ops/samanantar_v0.2_las/bn_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/bn_sents.tsv'],
   'gu': ['Gujrati','/Users/madakhil/dataset-ops/samanantar_v0.2_las/gu_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/gu_sents.tsv'],
   'kn': ['Kannada','/Users/madakhil/dataset-ops/samanantar_v0.2_las/kn_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/kn_sents.tsv'],
   'ml': ['Malayalam','/Users/madakhil/dataset-ops/samanantar_v0.2_las/ml_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/ml_sents.tsv'],
   'or': ["Oriya",'/Users/madakhil/dataset-ops/samanantar_v0.2_las/or_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/or_sents.tsv'],
   'pa': ["Punjabi",'/Users/madakhil/dataset-ops/samanantar_v0.2_las/pa_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/pa_sents.tsv'],
   'ta': ['Tamil','/Users/madakhil/dataset-ops/samanantar_v0.2_las/ta_metadata.tsv', '/Users/madakhil/dataset-ops/samanantar_v0.2_las/ta_sents.tsv']
}


MIN_LABSE_SCORE = 0.9
MIN_WORDS = 8
TEMPERATURE = 0.7
COUNT = 1100

def extract_dbl(val):
    return float(val.strip("[[").strip("]]"))
def extract_strin_gt(string):
    return len(string.split(" ")) > MIN_WORDS

def run_translations(tgt_lang, src_lang = 'en', chat_gpt_model = 'gpt-4o', temperature = 0.7, min_words = 8, min_labse_score = 0.9):
  
   print("Run stats")
   print("SRC LANG", src_lang)
   print("TGT LANG", tgt_lang)
   print("MIN_LABSE_SCORE", min_labse_score)
   print("MIN_WORDS", min_words)
   print("TEMPERATURE", temperature)

   current_timestamp = datetime.now()
   formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f").replace("-", "_").replace(":", "_").replace(".", "_")

   print("Run started at: ", formatted_timestamp)

   lang, sents_path, metadata_path = lang_dict[tgt_lang]

   sents = pd.read_csv(sents_path, delimiter="\t", header=0, quoting=csv.QUOTE_NONE)
   las_scores = pd.read_csv(metadata_path,  delimiter="\t", header=0, quoting=csv.QUOTE_NONE)
   sents_las_merge = pd.merge(sents, las_scores, on='idx',  how='inner')
   sents_las_merge['las_score_dbl'] = sents_las_merge['las'].apply(extract_dbl)
   sents_las_merge[f'src_gt_{min_words}_words'] = sents_las_merge['src'].apply(extract_strin_gt)
   sents_las_merge[f'tgt_gt_{min_words}_words'] = sents_las_merge['tgt'].apply(extract_strin_gt)
   sents_las_1000_filter_gt_0_90 = sents_las_merge[sents_las_merge[f'tgt_gt_{min_words}_words'] == True]
   sents_las_1000_filter_gt_0_90_lt_5_words = sents_las_1000_filter_gt_0_90[sents_las_1000_filter_gt_0_90['las_score_dbl'] > min_labse_score]
   dataset_output = f'./output/1000_sents/{src_lang}_{tgt_lang}_sents_las_1000_filter_lt_0_{str(min_labse_score).replace(".","_")}_gt_{min_words}_words_{formatted_timestamp}.tsv'
   print("Dataset Saved to: \n", dataset_output)
   #saving first 1000 sentences
   sents_las_1000_filter_gt_0_90_lt_5_words[:COUNT].to_csv(dataset_output, sep='\t', index=False)


for tgt, arr in lang_dict.items():
    run_translations(tgt)
    # print(tgt, arr)