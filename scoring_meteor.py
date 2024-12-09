from nltk.translate import meteor

from nltk.translate import meteor
from nltk import word_tokenize, download
import csv
import pandas as pd
import math


download('punkt_tab')
download('wordnet')

df = pd.read_csv("/Users/madakhil/dataset-ops/output/en_as_sents_las_500_filter_lt_0_0_9_gt_8_words_2024_11_01_15_26_52_577711.tsv", delimiter="\t", header=0, quoting=csv.QUOTE_NONE)

print(df.columns)
# print(df['tgt'])
# print(df['gpt_3_5_trans'])
# print(df['gpt_4o_trans'])
# print(df['ollama_trans'])

index = 0

for pred, ref in zip(df['gpt_3_5_trans'], df['tgt']):
    index += 1
    if(ref != ref or pred != pred):
        continue
    else:
        print(meteor([word_tokenize(ref)], word_tokenize(pred)))