import pandas as pd
import csv
import numpy as np


lang_dict = {
    "as" : ["Assamese","en_as_sents_bert_scores_2024_11_01_22_11_31_908317.tsv","en_as_sents_bleu_scores_2024_11_01_20_50_09_363831.tsv","en_as_sents_meteor_scores_2024_11_04_15_08_26_230259.tsv",],
    "bn" : ["Bengali","en_bn_sents_bert_scores_2024_11_01_22_11_31_908317.tsv","en_bn_sents_bleu_scores_2024_11_01_20_50_09_363831.tsv","en_bn_sents_meteor_scores_2024_11_04_15_08_26_230259.tsv",],
    "gu" : ["Gujrati","en_gu_sents_bert_scores_2024_11_01_22_11_31_908317.tsv","en_gu_sents_bleu_scores_2024_11_01_20_50_09_363831.tsv","en_gu_sents_meteor_scores_2024_11_04_15_08_26_230259.tsv",],
    "kn" : ["Kannada","en_kn_sents_bert_scores_2024_11_01_22_11_31_908317.tsv","en_kn_sents_bleu_scores_2024_11_01_20_50_09_363831.tsv","en_kn_sents_meteor_scores_2024_11_04_15_08_26_230259.tsv",],
    "ml" : ["Malayalam","en_ml_sents_bert_scores_2024_11_01_22_11_31_908317.tsv","en_ml_sents_bleu_scores_2024_11_01_20_50_09_363831.tsv","en_ml_sents_meteor_scores_2024_11_04_15_08_26_230259.tsv",],
    "or" : ["Oriya","en_or_sents_bert_scores_2024_11_01_22_11_31_908317.tsv","en_or_sents_bleu_scores_2024_11_01_20_50_09_363831.tsv","en_or_sents_meteor_scores_2024_11_04_15_08_26_230259.tsv",],
    "pa" : ["Punjabi","en_pa_sents_bert_scores_2024_11_01_22_11_31_908317.tsv","en_pa_sents_bleu_scores_2024_11_01_20_50_09_363831.tsv","en_pa_sents_meteor_scores_2024_11_04_15_08_26_230259.tsv",],
    "ta" : ["Tamil","en_ta_sents_bert_scores_2024_11_01_22_11_31_908317.tsv","en_ta_sents_bleu_scores_2024_11_01_20_50_09_363831.tsv","en_ta_sents_meteor_scores_2024_11_04_15_08_26_230259.tsv",],
}

for lang, arr in lang_dict.items():
    print("====Run stats====")
    language, bert_path, bleu_path, meteor_path = arr
    print(f"Running for language: {language}")

    print("Average BERT Scores")
    df_bert = pd.read_csv(f"/Users/madakhil/dataset-ops/output/{bert_path}", delimiter="\t", header=0, quoting=csv.QUOTE_NONE)
    print(df_bert.columns)
    print(f"Avg. gpt_35_f1_score {np.mean(df_bert['gpt_35_f1_score'])}")
    print(f"Avg. gpt_4o_f1_score {np.mean(df_bert['gpt_4o_f1_score'])}")
    print(f"Avg. ollama_f1_score {np.mean(df_bert['ollama_f1_score'])}")


    print("Average BLEU Scores")
    df_bleu = pd.read_csv(f"/Users/madakhil/dataset-ops/output/{bleu_path}", delimiter="\t", header=0, quoting=csv.QUOTE_NONE)
    print(df_bleu.columns)

    print(f"Avg. gpt_35_bleu_score {np.mean(df_bleu['gpt_35_bleu_score'])}")
    print(f"Avg. gpt_4o_bleu_score {np.mean(df_bleu['gpt_4o_bleu_score'])}")
    print(f"Avg. ollama_bleu_score {np.mean(df_bleu['ollama_bleu_score'])}")

    print("Average METEOR Scores")
    df_meteor = pd.read_csv(f"/Users/madakhil/dataset-ops/output/{meteor_path}", delimiter="\t", header=0, quoting=csv.QUOTE_NONE)
    print(df_meteor.columns)

    print(f"Avg. gpt_35_meteor_score {np.mean(df_meteor['gpt_35_meteor_score'])}")
    print(f"Avg. gpt_4o_meteor_score {np.mean(df_meteor['gpt_4o_meteor_score'])}")
    print(f"Avg. ollama_meteor_score {np.mean(df_meteor['ollama_meteor_score'])}")

    print("====End====")