"""
    This script categorizes the human_eval scores into the scale of 
    Perfect (90 - 100)
    Good (70 - 90)	
    Fair (51 - 69)	
    Poor (30 - 50)
    bad  (11 - 29)
    worst (0 - 10)
"""

from sklearn.metrics import cohen_kappa_score
import pandas as pd
import csv

# cohen_kappa_score(y1, y2)

langs = {
    'mr': ['Marathi', '/Users/madakhil/dataset-ops/final_eval/final_bleu_bert_scores_error_eval_en_to_mr_2024_10_27_20_18_49_354374.tsv'],
    'te': ['Telugu', '/Users/madakhil/dataset-ops/final_eval/final_bleu_bert_scores_error_eval_en_to_te_2024_10_27_19_00_56_603633.tsv'],
    'hi': ['Hindi', '/Users/madakhil/dataset-ops/final_eval/final_bleu_bert_scores_error_eval_en_to_hi_2024_10_28_16_20_04_647738-2.tsv']
}

def categorize(score):
    score_fl = float(score)
    if 0.90 <= score_fl <= 1.00:
        category = "Perfect"
    elif 0.70 <= score_fl < 0.90:
        category = "Good"
    elif 0.51 <= score_fl < 0.70:
        category = "Fair"
    elif 0.30 <= score_fl < 0.51:
        category = "Poor"
    elif 0.11 <= score_fl < 0.30:
        category = "Bad"
    elif 0.0 <= score_fl < 0.11:
        category = "Worst"

    return category

"""
Columns to categorize
human_eval_gpt_scores - GPT Translations evaluated by Human
human_eval_ollama_scores - Ollama Translation evaluated by Human
gpt_bleu_scores - GPT Translations evaluated by BLEU Scores
ollama_bleu_scores - OLLAMA Translations evaluated by BLEU Scores
gpt_f1_scores - GPT Translations evaluated by BERT Scores (F1 Measure)
ollama_f1_scores - OLLAMA Translations evaluated by BERT Scores (F1 Measure)
"""

for lang, arr in langs.items():
    path = arr[1]
    df = pd.read_csv(path, delimiter="\t", header=0, quoting=csv.QUOTE_NONE)[:100]
    df['human_eval_gpt_scores_catego'] = df['human_eval_gpt_scores'].apply(categorize)
    df['human_eval_ollama_scores_catego'] = df['human_eval_ollama_scores'].apply(categorize)
    df['gpt_bleu_scores_catego'] = df['gpt_bleu_scores'].apply(categorize)
    df['ollama_bleu_scores_catego'] = df['ollama_bleu_scores'].apply(categorize)
    df['gpt_f1_scores_catego'] = df['gpt_f1_scores'].apply(categorize)
    df['ollama_f1_scores_catego'] = df['ollama_f1_scores'].apply(categorize)

    print(df['human_eval_gpt_scores_catego'])
    dataset_output = f"{lang}_categorized_scores.tsv"
    print(f"Dataset saved to {dataset_output}")
    df.to_csv(dataset_output, sep='\t', index=False)


    


    

