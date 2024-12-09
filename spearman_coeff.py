from scipy.stats import spearmanr
import pandas as pd
import csv

langs = {
    'mr': ['Marathi', 'final_eval/mr_categorized_scores.tsv'],
    'te': ['Telugu', 'final_eval/te_categorized_scores.tsv'],
    'hi': ['Hindi', 'final_eval/hi_categorized_scores.tsv']
}


for lang, arr in langs.items():
    path = arr[1]
    language = arr[0]
    df = pd.read_csv(path, delimiter="\t", header=0, quoting=csv.QUOTE_NONE)[:100]
    human_eval_gpt_scores = df['human_eval_gpt_scores'] 
    human_eval_ollama_scores = df['human_eval_ollama_scores'] 
    gpt_bleu_scores = df['gpt_bleu_scores'] 
    ollama_bleu_scores = df['ollama_bleu_scores'] 
    gpt_bert_scores = df['gpt_f1_scores'] 
    ollama_bert_scores = df['ollama_f1_scores']

    # spearmanr(y1, y2)

    print(f"For {language}")
    print("AVG. SPEARMAN CORRELATION COEFF FOR BERT SCORES")
    print(" - BERT SCORES for GPT vs HUMAN EVAL SCORES:" + str(spearmanr(gpt_bert_scores, human_eval_gpt_scores).statistic))
    print(" - BERT SCORES for OLLAMA vs HUMAN EVAL SCORES: " + str(spearmanr(ollama_bert_scores,human_eval_ollama_scores).statistic))



    print("AVG. SPEARMAN CORRELATION COEFF FOR BLEU SCORES")
    print(" - BLEU SCORES for GPT vs HUMAN EVAL SCORES:" + str(spearmanr(human_eval_gpt_scores, gpt_bleu_scores).statistic))
    print(" - BLEU SCORES for OLLAMA vs HUMAN EVAL SCORES: " + str(spearmanr(human_eval_ollama_scores, ollama_bleu_scores).statistic))

