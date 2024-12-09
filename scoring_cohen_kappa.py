from sklearn.metrics import cohen_kappa_score
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
    human_eval_gpt_scores = df['human_eval_gpt_scores_catego'] 
    human_eval_ollama_scores = df['human_eval_ollama_scores_catego'] 
    gpt_bleu_scores = df['gpt_bleu_scores_catego'] 
    ollama_bleu_scores = df['ollama_bleu_scores_catego'] 
    gpt_bert_scores = df['gpt_f1_scores_catego'] 
    ollama_bert_scores = df['ollama_f1_scores_catego']

    # cohen_kappa_score(y1, y2)

    print(f"For {language}")
    print("AVG. COHEN'S KAPPA SCORE FOR BERT SCORES")
    print(" - BERT SCORES for GPT vs HUMAN EVAL SCORES:" + str(cohen_kappa_score(gpt_bert_scores, human_eval_gpt_scores)))
    print(" - BERT SCORES for OLLAMA vs HUMAN EVAL SCORES: " + str(cohen_kappa_score(ollama_bert_scores,human_eval_ollama_scores)))



    print("AVG. COHEN'S KAPPA SCORE FOR BLEU SCORES")
    print(" - BLEU SCORES for GPT vs HUMAN EVAL SCORES:" + str(cohen_kappa_score(human_eval_gpt_scores, gpt_bleu_scores)))
    print(" - BLEU SCORES for OLLAMA vs HUMAN EVAL SCORES: " + str(cohen_kappa_score(human_eval_ollama_scores, ollama_bleu_scores)))

