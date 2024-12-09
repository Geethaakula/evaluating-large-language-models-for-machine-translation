import pandas as pd
import csv
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np
import json
import requests
from fireworks.client import Fireworks

from nltk.translate import meteor
from nltk import word_tokenize, download
from bert_score import score
import evaluate


download('punkt_tab')

load_dotenv()
lang_dict = {
    # 'hi': ['Hindi', 'output/en_hi_sents.tsv',],
    # 'mr': ['Marathi', 'output/en_mr_sents.tsv',],
    'te': ['Telugu', 'output/en_te_sents.tsv',]
}

OPENAI_APIKEY = os.getenv('OPENAI_API_KEY')
url = "https://api.openai.com/v1/chat/completions"
headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {OPENAI_APIKEY}'
}
bleu = evaluate.load("bleu")



def get_meteor_scores(predictions, references):

   index = 0

   meteor_scores = []

   for pred, ref in zip(predictions, references):
      index += 1
      if(ref != ref or pred != pred): # checking if na
         continue
      else:
         meteor_scores.append(meteor([word_tokenize(ref)], word_tokenize(pred)))
        
   return meteor_scores


FIREWORKS_APIKEY = os.getenv('FIREWORKS_API_KEY')
client = Fireworks(api_key=FIREWORKS_APIKEY)

# prompt = """
# Translate with respectful language for authorities (e.g., police, ministers) by using plural forms in the target language rather than singular forms to indicate respect. Now 
# """

def call_open_ai_3_5(row, tgt_lang):
    payload = json.dumps({
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": f"translate this sentence into {tgt_lang}: {row}"
        }
      ],
      "temperature": 0.7
    })
    response = requests.request("POST", url, headers=headers, data=payload)
    resp_dict = dict(response.json())
    if("choices" in resp_dict):
      if(len(resp_dict["choices"]) > 0):
        return resp_dict["choices"][0]["message"]["content"]
    else:
       return "null"

def call_ollama_api(row, tgt_lang):
  response = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3-70b-instruct",
    messages=[{
      "role": "user",
      "content": f"{prompt} just translate this sentence into {tgt_lang} and give me the translation only: {row}",
    }],
  )
  # rawData = pd.read_csv(io.StringIO(response))
  return dict(dict(dict(response)['choices'][0])['message'])['content']


def call_open_ai_4o(row, tgt_lang):
    payload = json.dumps({
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": f"translate this sentence into {tgt_lang}: {row}"
        }
      ],
      "temperature": 0.7
    })
    response = requests.request("POST", url, headers=headers, data=payload)
    resp_dict = dict(response.json())
    if("choices" in resp_dict):
      if(len(resp_dict["choices"]) > 0):
        return resp_dict["choices"][0]["message"]["content"]
    else:
       return "null"


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



#scoring

for tgt, arr in lang_dict.items():
   # do the translation for 4o
   # do the scoring
   input_path = arr[1]
   df = pd.read_csv(input_path, delimiter="\t", header=0, quoting=csv.QUOTE_NONE)[:100]

   current_timestamp = datetime.now()
   formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f").replace("-", "_").replace(":", "_").replace(".", "_")
   print("Run stats:")
   print("Started at: ", formatted_timestamp)
   print(df.columns)
   
   print("AVERAGE BERT SCORES")
   print(f"Avg. gpt_35_f1_score {np.mean(df['gpt_f1_scores'])}")
   print(f"Avg. gpt_4o_f1_score {np.mean(df['gpt_4o_f1_score'])}")
   print(f"Avg. ollama_f1_score {np.mean(df['ollama_f1_scores'])}")

   print("AVERAGE BLEU Scores")
   print(f"Avg. gpt_35_bleu_score {np.mean(df['gpt_bleu_scores'])}")
   print(f"Avg. gpt_4o_bleu_score {np.mean(df['gpt_4o_bleu_score'])}")
   print(f"Avg. ollama_bleu_score {np.mean(df['ollama_bleu_scores'])}")

   print("AVERAGE Meteor Scores")
   print(f"Avg. gpt_35_meteor_score {np.mean(df['gpt_35_meteor_score'])}")
   print(f"Avg. gpt_4o_meteor_score {np.mean(df['gpt_4o_meteor_score'])}")
   print(f"Avg. ollama_meteor_score {np.mean(df['ollama_trans_meteor_score'])}")
   
  


  #  print(df.head(10))


   #print averages
    


