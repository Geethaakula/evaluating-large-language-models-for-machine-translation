import pandas as pd
import csv
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import requests
from fireworks.client import Fireworks

load_dotenv()
lang_dict = {
    'hi': ['Hindi'],
    'mr': ['Marathi'],
    'te': ['Telugu']
}

OPENAI_APIKEY = os.getenv('OPENAI_API_KEY')
url = "https://api.openai.com/v1/chat/completions"
headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {OPENAI_APIKEY}'
}

prompt = """
Translate with respectful language for authorities (e.g., police, ministers) by using plural forms in the target language rather than singular forms to indicate respect. Now 
"""

def call_open_ai_3_5(row, tgt_lang):
    payload = json.dumps({
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": f"{prompt} translate this sentence into {tgt_lang}: {row}"
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

def call_open_ai_4o(row, tgt_lang):
    payload = json.dumps({
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": f"{prompt} translate this sentence into {tgt_lang}: {row}"
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


FIREWORKS_APIKEY = os.getenv('FIREWORKS_API_KEY')
client = Fireworks(api_key=FIREWORKS_APIKEY)
def llm_translation(tgt_lang, src_lang = 'en', temperature = 0.7, min_words = 8, min_labse_score = 0.9):
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f").replace("-", "_").replace(":", "_").replace(".", "_")
    print("Run stats:")
    print("Started at: ", formatted_timestamp)
    print("SRC LANG", src_lang)
    print("TGT LANG", tgt_lang)
    print("MIN_LABSE_SCORE", min_labse_score)
    print("MIN_WORDS", min_words)
    print("TEMPERATURE", temperature)
    lang = arr[0]
    input_path = "/Users/madakhil/dataset-ops/new_sentences.tsv"
    print('Reading csv')
    input_sents = pd.read_csv(input_path, delimiter="\t", header=0, quoting=csv.QUOTE_NONE)[:1100]
    print(input_sents['src'])
    print("""STARTING GPT 3.5 TRANS""")
    input_sents['gpt_3_5_trans'] = input_sents['src'].apply(call_open_ai_3_5, args=(lang,))
    print("""END GPT 3.5 TRANS""")
    print("""STARTING GPT 4o TRANS""")
    input_sents['gpt_4o_trans'] = input_sents['src'].apply(call_open_ai_4o, args=(lang,))
    print("""END GPT 4o TRANS""")
    print("""STARTING OLLAMA TRANS""")
    input_sents['ollama_trans'] = input_sents['src'].apply(call_ollama_api, args=(lang,))
    print("""END OLLAMA TRANS""")
    dataset_output = f'./output/new_sentences_with_prompt_{str(min_labse_score).replace(".","_")}_en_to_{tgt_lang}_{formatted_timestamp}.tsv'
    input_sents.to_csv(dataset_output, sep='\t', index=False)
    print(f"Dataset for {tgt} {lang} saved to {dataset_output}")


for tgt, arr in lang_dict.items():
   llm_translation(tgt_lang=tgt)