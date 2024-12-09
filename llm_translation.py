import pandas as pd
import csv
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
import json
import csv
from fireworks.client import Fireworks


load_dotenv()


lang_dict = {
  'hi': ['Hindi', "/Users/madakhil/dataset-ops/output/1000_sents/en_hi_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_53_52_467516.tsv"],
  'mr': ['Marathi', "/Users/madakhil/dataset-ops/output/1000_sents/en_mr_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_55_21_623136.tsv"],
  'te': ['Telugu', "/Users/madakhil/dataset-ops/output/1000_sents/en_te_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_55_44_521083.tsv"],
  'as': ['Assamese',"/Users/madakhil/dataset-ops/output/1000_sents/en_as_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_56_14_845552.tsv"],
  'bn': ['Bangla', "/Users/madakhil/dataset-ops/output/1000_sents/en_bn_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_56_15_755531.tsv"],
  'gu': ['Gujrati', "/Users/madakhil/dataset-ops/output/1000_sents/en_gu_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_57_19_829589.tsv"],
  'kn': ['Kannada', "/Users/madakhil/dataset-ops/output/1000_sents/en_kn_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_57_39_466304.tsv"],
  'ml': ['Malayalam', "/Users/madakhil/dataset-ops/output/1000_sents/en_ml_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_58_04_748317.tsv"],
  'or': ["Oriya", "/Users/madakhil/dataset-ops/output/1000_sents/en_or_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_58_45_453461.tsv"],
  'pa': ["Punjabi", "/Users/madakhil/dataset-ops/output/1000_sents/en_pa_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_58_52_613551.tsv"],
  'ta': ['Tamil', "/Users/madakhil/dataset-ops/output/1000_sents/en_ta_sents_las_1000_filter_lt_0_0_9_gt_8_words_2024_11_06_20_59_11_203924.tsv"],
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

COUNT = 1000
index1 = 0
index2 = 0
index3 = 0

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

    print(f"{row}")
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
          "content": f"translate this sentence into {tgt_lang}: {row}"
        }
      ],
      "temperature": 0.7
    })

    print(f"{row}")
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

  # print(f"{row}")
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
    lang= arr[0]
    input_path = arr[1]
    print(f"{input_path}")

    print('Reading csv')
    input_sents = pd.read_csv(input_path, delimiter="\t", header=0, quoting=csv.QUOTE_NONE)[:COUNT]
    print(input_sents['tgt'])

    # print("""STARTING GPT 3.5 TRANS""")
    # input_sents['gpt_3_5_trans'] = input_sents['src'].apply(call_open_ai_3_5, args=(lang,))
    # print("""END GPT 3.5 TRANS""")

    # print("""STARTING GPT 4o TRANS""")
    # input_sents['gpt_4o_trans'] = input_sents['src'].apply(call_open_ai_4o, args=(lang,))
    # print("""END GPT 4o TRANS""")

    print("""STARTING OLLAMA TRANS""")
    input_sents['ollama_trans'] = input_sents['src'].apply(call_ollama_api, args=(lang,))
    print("""END OLLAMA TRANS""")

    dataset_output = f'./output/1000_sents/translations/{src_lang}_{tgt_lang}_with_prompt_sents_las_500_filter_lt_0_{str(min_labse_score).replace(".","_")}_gt_{min_words}_words_{formatted_timestamp}.tsv'

    input_sents.to_csv(dataset_output, sep='\t', index=False)

    print(f"Dataset for {tgt} {lang} saved to {dataset_output}")



    
for tgt, arr in lang_dict.items():
   llm_translation(tgt_lang=tgt)