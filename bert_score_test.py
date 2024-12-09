import pandas as pd
from bert_score import score


try:
          # score ([prediction],[reference])
          p= score(["এই ধৰনৰ ফাইল খোলাৰ বাবে এটা অতিৰিক্ত প্ৰোগ্ৰাম প্ৰয়োজন।"], ["এই ধৰণৰ ফাইল খোলাৰ বাবে এটা অতিৰিক্ত এপ্লিকেচনৰ প্ৰয়োজন:"], lang='as')
          print("f1 scores")
          print(p)
except Exception as e:
          print(e)