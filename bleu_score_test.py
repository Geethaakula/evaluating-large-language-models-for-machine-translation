import evaluate
bleu = evaluate.load("bleu")
try:
    # bleu.computer(prediction=[..predictions..], references=[...references...])
    op = bleu.compute(predictions=["এই ধৰনৰ ফাইল খোলাৰ বাবে এটা অতিৰিক্ত প্ৰোগ্ৰাম প্ৰয়োজন।"], references=["এই ধৰণৰ ফাইল খোলাৰ বাবে এটা অতিৰিক্ত এপ্লিকেচনৰ প্ৰয়োজন:"])
    print(round(op['bleu'], 2))
except Exception as e:
    print(e)