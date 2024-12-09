# To Run

Create a `.env` file and place in the OPENAI_APIKEY and FIREWORKS_AI_APIKEY

```
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXX
FIREWORKS_API_KEY=XXXXXXXXXXXXXXX
```

Then run each of the following file for respective task

- script.py - for filtering the datasets
- translations.py - for running translation on a dataset
- scoring-*.py - for running scores on the translations done by the LLMs
- classify_sentences.py - for converting the discrete human eval scores to continuous scores
- 