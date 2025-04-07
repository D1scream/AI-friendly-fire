import kagglehub
import pandas as pd
import random

dataset_path = kagglehub.dataset_download("shanegerami/ai-vs-human-text") + "/AI_Human.csv"

data = pd.read_csv(dataset_path)

ai_texts = data[data['generated'] == 1]['text']
human_texts = data[data['generated'] == 0]['text']

ai_samples = ai_texts.sample(n=10, random_state=32)
human_samples = human_texts.sample(n=10, random_state=32)

for idx, text in enumerate(ai_samples, 1):
    file_name = f"texts/ai{idx}.txt"
    with open(file_name, 'w', encoding='utf-8') as ai_file:
        ai_file.write(text)

for idx, text in enumerate(human_samples, 1):
    file_name = f"texts/human{idx}.txt"
    with open(file_name, 'w', encoding='utf-8') as human_file:
        human_file.write(text)
