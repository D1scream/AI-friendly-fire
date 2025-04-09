import os
import pickle
import kagglehub
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 10000
maxlen = 500
model_path = "models/RNNModel.keras"

tokenizer_path = 'tokenizer.pkl'
train_data_path = 'train_data.pkl'
test_data_path = 'test_data.pkl'

model = keras.models.load_model(model_path)

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

texts = []
file_names = []
folder_path = 'texts/' 
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        texts.append(read_text_from_file(file_path))
        file_names.append(file_name)

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

predictions = model.predict(padded_sequences)

RED = '\033[91m'  
GREEN = '\033[92m' 
RESET = '\033[0m'
for i, text in enumerate(texts):
    preview_text = ' '.join(text.split()[:15]) + '...' if len(text.split()) > 5 else text
    true_class = 'AI' if 'ai' in file_names[i].lower() else 'Human' 
    prediction_class = 'AI' if predictions[i] > 0.5 else 'Human'
    
    if true_class != prediction_class:
        color = RED 
    else:
        color = GREEN  
    
    print(f"{color}{file_names[i]}: {preview_text}{RESET}")
    print(f"{color}Expected: {true_class}{RESET}")
    print(f"{color}Prediction: {prediction_class} (Score: {predictions[i][0]:.4f}){RESET}")
    print("-" * 50)