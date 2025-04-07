import os
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 10000
maxlen = 500
model_path = "models/RNNModel.keras"
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

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

predictions = model.predict(padded_sequences)

for i, text in enumerate(texts):
    preview_text = ' '.join(text.split()[:15]) + '...' if len(text.split()) > 5 else text
    true_class = 'AI' if 'ai' in file_names[i].lower() else 'Human'
    prediction_class = 'AI' if predictions[i] > 0.5 else 'Human'
    
    print(f"Text {i + 1}: {preview_text}")
    print(f"True Class: {true_class}")
    print(f"Prediction: {prediction_class} (Score: {predictions[i][0]:.4f})")
    print("-" * 50)