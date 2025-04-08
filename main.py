
import os
import kagglehub
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
import pickle

from model import build_model

dataset_path = kagglehub.dataset_download("shanegerami/ai-vs-human-text")+"/AI_Human.csv"

max_features = 10000
maxlen = 500
model_path = "models/RNNModel.keras"

tokenizer_path = 'tokenizer.pkl'
train_data_path = 'train_data.pkl'
test_data_path = 'test_data.pkl'

data = pd.read_csv(dataset_path)

x_train, x_test, y_train, y_test = train_test_split(data['text'], data['generated'], test_size=0.2, random_state=32)
print("Train-test divided")

if os.path.exists(tokenizer_path):
    print("Loading tokenizer from file")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
else:
    print("Training tokenizer")
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved")

if os.path.exists(train_data_path) and os.path.exists(test_data_path):
    print("Loading preprocessed data from file")
    with open(train_data_path, 'rb') as f:
        x_train_pad, y_train = pickle.load(f)
    with open(test_data_path, 'rb') as f:
        x_test_pad, y_test = pickle.load(f)
else:
    print("Vectorizing and padding data")
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    x_train_pad = pad_sequences(x_train_seq, maxlen=100)
    x_test_pad = pad_sequences(x_test_seq, maxlen=100)

    with open(train_data_path, 'wb') as f:
        pickle.dump((x_train_pad, y_train), f)
    with open(test_data_path, 'wb') as f:
        pickle.dump((x_test_pad, y_test), f)
    print("Preprocessed data saved")

print("Data prepared.")

callbacks = [
    ModelCheckpoint(
        model_path,  
        monitor='val_loss', 
        save_best_only=True
    ), 
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )]
model = build_model()
history = model.fit(x_train_pad, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks
    )

model = keras.models.load_model(model_path)
model.summary()
 
test_loss, test_acc = model.evaluate(x_test_pad, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")