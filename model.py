from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential


max_features = 10000
maxlen = 500

def build_model():
    model = Sequential()

    model.add(layers.Embedding(max_features, output_dim=32, input_length=maxlen))
    
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dropout(0.25))
    
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    
    
    return model

def continue_train(model, x_train, y_train, epochs=5):
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.2
    )
    
    return model, history