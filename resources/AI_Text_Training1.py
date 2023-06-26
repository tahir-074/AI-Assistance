import json
import pickle
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D


def load_intents():
    with open('Text Ai Dataset.json') as file:
        data = json.load(file)
    return data

def train_model():
    data2 = load_intents()

    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    for intent in data2['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    num_classes = len(labels)
    print("num_classes:", num_classes)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 400
    max_len = 15
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])

    epochs = 200
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

    # Save the trained model
    model.save("resources1/chat_model1")

    # Save the fitted tokenizer
    with open('resources1/tokenizer1.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the fitted label encoder
    with open('resources1/label_encoder1.pickle', 'wb') as enc_file:
        pickle.dump(lbl_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    train_model()