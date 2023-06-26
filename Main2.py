
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import json
import numpy as np
from datetime import datetime
import pickle
import speech_recognition as sr
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import tokenizer_from_json

# Initialize GUI
root = tk.Tk()
root.title("AI Assistance")

def load_intents():
    with open('resources/Text Ai Dataset.json') as file:
        data = json.load(file)
    return data

# Load intents
data1 = load_intents()

# Load the tokenizer and label encoder
with open('resources/resources1/tokenizer1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('resources/resources1/label_encoder1.pickle', 'rb') as enc_file:
    lbl_encoder = pickle.load(enc_file)

# Load the trained model
model = keras.models.load_model("resources/resources1/chat_model1")

def get_bot_response(user_message):
    user_input = user_message

    # Preprocess the user input
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', maxlen=15)

    # Get the model's prediction
    prediction = model.predict(padded_sequences)
    predicted_label = lbl_encoder.inverse_transform([np.argmax(prediction)])[0]

    # Get the response based on the predicted label
    for intent in data1['intents']:
        if intent['tag'] == predicted_label:
            responses = intent['responses']
            bot_response = np.random.choice(responses)
            break
    else:
        bot_response = "I'm sorry, I don't understand."

    return bot_response

def send_message():
    user_message = user_input.get("1.0", tk.END).strip()
    if user_message:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_log.insert(tk.END, f"{username.get()} ({current_time}):\n{user_message}\n")
        bot_response = get_bot_response(user_message)
        conversation_log.insert(tk.END, f"Bot ({current_time}):\n{bot_response}\n")
        user_input.delete("1.0", tk.END)
        conversation_log.see(tk.END)

def save_conversation():
    conversation = conversation_log.get("1.0", tk.END)
    username_value = username.get()
    filename = f"resources/conversations1/_T_Training1_{username_value.upper()}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    with open(filename, 'w') as txt:
        txt.write(conversation)
    print(f"Conversation saved to {filename}")


is_listening = False

def toggle_microphone():
    global is_listening
    if is_listening:
        stop_listening()
    else:
        start_listening()
    is_listening = not is_listening
    microphone_button.configure(text="Stop Microphone" if is_listening else "Start Microphone")

def start_listening():
    global is_listening
    is_listening = True
    r = sr.Recognizer()
    r.pause_threshold = 0.7
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        conversation_log.insert(tk.END, "Listening...\n")
        conversation_log.see(tk.END)
        audio = r.listen(source)
    try:
        user_message = r.recognize_google(audio)
        conversation_log.insert(tk.END, f"{username.get()} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n{user_message}\n")
        bot_response = get_bot_response(user_message)
        conversation_log.insert(tk.END, f"Bot ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n{bot_response}\n")
        conversation_log.see(tk.END)
    except sr.UnknownValueError:
        conversation_log.insert(tk.END, "Sorry, I could not understand.\n")
        conversation_log.see(tk.END)
    except sr.RequestError as e:
        conversation_log.insert(tk.END, f"Error: {str(e)}\n")
        conversation_log.see(tk.END)
    is_listening = False

def stop_listening():
    global is_listening
    is_listening = False

# Set up the conversation log
conversation_frame = ttk.Frame(root)
conversation_frame.pack(pady=10)

conversation_log = ScrolledText(conversation_frame, height=20, width=60)
conversation_log.pack()

# Set up the user input
input_frame = ttk.Frame(root)
input_frame.pack(pady=10)

username_label = ttk.Label(input_frame, text="Username:")
username_label.grid(row=0, column=0, padx=10, sticky="e")

username = ttk.Entry(input_frame)
username.grid(row=0, column=1, padx=10, sticky="w")

user_input = ScrolledText(input_frame, height=3, width=40)
user_input.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

microphone_button = ttk.Button(input_frame, text="Start Microphone", command=toggle_microphone)
microphone_button.grid(row=1, column=2, padx=10, pady=5, sticky="w")

root.bind('<Return>', lambda event: send_message())  # Bind the Enter key to send message

# Set up the save button
save_button = ttk.Button(root, text="Save Conversation", command=save_conversation)
save_button.pack(pady=10)

root.mainloop()
