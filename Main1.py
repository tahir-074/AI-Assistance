import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import tokenizer_from_json
import pickle
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

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

# def save_conversation():
#     conversation = conversation_log.get("1.0", tk.END)
#     username_value = username.get()
#     filename = f"resources/conversations1/_T_Training1_{username_value.upper()}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
#     with open(filename, 'w') as txt:
#         txt.write(conversation)
#     print(f"Conversation saved to {filename}")
def save_conversation():
    conversation = conversation_log.get("1.0", tk.END)
    username_value = username.get()
    filename = f"resources/conversations1/Text_{username_value.upper()}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    with open(filename, 'w') as txt:
        txt.write(conversation)

    send_email(filename, "mt580183@gmail.com", "P1fQRZOXNpT96CYE", "mt580183@gmail.com")
    #print(f"Conversation saved.")


def send_email(filename, sender_email, sender_password, receiver_email):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Conversation Log"

    with open(filename, 'r') as txt:
        conversation_text = txt.read()

    message.attach(MIMEText(conversation_text, "plain"))

    try:
        smtp_server = "smtp-relay.sendinblue.com"
        smtp_port = 587

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        
        print("Success.")
    except smtplib.SMTPException as e:
        print(f"Error occurred: {str(e)}")

# Set up the conversation log
conversation_frame = ttk.Frame(root)
conversation_frame.pack(pady=10)

conversation_log = ScrolledText(conversation_frame, height=30, width=50, cursor="")
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

send_button = ttk.Button(input_frame, text="Send", command=send_message)
send_button.grid(row=1, column=2, padx=10, pady=5, sticky="w")

root.bind('<Return>', lambda event: send_message())  # Bind the Enter key to send message

# Set up the save button
# save_button = ttk.Button(root, text="Save Conversation", command=save_conversation)
# save_button.pack(pady=10)

def on_closing():
    save_conversation()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
