# AI Chatbot
The purpose of an AI chatbot system is to provide advice and treatment options for individuals who are experiencing mental health problems.

![image]![Text](https://github.com/tahir-074/AI-Assistance/assets/76201545/532e5679-a6ed-4488-a30b-a92f669563d5)

## Requirments
To run this project, you will need the following:

    Python 3.6 or above
    TensorFlow 2.4.0 or above
    NLTK 3.6.7
    Numpy 1.18.5 or above
    Pandas 1.0.4 or above
	Keras 3.7
	Scikit Learn 0.23.0
	Tkinter 8.6. 12
 You can install these packages by running the following command:

    pip install --file requirements.txt
	
## Usage

1. Download or clone the repository to your local machine
2. Install the required packages
3. Open a terminal and navigate to the project directory.
4. Run the following command to start the chatbot:

        python Main1.py

## Directory Structure
* JSON_FILES: contains the intents.json file used for training the model
* pickle_files: contains the label_encoder1.pickle and tokenizer1.pickle files used for loading the model
* AI_Text_Training1.py: script for training the model using neural network
* Main1.py: script for running the chatbot
chat_model1: trained model file

## Build With
* PyCharm 2021.3 (Community Edition)
* Python
* TensorFlow
* Keras
* Scikit-Learn

## Disclaimer
The training code provided here serves as a basic implementation for training a neural network model on a specific dataset. It is important to note that this code may not capture all possible scenarios and variations that could arise in a production environment.

Please consider the following:

1)Dataset: The code assumes the presence of a dataset file named "Text Ai Dataset.json" in the same directory. Ensure that you have the appropriate dataset file available and modify the code accordingly if your dataset has a different name or location.

2)Hyperparameters: The hyperparameters used in this code, such as vocab_size, embedding_dim, max_len, epochs, and the learning rate (0.001), are chosen as defaults and may not be optimal for your specific dataset or problem. It is recommended to experiment and fine-tune these hyperparameters based on your data and requirements.

3)Model Architecture: The neural network architecture implemented in this code consists of specific layers, such as Embedding, GlobalAveragePooling1D, and Dense. While these layers are commonly used for text classification tasks, they may not be the most suitable for your particular problem. Consider adapting the model architecture to better suit your dataset and objectives.

4)Error Handling and Input Validation: This code does not include extensive error handling and input validation. In a real-world scenario, it is important to implement proper error handling and validate user inputs to ensure the stability and security of the system.

5)Performance and Scalability: Depending on the size and complexity of your dataset, this code may have limitations in terms of performance and scalability. Consider optimizing the code, utilizing distributed computing, or exploring alternative solutions if you anticipate working with larger datasets or require faster training times.

It is strongly recommended to review, modify, and thoroughly test the code according to your specific requirements and best practices before using it in any production or critical system. The authors and contributors of this code disclaim any liability for any damages or issues arising from the use or misuse of this code.
