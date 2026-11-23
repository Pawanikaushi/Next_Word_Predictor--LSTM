# Next_Word_Predictor--LSTM
This model predict the next word when we give it at least three words. As a beginning this model is too simple and train with small data set.

# Next Word Predictor using LSTM

A deep learning-based **Next Word Prediction system** built using **Long Short-Term Memory (LSTM)** neural networks. The model learns patterns from the Sherlock Holmes text corpus and predicts the most probable next word based on the given input sequence.

---

## Project Overview

Language models are systems that learn the structure and patterns of human language. This project implements a simple recurrent neural network-based language model using LSTM to understand word sequences and generate predictions.

The model is trained on the **Sherlock Holmes text dataset** and learns relationships between words. Given a sequence of previous words, it predicts the most likely following word.

Example:
Input:
Sherlock Holmes was

Prediction:
detective


---

## Features

- Text preprocessing using Keras Tokenizer
- Word-level language modeling
- LSTM-based sequence learning
- Next word prediction
- Saved trained model for inference
- Interactive command-line prediction

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Jupyter Notebook
- LSTM (Long Short-Term Memory Networks)

---

## Model Architecture

The neural network architecture:
Input Sequence
|
v
Embedding Layer
|
v
LSTM Layer (1000 units)
|
v
LSTM Layer (1000 units)
|
v
Dense Layer
|
v
Softmax Output Layer
|
v
Predicted Next Word


The model learns word relationships by updating weights during training using backpropagation through time (BPTT).

---

## Dataset

The model is trained using:

**Sherlock Holmes Text Corpus**

The dataset contains literary text used to train the model to learn:
- Word frequency patterns
- Sentence structures
- Context relationships

---

## Project Structure
Next_Word_Predictor--LSTM
│
├── First.ipynb              # Model training notebook
├── test.py                  # Prediction script
├── Sherlock Holmes.txt      # Training dataset
├── nextword.h5              # Trained LSTM model
├── tokenizer.pkl            # Saved tokenizer
├── requirements.txt         # Python dependencies
├── README.md
└── .gitignore
