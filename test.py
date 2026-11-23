from pathlib import Path

from tensorflow.keras.models import load_model
import numpy as np
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "nextword.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"


if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not TOKENIZER_PATH.is_file():
    raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")

model = load_model(MODEL_PATH, compile=False)

with TOKENIZER_PATH.open("rb") as f:
    tokenizer = pickle.load(f)


def predict_next_word(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    # Pad short phrases to the 3-word input length expected by the model.
    sequence = sequence[-3:]
    sequence = ([0] * (3 - len(sequence))) + sequence

    predictions = model.predict(np.array([sequence]), verbose=0)
    predicted_index = int(np.argmax(predictions[0]))

    predicted_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break

    print("Predicted word:", predicted_word)
    return predicted_word


if __name__ == "__main__":
    while True:
        text = input("Enter a sentence: ")

        if text.lower() == "exit":
            break

        predict_next_word(model, tokenizer, text)