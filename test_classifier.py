import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "model/cyberbullying_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

comment = input("Enter a comment: ")

sequence = tokenizer.texts_to_sequences([comment])
padded = pad_sequences(sequence, maxlen=150, padding="pre")

prediction = model.predict(padded, verbose=0)[0][0]

print("\nPrediction score:", prediction)

if prediction >= 0.5:
    print("Result: Potentially harmful/toxic comment")
else:
    print("Result: Non-toxic comment")

