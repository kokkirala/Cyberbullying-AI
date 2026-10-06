import pandas as pd
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from sklearn.metrics import classification_report

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df["Comments"].astype(str).tolist()
y_train = train_df["label"].values

X_test = test_df["Comments"].astype(str).tolist()
y_test = test_df["label"].values

MAX_WORDS = 20000
MAX_LEN = 150

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

print("Vocabulary size:", len(tokenizer.word_index))

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="pre"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="pre"
)

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    epochs=5,
    batch_size=32
)

loss, accuracy = model.evaluate(
    X_test_pad,
    y_test
)

print("\nTest accuracy:", accuracy)

predictions = model.predict(
    X_test_pad
)

predicted_labels = (
    predictions >= 0.5
).astype(int).flatten()

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predicted_labels
    )
)

model.save(
    "model/cyberbullying_model.keras"
)

with open(
    "model/tokenizer.pkl",
    "wb"
) as file:
    pickle.dump(
        tokenizer,
        file
    )

print("\nModel and tokenizer saved successfully.")
