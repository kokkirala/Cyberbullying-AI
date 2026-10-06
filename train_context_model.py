import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from sklearn.metrics import classification_report


# =============================
# Load datasets
# =============================

train_df = pd.read_csv("data/cyberbullying_context_train.csv")
val_df = pd.read_csv("data/cyberbullying_context_validation.csv")
test_df = pd.read_csv("data/cyberbullying_context_test.csv")


# =============================
# Convert 6 classes to binary
# =============================

def make_binary_label(label):
    if label == "not_cyberbullying":
        return 0
    return 1


train_df["label"] = train_df["cyberbullying_type"].apply(make_binary_label)
val_df["label"] = val_df["cyberbullying_type"].apply(make_binary_label)
test_df["label"] = test_df["cyberbullying_type"].apply(make_binary_label)


# =============================
# Prepare text
# =============================

X_train = train_df["tweet_text"].astype(str).tolist()
y_train = train_df["label"].values

X_val = val_df["tweet_text"].astype(str).tolist()
y_val = val_df["label"].values

X_test = test_df["tweet_text"].astype(str).tolist()
y_test = test_df["label"].values


# =============================
# Tokenizer
# =============================

MAX_WORDS = 20000
MAX_LEN = 150

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)


# =============================
# Convert text to sequences
# =============================

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)


X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="pre"
)

X_val_pad = pad_sequences(
    X_val_seq,
    maxlen=MAX_LEN,
    padding="pre"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="pre"
)


# =============================
# Build model
# =============================

model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128
    ),

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


model.summary()


# =============================
# Train
# =============================

history = model.fit(
    X_train_pad,
    y_train,

    validation_data=(
        X_val_pad,
        y_val
    ),

    epochs=5,

    batch_size=32
)


# =============================
# Evaluate
# =============================

loss, accuracy = model.evaluate(
    X_test_pad,
    y_test
)

print("\nTest accuracy:", accuracy)


# =============================
# Classification report
# =============================

predictions = model.predict(
    X_test_pad,
    verbose=0
)

predicted_labels = (
    predictions >= 0.5
).astype(int).flatten()


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predicted_labels,
        target_names=[
            "not_cyberbullying",
            "cyberbullying"
        ]
    )
)


# =============================
# Save model
# =============================

model.save(
    "model/cyberbullying_context_model.keras"
)


print("\nContext model saved to:")

print(
    "model/cyberbullying_context_model.keras"
)
