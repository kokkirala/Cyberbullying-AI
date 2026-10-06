import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


MODEL_PATH = "model/cyberbullying_context_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

train_df = pd.read_csv(
    "data/cyberbullying_context_train.csv"
)

tokenizer = Tokenizer(
    num_words=20000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    train_df["tweet_text"].astype(str)
)


comments = [
    "he trying to harass me at that moment",
    "He keeps sending me threatening messages.",
    "You are stupid and nobody likes you",
    "I really enjoyed this video, thank you!"
]


sequences = tokenizer.texts_to_sequences(comments)

padded = pad_sequences(
    sequences,
    maxlen=150,
    padding="pre"
)

predictions = model.predict(
    padded,
    verbose=0
)

for comment, prediction in zip(comments, predictions):
    score = float(prediction[0])

    print("\nComment:")
    print(comment)

    print("Cyberbullying score:", score)

    if score >= 0.80:
        print("Result: Potential cyberbullying")
    else:
        print("Result: Not detected as cyberbullying")
