import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/cyberbullying_clean.csv")

X = df["Comments"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_df = pd.DataFrame({
    "Comments": X_train,
    "label": y_train
})

test_df = pd.DataFrame({
    "Comments": X_test,
    "label": y_test
})

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Train/test split completed.")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
