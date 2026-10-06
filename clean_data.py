import pandas as pd
import re

# Load dataset
df = pd.read_csv("data/cyberbullying_dataset.csv")

# Clean text
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Comments"] = df["Comments"].apply(clean_text)

# Remove empty comments
df = df[df["Comments"].str.len() > 0]

# Save cleaned dataset
df.to_csv("data/cyberbullying_clean.csv", index=False)

print("Cleaning completed.")
print("Rows:", len(df))
print("\nLabels:")
print(df["label"].value_counts())
print("\nFirst 5 rows:")
print(df.head())