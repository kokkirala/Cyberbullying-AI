import pickle
import chromadb
import tensorflow as tf

from chromadb.utils import embedding_functions
from langchain_ollama import ChatOllama
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================
# Load Model 1: Toxicity model
# ============================================

toxicity_model = tf.keras.models.load_model(
    "model/cyberbullying_model.keras"
)

with open("model/tokenizer.pkl", "rb") as file:
    toxicity_tokenizer = pickle.load(file)


# ============================================
# Load Model 2: Cyberbullying context model
# ============================================

context_model = tf.keras.models.load_model(
    "model/cyberbullying_context_model.keras"
)

context_tokenizer = Tokenizer(
    num_words=20000,
    oov_token="<OOV>"
)

import pandas as pd

context_train_df = pd.read_csv(
    "data/cyberbullying_context_train.csv"
)

context_tokenizer.fit_on_texts(
    context_train_df["tweet_text"].astype(str)
)


# ============================================
# Connect to ChromaDB
# ============================================

client = chromadb.PersistentClient(
    path="rag/chroma_db"
)

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_collection(
    name="cyberbullying_knowledge",
    embedding_function=embedding_function
)


# ============================================
# Connect to Qwen3 4B
# ============================================

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.2
)


# ============================================
# Model 1: Toxicity detection
# ============================================

def classify_toxicity(text):

    sequence = toxicity_tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=150,
        padding="pre"
    )

    prediction = toxicity_model.predict(
        padded,
        verbose=0
    )[0][0]

    return float(prediction)


# ============================================
# Model 2: Cyberbullying context detection
# ============================================

def classify_context(text):

    sequence = context_tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=150,
        padding="pre"
    )

    prediction = context_model.predict(
        padded,
        verbose=0
    )[0][0]

    return float(prediction)


# ============================================
# RAG retrieval
# ============================================

def retrieve_knowledge(text):

    results = collection.query(
        query_texts=[text],
        n_results=3
    )

    return "\n\n".join(
        results["documents"][0]
    )

# ============================================
# Simple contextual signal detection
# ============================================

def detect_context_signal(text):

    text = text.lower()

    harassment_words = [
        "harass",
        "harassing",
        "insulting",
        "bullying",
        "threatening",
        "threats",
        "make fun of me",
        "targeting me"
    ]

    repetition_words = [
        "every day",
        "everyday",
        "keeps",
        "repeatedly",
        "again and again",
        "constantly"
    ]

    has_harassment = any(
        word in text
        for word in harassment_words
    )

    has_repetition = any(
        word in text
        for word in repetition_words
    )

    return has_harassment and has_repetition

# ============================================
# AI Agent
# ============================================

def agent(text):

    print("\n[Agent] Analyzing input...")

    toxicity_score = classify_toxicity(text)

    context_score = classify_context(text)

    print(
        "[Agent] Toxicity score:",
        toxicity_score
    )

    print(
        "[Agent] Cyberbullying context score:",
        context_score
    )


    # ========================================
    # Combined decision
    # ========================================

    toxicity_detected = toxicity_score >= 0.5
    context_signal = detect_context_signal(text)
    print(f"[Agent] Context rule signal: {context_signal}")
    context_detected = context_score >= 0.80 or context_signal


    if not toxicity_detected and not context_detected:

        return {
            "toxicity_score": toxicity_score,
            "context_score": context_score,
            "result": "No potentially harmful behavior detected.",
            "guidance": (
                "No strong signs of toxic language or "
                "cyberbullying context were detected."
            )
        }


    print(
        "[Agent] Potentially harmful behavior detected."
    )

    print(
        "[Agent] Retrieving relevant knowledge..."
    )

    context = retrieve_knowledge(text)


    print(
        "[Agent] Sending information to Qwen3 4B..."
    )



    prompt = f"""
You are a supportive cyberbullying safety assistant.

User input:
{text}

AI analysis:

Toxicity score:
{toxicity_score:.3f}

Cyberbullying context score:
{context_score:.3f}

Context rule signal:
{context_signal}

Relevant safety knowledge:
{context}

Your task is to provide short, practical, and supportive safety guidance.

Important rules:

- Do not decide that the user is legally a victim of cyberbullying.
- Do not make legal conclusions.
- Do not encourage retaliation, revenge, harassment, or confrontation.
- Do not shame or blame the user.
- Consider targeting, repetition, threats, harassment, and severity.
- If the user appears to be experiencing harassment or repeated targeting,
  acknowledge the concern and suggest practical steps such as blocking,
  reporting, saving evidence, and seeking help from a trusted person.
- If the input is mainly toxic or insulting language, explain briefly
  that the language may be harmful and suggest respectful communication.
- If there is no clear indication that the user is being targeted,
  avoid assuming that cyberbullying is occurring.
- Use the provided knowledge when relevant.
- Do not provide country-specific emergency or crisis hotline numbers unless the user has clearly provided their country or location.
- Do not mention internal model details, scores, RAG, or AI analysis
  in the response.
- Keep the response concise, clear, calm, and supportive.
"""

    response = llm.invoke(prompt)


    if toxicity_detected and context_detected:
        result = "Potential toxic language and cyberbullying context detected."
        reason = "Both toxic language and cyberbullying context were detected."
    elif context_detected:
        result = "Potential cyberbullying context detected."
        if context_signal:
            reason = "Cyberbullying context detected through the context rule signal."
        else:
            reason = "Cyberbullying context detected by the context model."
    else:
        result = "Potentially harmful/toxic language detected."
        reason = "Toxic language was detected by the toxicity model."


    return {
        "toxicity_score": toxicity_score,
        "context_score": context_score,
        "context_signal": context_signal,
        "result": result,
        "reason": reason,
        "guidance": response.content
    }


# ============================================
# Test the agent
# ============================================

if __name__ == "__main__":
    text = input("Enter a comment or situation: ")

    result = agent(text)

    print("\nRESULT:")
    print(result["result"])

    print("\nTOXICITY SCORE:")
    print(result["toxicity_score"])

    print("\nCYBERBULLYING CONTEXT SCORE:")
    print(result["context_score"])

    print("\nAI GUIDANCE:")
    print(result["guidance"])
