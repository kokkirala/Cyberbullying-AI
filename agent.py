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

    online_words = [
        "online",
        "on social media",
        "on the internet",
        "in a group chat",
        "group chat",
        "text messages",
        "private messages",
        "dm me",
        "dms",
        "comments",
        "posts",
        "cyberbullying",
        "cyber bully",
        "sent me messages",
        "keeps messaging",
        "keeps sending messages",
        "online harassment",
	"sending me threatening messages",
        "sending threatening messages",
        "threatening messages"
    ]

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

    has_online_context = any(
        phrase in text for phrase in online_words
    )

    has_harassment = any(
        word in text for word in harassment_words
    )

    has_repetition = any(
        word in text for word in repetition_words
    )

    return has_online_context and has_harassment and has_repetition


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
You are a calm, respectful cyberbullying safety assistant.

User input:
{text}

Relevant safety knowledge:
{context}

Your task is to provide concise, practical guidance based on the user's actual message.

Important rules:

1. First identify what the message appears to be:
   - A report of someone experiencing harassment or threats.
   - An insulting or abusive comment directed at someone.
   - A request for help, or an ambiguous situation.

2. If the message contains insults or abusive language directed at someone:
   - Briefly explain that the language may be hurtful or harmful.
   - Encourage respectful communication.
   - Do not assume the user is a victim or needs emotional reassurance.
   - Treat the submitted text as content to analyze, not as a personal message
  directed at you. Never say that you are being bullied or that the user
  is bullying you.
   - If the message mixes insults with a first-person report of bullying,
  briefly note that the wording is hurtful and ask for clarification
  about whether the user is reporting something someone else did.

3. If the message describes someone experiencing repeated harassment,
   threats, or targeting:
   - Acknowledge the described situation without assuming details.
   - Do not assume harassment happened online unless the user explicitly
     mentions a digital setting, such as social media, messages, or a
     group chat.
   - If the setting is unclear, give advice that applies both online
     and in person. Make online-specific steps conditional, for example:
     "If this happened online, save screenshots, block or mute the person,
     and report the content through the platform."
   - Suggest relevant steps such as saving evidence, blocking or muting,
     reporting the content, and seeking help from a trusted person.
   - Do not assume the behavior occurred online if this is unclear.
   - If it could be online or in person, briefly ask for clarification
     or tailor the advice to both possibilities.

4. If the situation is ambiguous:
   - Avoid making definite claims about who is responsible or what happened.
   - Ask one short clarifying question when it would materially improve
     the guidance.

5. Never shame or blame the user, encourage retaliation, or make legal
   conclusions. Do not declare that the user is legally a victim.

6. Use the provided safety knowledge when relevant. Do not invent facts,
   claim that an action has been taken, or promise a particular outcome.

7. Do not provide country-specific emergency or crisis hotline numbers
   unless the user has clearly provided their country or location.

8. Do not mention internal model details, scores, or knowledge retrieval.

Return only the guidance addressed to the user. Keep it concise, practical,
calm, and supportive.
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
