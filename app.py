import pickle
import chromadb
import tensorflow as tf
import pandas as pd
import gradio as gr

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
temperature=0.2,
num_predict=300,
)

# ============================================

# Model 1: Toxicity

# ============================================

def classify_toxicity(text):

    sequence = toxicity_tokenizer.texts_to_sequences([text])

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

# Model 2: Cyberbullying context

# ============================================

def classify_context(text):


    sequence = context_tokenizer.texts_to_sequences([text])

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

# Main analysis

# ============================================
def analyze(text):

    if not text.strip():
        return (
            "### Please enter a comment or situation.",
            "",
            "",
            ""
        )

    toxicity_score = classify_toxicity(text)
    context_score = classify_context(text)

    toxicity_detected = toxicity_score >= 0.5
    context_signal = detect_context_signal(text)

    context_detected = (
        context_score >= 0.80
        or context_signal
    )

    # ========================================
    # No concerning behavior
    # ========================================

    if not toxicity_detected and not context_detected:

        return (
            "### 🟢 No concerning behavior detected",
            f"**Toxicity score:** `{toxicity_score:.3f}`",
            f"**Cyberbullying context score:** `{context_score:.3f}`",
            "No strong signs of toxic language or "
            "cyberbullying context were detected."
        )

    # ========================================
    # Retrieve RAG knowledge
    # ========================================

    results = collection.query(
        query_texts=[text],
        n_results=3
    )

    knowledge = "\n\n".join(
        results["documents"][0]
    )

    # ========================================
    # Ask Qwen
    # ========================================

    prompt = f"""
You are a helpful cyberbullying support assistant.

User input:
{text}

Toxicity score:
{toxicity_score:.3f}

Cyberbullying context score:
{context_score:.3f}

Relevant knowledge:
{knowledge}

Give a short, supportive response.

Important:
- Do not encourage retaliation.
- Do not encourage harassment.
- Do not make legal conclusions.
- Do not automatically claim that every rude comment is cyberbullying.
- Consider context, targeting, repetition, threats, and severity.
- If the person is describing harassment happening to them,
  provide practical safety guidance.
- Keep the response simple and supportive.
"""


    response = llm.invoke(prompt)

    guidance = str(response.content).strip()

    if not guidance:
        guidance = (
            "Please save evidence, avoid responding aggressively, "
            "and consider blocking or reporting the account."
        )

    # ========================================
    # Determine result
    # ========================================

    if toxicity_detected and context_detected:

        result = (
            "### 🔴 Potential toxic language "
            "and cyberbullying context detected"
        )

    elif context_detected:

        result = (
            "### 🟠 Potential cyberbullying context detected"
        )

    else:

        result = (
            "### 🔴 Potentially harmful/toxic language detected"
        )

    return (
        result,
        f"**Toxicity score:** `{toxicity_score:.3f}`",
        f"**Cyberbullying context score:** `{context_score:.3f}`",
        guidance
    )

# ============================================

# Gradio interface

# ============================================

with gr.Blocks(
    title="Cyberbullying AI",
    css="""
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
    }
    """
) as demo:

    gr.Markdown("""
# 🛡️ Cyberbullying AI

### AI-powered online safety analysis

This system combines:

- **Toxicity detection**
- **Cyberbullying context detection**
- **RAG knowledge retrieval**
- **Qwen3 4B**
- **AI safety guidance**

> **Note:** This system identifies potentially harmful language
> or cyberbullying context. It does not make a final legal
> determination of cyberbullying.
""")

    comment_input = gr.Textbox(
        label="Enter a comment or situation",
        placeholder="Example: He keeps sending me threatening messages.",
        lines=6
    )

    analyze_button = gr.Button(
        "🔍 Analyze",
        variant="primary"
    )

    result_output = gr.Markdown()

    toxicity_output = gr.Markdown()

    context_output = gr.Markdown()

    gr.Markdown("## 🤖 AI Guidance")

    guidance_output = gr.Markdown()

    analyze_button.click(
        fn=analyze,
        inputs=comment_input,
        outputs=[
            result_output,
            toxicity_output,
            context_output,
            guidance_output
        ]
    )


if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
