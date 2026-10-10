# Cyberbullying AI

An AI-powered cyberbullying analysis system combining machine learning, contextual detection, rule-based signals, retrieval-augmented generation (RAG), and a locally hosted large language model.

## Overview

The system analyzes user-provided comments and descriptions of possible harassment using:

1. A toxicity detection model.
2. A cyberbullying context detection model.
3. Rule-based signals for patterns such as repeated harassment and threats.
4. ChromaDB for retrieving relevant safety information.
5. Qwen3 4B through Ollama to generate supportive safety guidance.
6. Gradio for the web interface.

**Important:** Predictions are preliminary signals, not proof that cyberbullying occurred. The system can make mistakes, particularly when distinguishing online harassment from in-person bullying.

## Architecture

```text
User Input
    |
    +------------------------+
    |                        |
    v                        v
Toxicity Model        Context Model
    |                        |
    |                Context Rule Signal
    |                        |
    +-----------+------------+
                |
                v
        Detection Decision
                |
        Potential concern?
          /           \
        No             Yes
        |               |
        v               v
  Safe response    ChromaDB RAG
                        |
                        v
                    Qwen3 4B
                        |
                        v
                 Safety Guidance
```

## Requirements

- Python version compatible with the installed TensorFlow release.
- Git.
- Ollama installed and running.
- The `qwen3:4b` model downloaded through Ollama.
- The project's model files and datasets.

TensorFlow on native Windows uses CPU execution in this setup; GPU support may require a different environment.

The trained models, tokenizer, and context-training dataset must be present at the paths expected by the application. The application also requires the ChromaDB knowledge base, which can be created by running `python build_rag.py`.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/kokkirala/Cyberbullying-AI.git
cd Cyberbullying-AI
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate
```

Install Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the local language model:

```bash
ollama pull qwen3:4b
```

Ensure Ollama is running before testing the complete agent or using the application.

## Prepare the Knowledge Base

Build the ChromaDB knowledge collection:

```bash
python build_rag.py
```

Run this step again if the local knowledge database is missing or needs rebuilding.

## Run the Application

Start the Gradio application:

```bash
python app.py
```

Open the local URL printed in the terminal, typically `http://127.0.0.1:7860`.

## Tests

Run the individual tests:

```bash
python test_classifier.py
python test_context_model.py
python test_rag.py
python test_agent.py
```

The classifier and context-model tests may print TensorFlow informational messages. These are not necessarily errors.

Some tests call an interactive input prompt or the local Ollama model. Keep Ollama running when testing the complete agent.

## Project Structure

```text
Cyberbullying-AI/
├── agent.py
├── app.py
├── build_rag.py
├── clean_data.py
├── train_model.py
├── train_context_model.py
├── train_test_split.py
├── test_agent.py
├── test_classifier.py
├── test_context_model.py
├── test_rag.py
├── data/
├── knowledge/
├── model/
├── requirements.txt
└── README.md
```

## Project Status

The application and its main test scripts have been run locally. The system combines trained machine-learning models, rule-based context signals, retrieval-augmented generation, and a locally hosted language model.

Testing on sample inputs does not guarantee accurate results for every real-world situation. Further evaluation on diverse, independently labeled data is needed to measure reliability.

## Limitations and Safety

- Model predictions can produce false positives and false negatives.
- A toxic comment is not automatically cyberbullying; context, repetition, targeting, and whether the behavior is online matter.
- Rule-based signals may flag descriptions of in-person bullying.
- Generated guidance may be incomplete or inconsistent. Use human judgment and seek trusted support where appropriate.
- Do not treat this tool as a substitute for professional, legal, or emergency assistance.
- Avoid entering private or identifying information into the application.

## License

No license is specified yet. Check the repository for licensing terms before redistributing or reusing the project.
