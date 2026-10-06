# Cyberbullying AI

An AI-powered cyberbullying analysis system that combines machine learning, contextual detection, rule-based signals, retrieval-augmented generation (RAG), and a local large language model.

## Overview

The system analyzes user-provided comments or online harassment situations using two machine learning models:

1. A toxicity detection model
2. A cyberbullying context detection model

It also uses rule-based contextual signals to identify patterns such as repeated harassment, threats, and targeting.

When potentially harmful behavior is detected, the system retrieves relevant information from a cyberbullying knowledge base using ChromaDB and uses Qwen3 4B through Ollama to generate supportive safety guidance.

The application is presented through a Gradio web interface.

## Architecture

```text
User Input
    |
    +------------------------+
    |                        |
    v                        v
Toxicity Model        Context Model
    |                        |
    |                        v
    |                Context Score
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
