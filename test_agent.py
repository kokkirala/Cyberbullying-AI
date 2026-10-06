from agent import agent

test_cases = [
    "I really enjoyed this video, thank you!",
    "You are stupid and nobody likes you.",
    "He keeps harassing me every day and threatening me.",
    "They make fun of me every day at school.",
    "This is a beautiful day.",
    "Stop insulting me repeatedly.",
    "You're a disgusting loser and you keep bullying me every day.",
]

for text in test_cases:
    print("\n" + "=" * 60)
    print("INPUT:", text)

    result = agent(text)

    print("RESULT:", result["result"])
    print("REASON:", result.get("reason", "No concern detected."))
    print("TOXICITY:", round(result["toxicity_score"], 3))
    print("CONTEXT:", round(result["context_score"], 3))
    print("GUIDANCE:", result["guidance"])
