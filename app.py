import gradio as gr

from agent import agent


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

    result = agent(text)

    if result["result"] == "No potentially harmful behavior detected.":

        heading = "### 🟢 No concerning behavior detected"

    elif (
        "toxic language" in result["result"].lower()
        and "cyberbullying" in result["result"].lower()
    ):

        heading = (
            "### 🔴 Potential toxic language "
            "and cyberbullying context detected"
        )

    elif "cyberbullying" in result["result"].lower():

        heading = "### 🟠 Potential cyberbullying context detected"

    else:

        heading = "### 🔴 Potentially harmful/toxic language detected"

    return (
        heading,
        f"**Toxicity score:** `{result['toxicity_score']:.3f}`",
        f"**Cyberbullying context score:** `{result['context_score']:.3f}`",
        result["guidance"]
    )
# ============================================
# Gradio interface
# ============================================

with gr.Blocks(
    title="Cyberbullying AI"
) as demo:

    gr.Markdown("""
# 🛡️ Cyberbullying AI

### AI-powered online safety analysis

This system combines:

- **Toxicity detection**
- **Cyberbullying context detection**
- **RAG knowledge retrieval**
- **Qwen3 4B**
- **AI agent reasoning**
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


# ============================================
# Launch
# ============================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css="""
        .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
        }
        """
    )
