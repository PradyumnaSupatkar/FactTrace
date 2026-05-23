import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


#App setup
st.set_page_config(
    page_title="TruthLens",
    page_icon="🔬",
    layout="centered",
)

st.title("FactTrace")
st.markdown(
    "Fine-tuned **DistilBERT** on the LIAR dataset · "
    "binary fake-vs-real classification with **SHAP word-level explanations**."
)


#Project settings
MODEL_DIR = "./model/saved_distilbert"
MAX_LENGTH = 128

LABELS = ["FAKE", "REAL"]

LABEL_STYLES = {
    "FAKE": {"color": "#ef4444", "emoji": "🔴"},
    "REAL": {"color": "#22c55e", "emoji": "🟢"},
}

EXAMPLES = [
    "Scientists have confirmed that regular sleep of 7-8 hours improves cognitive performance.",
    "The government has been secretly adding chemicals to tap water to control the population.",
    "A new study published in Nature links air pollution to increased risk of dementia.",
    "Eating a single piece of fruit daily eliminates all risk of developing diabetes.",
    "The Federal Reserve raised interest rates by 25 basis points at its latest meeting.",
    "Microwave ovens emit radiation that permanently damages your DNA with every use.",
    "Several states have passed legislation expanding early voting access for residents.",
    "A viral photo shows the president shaking hands with a known criminal organization leader.",
]


#Model loading
@st.cache_resource(show_spinner="Loading TruthLens model…")
def load_model():
    if not os.path.exists(MODEL_DIR):
        st.error(
            f"Model not found at `{MODEL_DIR}`. "
            "Run `train_model.ipynb` first to train and save the model."
        )
        st.stop()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#Prediction helper
def predict_proba(texts):
    """Return probabilities in this order: [FAKE, REAL]."""
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits

    return torch.softmax(logits, dim=-1).cpu().numpy()


#Plot helpers
def plot_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.barh(
        LABELS,
        probabilities,
        color=["#ef4444", "#22c55e"],
        height=0.5,
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10, color="white")
    ax.invert_yaxis()
    ax.tick_params(colors="white", left=False)
    ax.spines[["top", "right", "left", "bottom"]].set_color("#444")

    fig.tight_layout()
    return fig


def plot_shap_values(shap_values, predicted_label, confidence):

    values = shap_values[0, :, 1].values
    tokens = shap_values[0, :, 1].data

    top_k = min(12, len(values))
    top_indices = np.argsort(np.abs(values))[-top_k:]
    sorted_indices = top_indices[np.argsort(values[top_indices])]

    plot_values = values[sorted_indices]
    plot_tokens = [str(token) for token in tokens[sorted_indices]]
    colors = ["#ef4444" if value < 0 else "#22c55e" for value in plot_values]

    fig, ax = plt.subplots(figsize=(7, max(3, top_k * 0.45)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.barh(plot_tokens, plot_values, color=colors)
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value  (negative → FAKE, positive → REAL)", color="white")
    ax.set_title(f"Top tokens · predicted: {predicted_label} ({confidence:.1%})", color="white")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")

    fig.tight_layout()
    return fig


#Input area
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.markdown("### Enter a news statement")

input_col, button_col = st.columns([3, 1])

with button_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎲 Random example"):
        st.session_state.input_text = random.choice(EXAMPLES)
        st.rerun()

with input_col:
    user_text = st.text_area(
        "Statement",
        height=110,
        placeholder="Paste or type a news statement…",
        label_visibility="collapsed",
        key="input_text",
    )

show_shap = st.checkbox("Show SHAP word-level explanation", value=True)
run_analysis = st.button("Analyse", type="primary", use_container_width=True)


#Prediction output
if run_analysis and user_text.strip():
    with st.spinner("Analysing…"):
        probabilities = predict_proba([user_text])[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = LABELS[predicted_index]
        confidence = probabilities[predicted_index]
        style = LABEL_STYLES[predicted_label]

    st.markdown("---")
    st.markdown(
        f"<h2 style='color:{style['color']};text-align:center'>"
        f"{style['emoji']} {predicted_label}"
        f"</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;color:gray'>"
        f"Confidence: <b>{confidence:.1%}</b>"
        f"</p>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Prediction probabilities")
    probability_fig = plot_probabilities(probabilities)
    st.pyplot(probability_fig)
    plt.close(probability_fig)

    if show_shap:
        st.markdown("#### SHAP word-level explanation")
        st.caption(
            "**Red** bars push toward FAKE · **Green** bars push toward REAL. "
            "Showing the top 12 most influential tokens."
        )

        with st.spinner("Computing SHAP values…"):
            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(predict_proba, masker, max_evals=300)
            shap_values = explainer([user_text])

        shap_fig = plot_shap_values(shap_values, predicted_label, confidence)
        st.pyplot(shap_fig)
        plt.close(shap_fig)

elif run_analysis:
    st.warning("Please enter a statement first.")