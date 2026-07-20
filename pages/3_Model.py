"""
pages/3_Model.py
-------------------
Explains the NLP + neural network pipeline in plain language, for
anyone (interviewer, viva panel, curious user) who wants to understand
how the prediction actually happens, step by step.
"""

import streamlit as st

from utils.helpers import init_session_state, load_css, render_navbar, render_footer

init_session_state()
load_css()
render_navbar(active="Model")

st.markdown("## ● How the Model Works")
st.markdown(
    "<p style='color:var(--text-muted)'>Every prediction goes through the same "
    "pipeline, end to end.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------
steps = [
    ("1", "Lowercasing", "All text is converted to lowercase so \"Great\" and \"great\" are treated identically."),
    ("2", "URL / Punctuation / Number Removal", "Links, punctuation marks, and digits are stripped out -- they carry no sentiment signal on their own."),
    ("3", "Tokenization", "The cleaned sentence is split into individual words (tokens)."),
    ("4", "Stopword Removal", "Common filler words (\"the\", \"is\", \"and\"...) are removed since they don't carry sentiment."),
    ("5", "TF-IDF Vectorization", "Each remaining word is converted into a numeric weight based on how distinctive it is across the whole training set -- rare, meaningful words get more weight than common ones."),
    ("6", "Neural Network Forward Pass", "The TF-IDF vector is fed through a feedforward neural network (Dense layers with Dropout for regularization)."),
    ("7", "Softmax Output", "The final layer outputs three probabilities (positive/negative/neutral) that sum to 100%."),
    ("8", "Prediction", "The highest-probability class is returned as the final sentiment, along with its confidence score."),
]

for num, title, desc in steps:
    st.markdown(
        f"""
        <div class="pipeline-step">
            <div class="step-num">{num}</div>
            <div><strong>{title}</strong><br><span style="color:var(--text-muted); font-size:0.92rem">{desc}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------
# Architecture diagram (simple HTML/CSS box diagram -- no external
# image dependency, renders identically in light/dark mode)
# ---------------------------------------------------------------
st.markdown("### Neural Network Architecture")

st.markdown(
    """
    <div class="glass-card" style="font-family: var(--font-mono); line-height:2;">
    Input Layer &nbsp;(TF-IDF vector, up to 8,000 features)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Dense(128, activation='relu')<br>
    Dropout(0.3)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Dense(64, activation='relu')<br>
    Dropout(0.2)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Dense(32, activation='relu')<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Dense(3, activation='softmax')<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Output: [P(negative), P(neutral), P(positive)]
    </div>
    """,
    unsafe_allow_html=True,
)

detail_col1, detail_col2 = st.columns(2)
with detail_col1:
    st.markdown(
        """
        <div class="glass-card">
            <h4>Training Configuration</h4>
            <ul style="color:var(--text-muted)">
                <li><strong>Optimizer:</strong> Adam</li>
                <li><strong>Loss function:</strong> Sparse Categorical Crossentropy</li>
                <li><strong>Class weighting:</strong> Balanced (compensates for the dataset
                having more negative words than positive, since real sentiment
                lexicons are naturally skewed that way)</li>
                <li><strong>Early stopping:</strong> Monitors validation loss, restores
                best weights, patience of 5 epochs</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with detail_col2:
    st.markdown(
        """
        <div class="glass-card">
            <h4>Why TF-IDF + Dense layers?</h4>
            <p style="color:var(--text-muted)">
            This is a <strong>bag-of-words</strong> approach: it's fast, interpretable,
            and works well for a project of this scope. Its main trade-off is that it
            doesn't track word order the way a sequence model (LSTM/Transformer) would --
            a deliberate simplicity-vs-nuance choice for this pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_footer()
