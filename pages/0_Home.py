"""
pages/0_Home.py
-----------------
Product landing page, shown once the user is signed in. This used to
live directly in app.py, but app.py is now a slim router (auth gate +
st.navigation) so every page -- including Home -- is a real, orderable
entry in the nav list.
"""

import streamlit as st

from utils.helpers import init_session_state, load_css, render_navbar, render_footer
from utils.metrics import get_headline_numbers, get_ner_accuracy

init_session_state()
load_css()
render_navbar(active="Home")

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-eyebrow">Two self-trained neural networks &middot; Zero third-party AI APIs</div>
        <div class="hero-title">Know the room<br>before you <span class="accent">post</span>.</div>
        <div class="hero-subtitle">
            Paste any text and get an instant Positive / Negative / Neutral verdict,
            powered by a custom-trained Bidirectional LSTM and a dedicated Named
            Entity Recognition model.
        </div>
        <div class="badge-row">
            <span class="tech-badge">TensorFlow / Keras</span>
            <span class="tech-badge">Bidirectional LSTM</span>
            <span class="tech-badge">Named Entity Recognition</span>
            <span class="tech-badge">Streamlit</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_b:
    st.page_link("pages/1_Analyzer.py", label="Try the Analyzer  →", icon="🚀")

st.write("")
st.write("")

# ---------------------------------------------------------------
# Headline stats
# ---------------------------------------------------------------
try:
    numbers = get_headline_numbers()
    ner_accuracy = get_ner_accuracy()

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.markdown(f"""<div class="stat-tile"><div class="stat-value">{numbers['test_accuracy']*100:.1f}%</div>
                    <div class="stat-label">Sentiment Model Accuracy</div></div>""", unsafe_allow_html=True)
    with stats_col2:
        ner_display = f"{ner_accuracy*100:.1f}%" if ner_accuracy is not None else "—"
        st.markdown(f"""<div class="stat-tile"><div class="stat-value">{ner_display}</div>
                    <div class="stat-label">NER Model Accuracy</div></div>""", unsafe_allow_html=True)
    with stats_col3:
        st.markdown(f"""<div class="stat-tile"><div class="stat-value">{numbers['total_training_rows']:,}</div>
                    <div class="stat-label">Sentiment Training Rows</div></div>""", unsafe_allow_html=True)
    with stats_col4:
        st.markdown(f"""<div class="stat-tile"><div class="stat-value">{numbers['vocab_size']:,}</div>
                    <div class="stat-label">Vocabulary Size</div></div>""", unsafe_allow_html=True)
except Exception:
    pass

st.write("")
st.write("")

# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------
st.markdown('<div class="section-eyebrow">How it works</div>', unsafe_allow_html=True)
st.markdown("### One pipeline, two models")

st.markdown(
    """
    <div class="pipeline-step"><span class="step-num">01</span>
        <div><b>Name detection.</b> A Bidirectional LSTM NER tagger flags and removes
        person names before analysis.</div>
    </div>
    <div class="pipeline-step"><span class="step-num">02</span>
        <div><b>Text cleaning.</b> Lowercasing, URL/punctuation stripping, and stopword
        removal, with negation words preserved.</div>
    </div>
    <div class="pipeline-step"><span class="step-num">03</span>
        <div><b>Tokenizing + padding.</b> Text is converted into a fixed-length sequence
        of word IDs.</div>
    </div>
    <div class="pipeline-step"><span class="step-num">04</span>
        <div><b>Sentiment prediction.</b> A Bidirectional LSTM outputs
        Positive / Negative / Neutral with a confidence score.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------------
# Feature overview
# ---------------------------------------------------------------
st.markdown('<div class="section-eyebrow">Explore</div>', unsafe_allow_html=True)
st.markdown("### What's inside")

feat_col1, feat_col2, feat_col3 = st.columns(3)
with feat_col1:
    st.markdown(
        """
        <div class="glass-card">
            <h4>▲ Analyzer</h4>
            <p style="color:var(--text-muted)">Instant sentiment verdicts with a
            confidence gauge, probability breakdown, and prediction history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with feat_col2:
    st.markdown(
        """
        <div class="glass-card">
            <h4>◆ Dashboard</h4>
            <p style="color:var(--text-muted)">Class balance, common words per
            sentiment, and comment length distributions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with feat_col3:
    st.markdown(
        """
        <div class="glass-card">
            <h4>● Performance</h4>
            <p style="color:var(--text-muted)">Confusion matrix, precision/recall/F1,
            training curves, and ROC.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_footer()
