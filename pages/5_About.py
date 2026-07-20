"""
pages/5_About.py
-------------------
Project description, tech stack, and developer credits.
"""

import streamlit as st

from utils.helpers import init_session_state, load_css, render_navbar, render_footer
from config import PROJECT_TITLE, PROJECT_TAGLINE

init_session_state()
load_css()
render_navbar(active="About")

st.markdown("## i About This Project")

st.markdown(
    f"""
    <div class="glass-card">
        <h4>{PROJECT_TITLE}</h4>
        <p style="color:var(--text-muted); font-size:0.95rem; margin-top:-6px;">{PROJECT_TAGLINE}</p>
        <p style="color:var(--text-muted)">
        This project analyzes the sentiment of Reddit-style posts and comments using a
        custom-trained neural network -- no third-party sentiment APIs, no HuggingFace,
        no OpenAI. Every prediction comes from a TF-IDF vectorizer and neural network
        trained from scratch on a manually assembled Reddit-style dataset (over 20,000
        rows, combining synthetic templates, real social-media text, and established
        sentiment lexicons for broad vocabulary coverage).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------------
# Developer cards
# ---------------------------------------------------------------
st.markdown("### Developers")
dev_col1, dev_col2 = st.columns(2)

with dev_col1:
    st.markdown(
        """
        <div class="dev-card">
            <div class="dev-avatar">PK</div>
            <div class="dev-name">Pawni Kaushik</div>
            <div class="dev-role">Developer &middot; Internship Project 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with dev_col2:
    st.markdown(
        """
        <div class="dev-card">
            <div class="dev-avatar">KS</div>
            <div class="dev-name">Khushi Sharma</div>
            <div class="dev-role">Developer &middot; Internship Project 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.divider()

# ---------------------------------------------------------------
# Tech stack
# ---------------------------------------------------------------
st.markdown("### Tech Stack")
tech_col1, tech_col2, tech_col3 = st.columns(3)
with tech_col1:
    st.markdown(
        """
        <div class="glass-card">
            <strong>ML / NLP</strong>
            <ul style="color:var(--text-muted)">
                <li>TensorFlow / Keras</li>
                <li>Scikit-learn (TF-IDF)</li>
                <li>NLTK</li>
                <li>Pandas / NumPy</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with tech_col2:
    st.markdown(
        """
        <div class="glass-card">
            <strong>App / Visualization</strong>
            <ul style="color:var(--text-muted)">
                <li>Streamlit (multipage)</li>
                <li>Plotly</li>
                <li>Custom CSS (glassmorphism)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with tech_col3:
    st.markdown(
        """
        <div class="glass-card">
            <strong>Data Sources</strong>
            <ul style="color:var(--text-muted)">
                <li>Manually curated templates</li>
                <li>NLTK Twitter Samples corpus</li>
                <li>VADER + Bing Liu sentiment lexicons</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.divider()

# ---------------------------------------------------------------
# Acknowledgements
# ---------------------------------------------------------------
st.markdown("### Acknowledgements")
st.markdown(
    """
    <div class="glass-card" style="color:var(--text-muted)">
    Built as part of an NLP/Machine Learning internship project. Thanks to the
    open-source NLTK, scikit-learn, TensorFlow, and Streamlit communities whose
    tools made this possible.
    </div>
    """,
    unsafe_allow_html=True,
)

render_footer()
