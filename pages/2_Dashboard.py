"""
pages/2_Dashboard.py
----------------------
Dataset-facing dashboard: class balance, most common words per class,
comment length distributions -- generated from the actual training
dataset (reddit_sentiment_dataset_v9.csv), not simulated numbers.
"""

import streamlit as st

from utils.helpers import init_session_state, load_css, render_navbar, render_footer
from utils.metrics import get_dataset_stats
from utils.charts import sentiment_pie_chart, sentiment_bar_chart, word_frequency_chart, length_histogram
from config import COLORS

init_session_state()
load_css()
render_navbar(active="Dashboard")

stats = get_dataset_stats()
dark = st.session_state.dark_mode

st.markdown("## ◆ Dataset Dashboard")
st.markdown(
    "<p style='color:var(--text-muted)'>A look at the data the model was actually trained on.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="stat-tile"><div class="stat-value">{stats['total_rows']:,}</div>
                <div class="stat-label">Total Comments</div></div>""", unsafe_allow_html=True)
with c2:
    pos_pct = stats['class_counts'].get('positive', 0) / stats['total_rows'] * 100
    st.markdown(f"""<div class="stat-tile"><div class="stat-value" style="color:var(--positive)">{pos_pct:.1f}%</div>
                <div class="stat-label">Positive</div></div>""", unsafe_allow_html=True)
with c3:
    neg_pct = stats['class_counts'].get('negative', 0) / stats['total_rows'] * 100
    st.markdown(f"""<div class="stat-tile"><div class="stat-value" style="color:var(--negative)">{neg_pct:.1f}%</div>
                <div class="stat-label">Negative</div></div>""", unsafe_allow_html=True)
with c4:
    neu_pct = stats['class_counts'].get('neutral', 0) / stats['total_rows'] * 100
    st.markdown(f"""<div class="stat-tile"><div class="stat-value" style="color:var(--neutral)">{neu_pct:.1f}%</div>
                <div class="stat-label">Neutral</div></div>""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------
# Class distribution charts
# ---------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("#### Class Distribution")
    st.plotly_chart(sentiment_pie_chart(stats["class_counts"], dark=dark), width="stretch")
with chart_col2:
    st.markdown("#### Class Counts")
    st.plotly_chart(sentiment_bar_chart(stats["class_counts"], dark=dark), width="stretch")

st.divider()

# ---------------------------------------------------------------
# Most common words per class
# ---------------------------------------------------------------
st.markdown("#### Most Common Words per Sentiment")
word_tabs = st.tabs(["Positive", "Negative", "Neutral"])
color_map = {"positive": COLORS["positive"], "negative": COLORS["negative"], "neutral": COLORS["neutral"]}

for tab, label in zip(word_tabs, ["positive", "negative", "neutral"]):
    with tab:
        words = stats["top_words_by_class"].get(label, [])
        if words:
            st.plotly_chart(
                word_frequency_chart(words, color=color_map[label], dark=dark),
                width="stretch",
            )
        else:
            st.caption("No data available for this class.")

st.divider()

# ---------------------------------------------------------------
# Comment length distributions
# ---------------------------------------------------------------
st.markdown("#### Comment Length Distribution")
len_col1, len_col2 = st.columns(2)
with len_col1:
    st.markdown("**Character length**")
    st.plotly_chart(
        length_histogram(stats["char_length_bins"], stats["char_length_hist"], dark=dark, color=COLORS["brand_orange"]),
        width="stretch",
    )
    st.caption(f"Mean: {stats['char_length_mean']:.0f} chars | Median: {stats['char_length_median']:.0f} chars")
with len_col2:
    st.markdown("**Word count**")
    st.plotly_chart(
        length_histogram(stats["word_count_bins"], stats["word_count_hist"], dark=dark, color=COLORS["positive"]),
        width="stretch",
    )
    st.caption(f"Mean: {stats['word_count_mean']:.1f} words | Median: {stats['word_count_median']:.0f} words")

render_footer()
