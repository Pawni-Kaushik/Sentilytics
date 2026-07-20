"""
pages/1_Analyzer.py
---------------------
The main interactive page: type/paste text, get a sentiment verdict
with confidence gauge, probability breakdown, contributing words, and
a running history you can export as CSV.
"""

import time
import streamlit as st
import pandas as pd

from utils.helpers import (
    init_session_state, load_css, render_navbar, render_footer,
    add_to_history, get_history_df, clear_history, search_history,
)
from utils.loader import load_model_artifacts
from utils.predictor import predict_sentiment, get_top_contributing_words
from utils.charts import probability_bar_chart, confidence_gauge

init_session_state()
load_css()
render_navbar(active="Analyzer")

model, tokenizer, encoder = load_model_artifacts()

st.markdown("## ▲ Sentiment Analyzer")
st.markdown(
    "<p style='color:var(--text-muted)'>Paste a Reddit-style comment, post, or any text below.</p>",
    unsafe_allow_html=True,
)
st.caption("💡 Please enter names with capital alphabets (e.g. \"Rahul\", \"Singh\") — the name-detection model relies on capitalization to correctly identify and mask person names before scoring.")

user_text = st.text_area(
    "Your text",
    height=160,
    placeholder="e.g. This update completely ruined the app, I want the old version back.",
    label_visibility="collapsed",
)

analyze_clicked = st.button("Analyze Sentiment", type="primary")

if analyze_clicked:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        placeholder = st.empty()
        placeholder.info("🤖 AI is analyzing your comment...")
        start = time.time()
        result = predict_sentiment(user_text, model, tokenizer, encoder)
        elapsed = round(time.time() - start, 4)
        placeholder.empty()

        add_to_history(user_text, result)

        sentiment = result["prediction"]
        arrow = {"positive": "▲", "negative": "▼", "neutral": "—"}[sentiment]
        label_text = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}[sentiment]
        emoji_row = {
            "positive": "😊 🎉 💚 ✨",
            "negative": "😡 💔 ⚠️ 😞",
            "neutral": "😐 💬 🤔",
        }[sentiment]

        result_col, gauge_col = st.columns([1.3, 1])

        with result_col:
            st.markdown(
                f"""
                <div class="result-card result-{sentiment}">
                    <div class="vote-arrow">{arrow}</div>
                    <div class="result-label">{label_text}</div>
                    <div class="result-sub">{emoji_row}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result["is_oov"]:
                st.caption(
                    "⚠️ None of these words were seen during training -- defaulting to "
                    "Neutral rather than guessing blindly."
                )

        with gauge_col:
            st.plotly_chart(
                confidence_gauge(result["confidence"], sentiment, dark=st.session_state.dark_mode),
                width="stretch",
            )
            st.caption(f"Prediction time: {elapsed} sec")

        st.markdown("#### Probability Distribution")
        st.plotly_chart(
            probability_bar_chart(result["probabilities"], dark=st.session_state.dark_mode),
            width="stretch",
        )

        word_col1, word_col2 = st.columns(2)
        with word_col1:
            st.markdown("#### Top Contributing Words")
            top_words = get_top_contributing_words(result["cleaned_text"], tokenizer)
            if top_words:
                words_df = pd.DataFrame(top_words, columns=["Word", "Rarity Score"])
                st.dataframe(words_df, width="stretch", hide_index=True)
                st.caption(
                    "Rarity Score (0-100): how uncommon this word was in the training "
                    "vocabulary. Rarer, more specific words tend to carry more sentiment "
                    "signal than common ones like \"the\" or \"is\"."
                )
            else:
                st.caption("No recognized words to rank.")

        with word_col2:
            st.markdown("#### Cleaned Text")
            st.code(result["cleaned_text"] or "(nothing left after cleaning)", language="text")

        st.markdown("#### Copy Result")
        result_summary = (
            f"Text: {user_text}\n"
            f"Sentiment: {label_text}\n"
            f"Confidence: {result['confidence']:.2f}%"
        )
        st.text_area("Copyable summary", result_summary, height=100, label_visibility="collapsed")

st.divider()

# ---------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------
st.markdown("### 📊 Batch Sentiment Breakdown")
st.markdown(
    "<p style='color:var(--text-muted)'>Paste multiple comments, one per line, "
    "to see the overall positive/negative/neutral split. This is the same logic "
    "that runs automatically on live news data on the Live Search page.</p>",
    unsafe_allow_html=True,
)

batch_text = st.text_area(
    "Batch input",
    height=140,
    placeholder="This update is amazing!\nWorst experience ever.\nThe event starts at 6pm.",
    label_visibility="collapsed",
    key="batch_input",
)

if st.button("Analyze Batch"):
    lines = [line.strip() for line in batch_text.split("\n") if line.strip()]
    if not lines:
        st.warning("Please paste at least one line of text.")
    else:
        results = []
        for line in lines:
            result = predict_sentiment(line, model, tokenizer, encoder)
            results.append(result["prediction"])
            add_to_history(line, result)
        results_df = pd.DataFrame({"text": lines, "sentiment": results})
        counts = results_df["sentiment"].value_counts()
        percentages = (counts / counts.sum() * 100).round(2)

        pct_col1, pct_col2, pct_col3 = st.columns(3)
        with pct_col1:
            st.markdown(f"""<div class="stat-tile"><div class="stat-value" style="color:var(--positive)">
                        {percentages.get('positive', 0.0)}%</div><div class="stat-label">Positive</div></div>""",
                        unsafe_allow_html=True)
        with pct_col2:
            st.markdown(f"""<div class="stat-tile"><div class="stat-value" style="color:var(--negative)">
                        {percentages.get('negative', 0.0)}%</div><div class="stat-label">Negative</div></div>""",
                        unsafe_allow_html=True)
        with pct_col3:
            st.markdown(f"""<div class="stat-tile"><div class="stat-value" style="color:var(--neutral)">
                        {percentages.get('neutral', 0.0)}%</div><div class="stat-label">Neutral</div></div>""",
                        unsafe_allow_html=True)

        st.dataframe(results_df, width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------
st.markdown("### 🕒 Prediction History (this session)")
history_df = get_history_df()

if history_df.empty:
    st.caption("No predictions yet -- analyze some text above to build a history.")
else:
    # --- Per-label counts, so the filter has context before you use it ---
    label_counts = history_df["sentiment"].value_counts()
    count_col1, count_col2, count_col3, count_col4 = st.columns(4)
    with count_col1:
        st.markdown(f"""<div class="stat-tile"><div class="stat-value">{len(history_df)}</div>
                    <div class="stat-label">Total</div></div>""", unsafe_allow_html=True)
    with count_col2:
        st.markdown(f"""<div class="stat-tile"><div class="stat-value" style="color:var(--positive)">
                    {label_counts.get('positive', 0)}</div><div class="stat-label">Positive</div></div>""",
                    unsafe_allow_html=True)
    with count_col3:
        st.markdown(f"""<div class="stat-tile"><div class="stat-value" style="color:var(--negative)">
                    {label_counts.get('negative', 0)}</div><div class="stat-label">Negative</div></div>""",
                    unsafe_allow_html=True)
    with count_col4:
        st.markdown(f"""<div class="stat-tile"><div class="stat-value" style="color:var(--neutral)">
                    {label_counts.get('neutral', 0)}</div><div class="stat-label">Neutral</div></div>""",
                    unsafe_allow_html=True)

    st.write("")

    # --- Filter by label AND by keyword (both apply together) ---
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        label_filter = st.selectbox(
            "Filter by sentiment",
            options=["All", "Positive", "Negative", "Neutral"],
            index=0,
        )
    with filter_col2:
        keyword = st.text_input(
            "Filter by word",
            placeholder="e.g. type a word like 'refund' to see only posts mentioning it",
        )

    filtered_df = history_df
    if label_filter != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == label_filter.lower()]
    filtered_df = search_history(filtered_df, keyword)

    if filtered_df.empty:
        st.caption("No predictions match this filter yet.")
    else:
        st.caption(f"Showing {len(filtered_df)} of {len(history_df)} predictions.")
        st.dataframe(filtered_df, width="stretch", hide_index=True)

    hist_col1, hist_col2 = st.columns([1, 5])
    with hist_col1:
        st.download_button(
            "⬇ Download CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name=f"prediction_history_{label_filter.lower()}.csv",
            mime="text/csv",
        )
    with hist_col2:
        if st.button("Clear History"):
            clear_history()
            st.rerun()

render_footer()
