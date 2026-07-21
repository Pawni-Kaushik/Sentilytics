"""
pages/6_Live_News_Search.py
--------------------------------
Type any keyword/brand/topic -> finds real, live news articles about it
(via the GNews API) -> runs each article's title + description/content
through the trained sentiment model -> shows an overall
positive/negative/neutral breakdown.

This replaces the earlier Reddit-scraping version of this page.
Reddit was dropped because Streamlit Community Cloud (like most cloud
hosts) runs on datacenter IP ranges that Reddit blocks at the network
level, regardless of how the scraper was built -- see news_service.py
for the full explanation. GNews is a real, authenticated API call, so
it works identically whether run locally or deployed.

IMPORTANT implementation detail: search results are stored in
st.session_state (not a local variable). Streamlit reruns the entire
script on ANY widget interaction -- including just changing a filter
dropdown -- and a st.button()'s "clicked" state only stays True for the
single rerun triggered by that exact click. If results were kept in a
local variable, changing the sentiment/type filter below would trigger
a rerun where search_clicked is False again, making the whole results
section (including the table you're trying to filter) disappear. Storing
results in session_state means they persist across reruns caused by
any other widget, and are only replaced when a NEW search is run.
"""

import streamlit as st
import pandas as pd
from collections import Counter

from utils.helpers import init_session_state, load_css, render_navbar, render_footer
from utils.loader import load_model_artifacts
from utils.predictor import predict_long_text
from utils.charts import sentiment_pie_chart, word_cloud_chart

from news_service import search_news

init_session_state()
load_css()
render_navbar(active="Live Search")

model, tokenizer, encoder = load_model_artifacts()

st.markdown("## 🔎 Live News Search")
st.markdown(
    "<p style='color:var(--text-muted)'>Type a keyword, brand, or topic. This searches "
    "live news coverage (via the GNews API) and runs each article's title and "
    "description through the trained sentiment model.</p>",
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How this works"):
    st.markdown(
        """
        - Uses the [GNews API](https://gnews.io) to fetch live news articles matching your
          keyword -- a real authenticated API call, not scraping, so it works the same
          whether run locally or deployed.
        - Each article contributes its title + description/content snippet as one piece
          of text run through the sentiment model.
        - Results depend on what GNews has indexed recently; very niche or very recent
          topics may return fewer articles.
        - Each search takes a couple of seconds -- much faster than the old
          browser-scraping approach, since it's a single API call instead of loading
          full pages one by one.
        """
    )

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Search keyword",
        placeholder="e.g. Maruti Suzuki, iPhone 17, or any topic",
        label_visibility="collapsed",
    )
with col2:
    search_clicked = st.button("Search & Analyze", type="primary", width="stretch")

max_articles = st.slider("Max articles to fetch", min_value=1, max_value=10, value=10)

# ---------------------------------------------------------------
# Run a new search only when the button is freshly clicked. The
# results get written into session_state so they survive later
# reruns triggered by the filter widgets below.
# ---------------------------------------------------------------
if search_clicked:
    if not query.strip():
        st.warning("Please enter a keyword to search for.")
    else:
        with st.spinner(f"Searching news for '{query}'..."):
            articles, error = search_news(query, max_results=max_articles)

        if error == "missing_key":
            st.error(
                "GNEWS_API_KEY isn't set. Add it locally to a `.env` file, or on Streamlit "
                "Community Cloud go to your app -> Settings -> Secrets and add:\n\n"
                "`GNEWS_API_KEY = \"your-real-key-here\"`"
            )
            st.session_state.pop("live_search_results", None)
        elif error == "invalid_key":
            st.error(
                "GNews rejected this API key (401 Unauthorized). Double-check it was copied "
                "correctly (no extra spaces/quotes) from https://gnews.io/dashboard, and that "
                "the same key is saved in Streamlit Cloud's Secrets."
            )
            st.session_state.pop("live_search_results", None)
        elif error == "quota_exceeded":
            st.warning(
                "GNews's free-tier daily limit (100 requests) has been reached for this key. "
                "It resets at **00:00 UTC**. This is also why older searches from earlier today "
                "might still show results here -- those are saved from before the limit was hit, "
                "not a fresh call. Try again after the reset, or reuse a keyword you already "
                "searched today."
            )
            st.session_state.pop("live_search_results", None)
        elif error == "network_error":
            st.error("Couldn't reach GNews right now (network/timeout issue). Please try again.")
            st.session_state.pop("live_search_results", None)
        elif not articles:
            st.info(
                f"The search worked, but GNews doesn't have live coverage for '{query}' right "
                "now. Try a broader or more common keyword."
            )
            st.session_state.pop("live_search_results", None)
        else:
            all_rows = []
            total = len(articles)
            progress_text = st.empty()
            progress_bar = st.progress(0)
            for i, item in enumerate(articles):
                progress_text.markdown(
                    f"<p style='color:var(--text-muted)'>Analyzing article {i + 1} of "
                    f"{total} through the sentiment model...</p>",
                    unsafe_allow_html=True,
                )
                # Combine description + content for the richest text
                # available; GNews truncates "content" on the free tier,
                # so this may sometimes just be the description.
                combined_text = f"{item['title']} {item['preview']} {item['content']}".strip()
                if not combined_text:
                    progress_bar.progress((i + 1) / total)
                    continue

                result = predict_long_text(combined_text, model, tokenizer, encoder)
                all_rows.append({
                    "post_title": item["title"],
                    "type": "article",
                    "text": combined_text,
                    "cleaned_text": result["cleaned_text"],
                    "sentiment": result["prediction"],
                    "confidence": round(result["confidence"], 1),
                    "chunks_analyzed": len(result.get("chunk_results", [])) or 1,
                    "url": item["url"],
                    "source": item.get("source", ""),
                })
                progress_bar.progress((i + 1) / total)

            progress_text.empty()
            progress_bar.empty()

            if not all_rows:
                st.error("Articles were found, but none had usable text. Try a different keyword.")
                st.session_state.pop("live_search_results", None)
            else:
                st.session_state["live_search_results"] = {
                    "query": query,
                    "results_df": pd.DataFrame(all_rows),
                    "post_count": len(all_rows),
                }

# ---------------------------------------------------------------
# Always render from session_state (if present) -- this runs on
# EVERY rerun, whether triggered by the search button or by the
# filter dropdowns below, so results stay visible while filtering.
# ---------------------------------------------------------------
if "live_search_results" in st.session_state:
    stored = st.session_state["live_search_results"]
    results_df = stored["results_df"]
    stored_query = stored["query"]
    post_count = stored["post_count"]

    st.success(f"Analyzed {post_count} article(s) for '{stored_query}'.")
    st.caption(
        "Longer articles are split into sentence-sized chunks and scored individually, "
        "then combined into one length-weighted verdict — the model was trained almost entirely "
        "on short 1-2 sentence snippets, so this keeps every prediction closer to what it actually knows."
    )

    # --- Overall breakdown (always shows ALL results, unfiltered) ---
    st.markdown("### Overall sentiment breakdown")
    class_counts = results_df["sentiment"].value_counts().to_dict()

    chart_col, stat_col = st.columns([2, 1])
    with chart_col:
        dark = st.session_state.get("dark_mode", True)
        st.plotly_chart(sentiment_pie_chart(class_counts, dark=dark), width="stretch")
    with stat_col:
        total = len(results_df)
        for label in ["positive", "negative", "neutral"]:
            pct = (class_counts.get(label, 0) / total * 100) if total else 0
            st.markdown(
                f"""<div class="stat-tile"><div class="stat-value" style="color:var(--{label})">
                {pct:.1f}%</div><div class="stat-label">{label.capitalize()}</div></div>""",
                unsafe_allow_html=True,
            )

    st.write("")

    # --- Word cloud of the most common words across everything scraped ---
    st.markdown("### Word Cloud")
    st.caption("Most frequent words across every article for this search (bigger = more common).")

    # Words that are structurally common in this pipeline but carry no
    # topical meaning -- excluded so the cloud highlights real content
    # instead of masked names or bare negation words.
    _WORDCLOUD_EXCLUDE = {
        "person", "not", "no", "nor", "never", "but", "however",
        "although", "though", "yet", "still",
    }
    word_counter = Counter()
    for cleaned in results_df["cleaned_text"]:
        for w in str(cleaned).split():
            if len(w) > 2 and w not in _WORDCLOUD_EXCLUDE:
                word_counter[w] += 1

    top_words = word_counter.most_common(40)
    if top_words:
        dark = st.session_state.get("dark_mode", True)
        cloud_fig = word_cloud_chart(top_words, dark=dark)
        st.plotly_chart(cloud_fig, width="stretch")
    else:
        st.caption("Not enough distinct words to build a word cloud yet.")

    st.write("")

    # --- Per-article detail, filterable by sentiment label ---
    st.markdown("### Individual results")

    result_label_filter = st.selectbox(
        "Filter by sentiment",
        options=["All", "Positive", "Negative", "Neutral"],
        index=0,
        key="live_search_label_filter",
    )

    display_df = results_df
    if result_label_filter != "All":
        display_df = display_df[display_df["sentiment"] == result_label_filter.lower()]

    if display_df.empty:
        st.caption("No results match this filter.")
    else:
        st.caption(f"Showing {len(display_df)} of {len(results_df)} results.")
        st.dataframe(
            display_df[["type", "text", "sentiment", "confidence", "chunks_analyzed", "source", "post_title"]],
            width="stretch",
            hide_index=True,
        )

    st.download_button(
        "⬇ Download results as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"news_search_{stored_query.replace(' ', '_')}.csv",
        mime="text/csv",
    )

render_footer()
