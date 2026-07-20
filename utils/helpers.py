"""
utils/helpers.py
------------------
Shared UI plumbing used by every page: theme CSS injection, the
custom top navbar, prediction-history session state, and the footer.

Keeping this in one module means every page renders the same nav/
footer/theme consistently instead of copy-pasting markup five times.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from config import CSS_PATH, COLORS, NAV_PAGES, PROJECT_TITLE, DEVELOPERS, HISTORY_PATH


def load_history_from_disk():
    """Reads saved prediction history from disk, if it exists."""
    if HISTORY_PATH.exists():
        try:
            df = pd.read_csv(HISTORY_PATH)
            return df.to_dict("records")
        except Exception:
            return []
    return []


def save_history_to_disk():
    """
    Writes the current session's history to disk (chronological order --
    oldest first) so it survives a reload. Kept separate from
    get_history_df(), which reverses order only for DISPLAY purposes --
    saving the reversed view here would flip the order every time the
    app restarts.
    """
    if not st.session_state.prediction_history:
        df = pd.DataFrame(columns=["timestamp", "text", "sentiment", "confidence"])
    else:
        df = pd.DataFrame(st.session_state.prediction_history)
        if "timestamp" not in df.columns:
            df["timestamp"] = ""
        df = df[["timestamp", "text", "sentiment", "confidence"]]
    df.to_csv(HISTORY_PATH, index=False)


def init_session_state():
    """Sets up session_state defaults used across pages."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = load_history_from_disk()


def load_css():
    """
    Reads the stylesheet template and fills in the color tokens for
    whichever mode (dark/light) is currently active in session_state.
    """
    dark = st.session_state.get("dark_mode", True)

    if dark:
        tokens = {
            "__BG__": COLORS["dark_bg"],
            "__SURFACE__": COLORS["dark_surface"],
            "__SURFACE_ALT__": COLORS["dark_surface_alt"],
            "__TEXT__": COLORS["dark_text"],
            "__TEXT_MUTED__": COLORS["dark_text_muted"],
            "__BORDER__": COLORS["dark_border"],
        }
    else:
        tokens = {
            "__BG__": COLORS["light_bg"],
            "__SURFACE__": COLORS["light_surface"],
            "__SURFACE_ALT__": COLORS["light_surface_alt"],
            "__TEXT__": COLORS["light_text"],
            "__TEXT_MUTED__": COLORS["light_text_muted"],
            "__BORDER__": COLORS["light_border"],
        }

    css = CSS_PATH.read_text()
    for placeholder, value in tokens.items():
        css = css.replace(placeholder, value)

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_navbar(active: str):
    """
    Renders the sticky top navbar with links to every page.

    IMPORTANT: uses st.page_link (real Streamlit navigation) instead of raw
    <a href> tags. Raw anchor tags force a full browser page reload, which
    tears down the Streamlit session and wipes st.session_state -- that was
    why the login screen kept reappearing when moving between pages. Real
    navigation happens client-side and keeps the session (and login) intact.
    """
    from utils.auth import is_authenticated, logout as auth_logout

    st.markdown('<div class="topnav-brand-row">', unsafe_allow_html=True)
    brand_col, theme_col, logout_col = st.columns([6, 1.3, 1.1])
    with brand_col:
        badge = f" &middot; 👤 {st.session_state.username}" if is_authenticated() else ""
        st.markdown(
            f'<div class="brand"><span class="arrow">▲</span> Sentilytics'
            f'<span class="user-badge">{badge}</span></div>',
            unsafe_allow_html=True,
        )
    with theme_col:
        mode_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
        if st.button(mode_label, key=f"theme_toggle_{active}", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    with logout_col:
        if is_authenticated():
            if st.button("Logout", key=f"logout_{active}", use_container_width=True):
                auth_logout()
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    nav_cols = st.columns(len(NAV_PAGES))
    for col, page in zip(nav_cols, NAV_PAGES):
        with col:
            st.page_link(page["path"], label=page["label"], icon=page["icon"])

    st.markdown('<div class="topnav-divider"></div>', unsafe_allow_html=True)


def render_footer():
    st.markdown(
        f"""
        <div class="app-footer">
            {PROJECT_TITLE}<br>
            Built by {DEVELOPERS[0]} &amp; {DEVELOPERS[1]} &middot; Internship Project 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_to_history(original_text: str, result: dict):
    """Appends a prediction result to the session's history log, and saves it to disk."""
    st.session_state.prediction_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": original_text,
        "sentiment": result["prediction"],
        "confidence": round(result["confidence"], 2),
    })
    save_history_to_disk()


def get_history_df() -> pd.DataFrame:
    """
    Returns the history as a DataFrame, newest predictions first.
    Handles old history files saved before the "timestamp" column existed,
    by filling in a blank timestamp for those older rows instead of crashing.
    """
    if not st.session_state.prediction_history:
        return pd.DataFrame(columns=["timestamp", "text", "sentiment", "confidence"])

    df = pd.DataFrame(st.session_state.prediction_history)
    if "timestamp" not in df.columns:
        df["timestamp"] = ""

    df = df[["timestamp", "text", "sentiment", "confidence"]]
    return df.iloc[::-1].reset_index(drop=True)  # newest first


def search_history(history_df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """
    Filters the history DataFrame to rows whose text contains the given
    keyword (case-insensitive, partial match). Used by the "filter by
    word" feature on the Analyzer page.
    """
    if not keyword.strip():
        return history_df
    return history_df[history_df["text"].str.contains(keyword, case=False, na=False)]


def clear_history():
    st.session_state.prediction_history = []
    save_history_to_disk()
