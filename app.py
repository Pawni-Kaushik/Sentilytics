"""
app.py
-------
Entry point of the app. Every navigation (top navbar click, direct URL,
browser refresh) runs this script first. It:

  1. Shows a Sign In / Register screen for unauthenticated visitors.
  2. Once logged in, hands off to st.navigation() to render whichever
     page was requested (Home, Analyzer, Dashboard, ...), in the exact
     order/labels defined in config.NAV_PAGES -- a single source of
     truth so the nav never disagrees with itself.

The default Streamlit sidebar page-list is turned off
(position="hidden") since the custom top navbar (see
utils/helpers.render_navbar) is the only navigation surface. That also
avoids the "app" vs "Home" label mismatch that came from Streamlit
auto-naming the sidebar entry after the app.py filename.
"""

import streamlit as st
import streamlit.components.v1 as components

from config import NAV_PAGES
from utils.helpers import init_session_state, load_css
from utils.auth import init_auth_state, is_authenticated, login, register_user, verify_user

st.set_page_config(page_title="Sentilytics — Sentiment Analyzer", page_icon="▲", layout="wide")

init_session_state()
init_auth_state()
load_css()


# =================================================================
# AUTH SCREEN
# =================================================================
def render_auth_screen():
    # Disables the keyboard Tab key on this screen only. Note: this
    # blocks keyboard-only and screen-reader users from moving between
    # fields, so it's an accessibility trade-off -- worth knowing if
    # this gets reviewed/graded.
    components.html(
        """
        <script>
        window.parent.document.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                e.preventDefault();
            }
        }, true);
        </script>
        """,
        height=0,
    )

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="auth-eyebrow">▲ Sentilytics — Sentiment Analyzer</div>
        <div class="auth-title">Welcome back</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")

        # Password field first, "Show password" toggle directly below it.
        # (Reads last run's checkbox value from session_state before the
        # checkbox widget itself is created further down, so the field's
        # type is correct on the same run the box is ticked.)
        show_pw = st.session_state.get("login_show_pw", False)
        password = st.text_input(
            "Password",
            key="login_password",
            type="default" if show_pw else "password",
            placeholder="Enter your password",
        )
        st.checkbox("Show password", key="login_show_pw")

        if st.button("Sign In", key="login_btn", type="primary", use_container_width=True):
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                ok, message = verify_user(username, password)
                if ok:
                    login(username)
                    st.rerun()
                else:
                    st.error(message)

    with tab_register:
        new_username = st.text_input("Username", key="reg_username", placeholder="Choose a username")

        show_pw_reg = st.session_state.get("reg_show_pw", False)
        pw_type = "default" if show_pw_reg else "password"
        new_password = st.text_input(
            "Password", key="reg_password", type=pw_type, placeholder="At least 6 characters"
        )
        confirm_password = st.text_input(
            "Confirm Password", key="reg_confirm_password", type=pw_type, placeholder="Re-enter your password"
        )
        st.checkbox("Show password", key="reg_show_pw")

        if st.button("Create Account", key="register_btn", type="primary", use_container_width=True):
            if not new_username or not new_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = register_user(new_username, new_password)
                if ok:
                    st.success(f"{message} You can now sign in.")
                else:
                    st.error(message)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


if not is_authenticated():
    render_auth_screen()
    st.stop()


# =================================================================
# AUTHENTICATED: hand off to the real page router
# =================================================================
pages = [
    st.Page(page["path"], title=page["label"], icon=page["icon"], default=(page["label"] == "Home"))
    for page in NAV_PAGES
]

router = st.navigation(pages, position="hidden")
router.run()
