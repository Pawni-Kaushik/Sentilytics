"""
utils/auth.py
---------------
Username/password authentication for the app.

Users are stored in a hosted Postgres database (see DEPLOYMENT.md for
setup), NOT a local file. Streamlit Community Cloud's filesystem is
ephemeral -- any file written locally (like the old users.json) gets
wiped whenever the app container sleeps and wakes back up, restarts,
or is redeployed. A real database survives all of that, which a local
JSON file (or several of them) never would.

Passwords are never stored in plain text -- PBKDF2-HMAC-SHA256 with a
per-user random salt is used for hashing.
"""

import hashlib
import secrets
from datetime import datetime

import streamlit as st
from sqlalchemy import text

PBKDF2_ITERATIONS = 200_000


def _get_conn():
    """Returns a Streamlit SQL connection, creating the users table
    on first use if it doesn't exist yet."""
    conn = st.connection("sql", type="sql")
    with conn.session as s:
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                salt TEXT NOT NULL,
                hash TEXT NOT NULL,
                created TEXT NOT NULL
            )
        """))
        s.commit()
    return conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()


def username_exists(username: str) -> bool:
    conn = _get_conn()
    df = conn.query(
        "SELECT 1 FROM users WHERE username = :u",
        params={"u": username.strip().lower()},
        ttl=0,
    )
    return not df.empty


def register_user(username: str, password: str) -> tuple[bool, str]:
    """Creates a new account. Returns (success, message)."""
    username = username.strip().lower()

    if not username or not password:
        return False, "Username and password are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    conn = _get_conn()

    if username_exists(username):
        return False, "That username is already taken."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with conn.session as s:
        s.execute(
            text(
                "INSERT INTO users (username, salt, hash, created) "
                "VALUES (:u, :s, :h, :c)"
            ),
            {"u": username, "s": salt, "h": password_hash, "c": created},
        )
        s.commit()

    return True, "Account created successfully."


def verify_user(username: str, password: str) -> tuple[bool, str]:
    """Checks credentials. Returns (success, message)."""
    username = username.strip().lower()
    conn = _get_conn()

    df = conn.query(
        "SELECT salt, hash FROM users WHERE username = :u",
        params={"u": username},
        ttl=0,
    )
    if df.empty:
        return False, "Invalid username or password."

    record = df.iloc[0]
    if _hash_password(password, record["salt"]) != record["hash"]:
        return False, "Invalid username or password."

    return True, "Login successful."


def init_auth_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None


def is_authenticated() -> bool:
    init_auth_state()
    return st.session_state.authenticated


def login(username: str):
    st.session_state.authenticated = True
    st.session_state.username = username.strip().lower()


def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
