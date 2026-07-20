"""
utils/auth.py
---------------
Local username/password authentication for the app.

Users are stored in users.json (BASE_DIR/users.json) as:
    { "username": {"salt": "...", "hash": "...", "created": "..."} }

Passwords are never stored in plain text -- PBKDF2-HMAC-SHA256 with a
per-user random salt is used for hashing.
"""

import json
import hashlib
import secrets
from datetime import datetime

import streamlit as st

from config import USERS_PATH

PBKDF2_ITERATIONS = 200_000


def _load_users() -> dict:
    if USERS_PATH.exists():
        try:
            return json.loads(USERS_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_users(users: dict):
    USERS_PATH.write_text(json.dumps(users, indent=2))


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()


def username_exists(username: str) -> bool:
    return username.strip().lower() in _load_users()


def register_user(username: str, password: str) -> tuple[bool, str]:
    """Creates a new account. Returns (success, message)."""
    username = username.strip().lower()

    if not username or not password:
        return False, "Username and password are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if username in users:
        return False, "That username is already taken."

    salt = secrets.token_hex(16)
    users[username] = {
        "salt": salt,
        "hash": _hash_password(password, salt),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_users(users)
    return True, "Account created successfully."


def verify_user(username: str, password: str) -> tuple[bool, str]:
    """Checks credentials. Returns (success, message)."""
    username = username.strip().lower()
    users = _load_users()

    record = users.get(username)
    if not record:
        return False, "Invalid username or password."

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
