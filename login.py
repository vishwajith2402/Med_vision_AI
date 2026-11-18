from __future__ import annotations

import hashlib
import hmac

import streamlit as st


USERS = {
    "admin": "admin123",
    "doctor": "medai123",
}


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


PASSWORD_HASHES = {username: _hash_password(password) for username, password in USERS.items()}


def authenticate(username: str, password: str) -> bool:
    expected_hash = PASSWORD_HASHES.get(username)
    if expected_hash is None:
        return False
    return hmac.compare_digest(expected_hash, _hash_password(password))


def require_login() -> bool:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if st.session_state.logged_in:
        with st.sidebar:
            st.success(f"Logged in as {st.session_state.username}")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.rerun()
        return True

    st.markdown(
        """
        <div class="med-hero">
            <div class="med-hero-grid">
                <div>
                    <div class="med-kicker">Protected clinical workspace</div>
                    <h1 class="med-title">MedAI Vision</h1>
                    <div class="med-subtitle">
                        Sign in to access medical image screening and prediction tools.
                    </div>
                </div>
                <div class="medical-signal">
                    <div class="medical-cross" aria-label="Medical cross"></div>
                    <div class="ecg-card">
                        <div class="ecg-label">Patient Signal</div>
                        <svg class="ecg-svg" viewBox="0 0 260 70" role="img" aria-label="Animated ECG line">
                            <line class="ecg-grid-line" x1="0" y1="18" x2="260" y2="18"></line>
                            <line class="ecg-grid-line" x1="0" y1="36" x2="260" y2="36"></line>
                            <line class="ecg-grid-line" x1="0" y1="54" x2="260" y2="54"></line>
                            <path class="ecg-path" d="M0 38 H38 L48 38 L58 18 L70 58 L84 34 L104 34 L116 38 H146 L158 26 L170 46 L182 34 H214 L226 38 H260"></path>
                            <circle class="ecg-dot" cx="70" cy="58" r="4"></circle>
                        </svg>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        st.markdown("### Secure Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if authenticate(username.strip(), password):
            st.session_state.logged_in = True
            st.session_state.username = username.strip()
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.info("Demo users: admin/admin123 or doctor/medai123")
    return False
