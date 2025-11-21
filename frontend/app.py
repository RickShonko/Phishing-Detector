"""Streamlit UI that mirrors the existing HTML/CSS/JS frontend."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Dict, Optional

import requests
import streamlit as st

API_URL = os.getenv("PHISHING_API_URL", "http://localhost:8000/analyze-message")
PAGE_TITLE = "Kenyan Phishing Detector"
PAGE_ICON = "🛡️"
CSS_PATH = Path(__file__).resolve().parents[1] / "style.css"

st.set_page_config(page_title=f"{PAGE_TITLE} — Secure Scan", page_icon=PAGE_ICON, layout="wide")


def load_css() -> None:
    """Inject the original CSS so the Streamlit app keeps the same look."""
    fallback = """body{background:#0b0f12;color:#dff7ee;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}"""
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else fallback
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_badge_class(label: str) -> str:
    text = label.lower()
    if "phishing" in text or "high" in text:
        return "red"
    if "suspicious" in text or "medium" in text:
        return "orange"
    if "legitimate" in text or "safe" in text:
        return "green"
    return "neutral"


def call_backend(message: str, sender: Optional[str]) -> Dict:
    payload = {"message": message}
    if sender:
        payload["sender"] = sender

    response = requests.post(API_URL, json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def sanitize(value: Optional[str]) -> str:
    return html.escape(value or "")


load_css()

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
    st.session_state.badge_label = "No analysis yet"

st.markdown(
    f"""
    <div class="app-shell">
      <header class="app-header">
        <div class="logo">{PAGE_ICON} Kenyan Phishing Detector</div>
        <div class="sub">Cybersecurity demo • Dark scanner theme</div>
      </header>
    </div>
    """,
    unsafe_allow_html=True,
)

main_container = st.container()
analysis_error = None

with main_container:
    left_col, right_col = st.columns([1.15, 0.85])

    with left_col:
        st.markdown("<section class='input-card'>", unsafe_allow_html=True)
        with st.form("analysis_form", clear_on_submit=False):
            st.markdown(
                "<label class='label' for='messageInput'>Paste SMS / WhatsApp / Email message</label>",
                unsafe_allow_html=True,
            )
            message = st.text_area(
                label="messageInput",
                label_visibility="collapsed",
                placeholder="Paste a suspicious message here...",
                height=210,
            )

            st.markdown("<div class='sender-selection'>", unsafe_allow_html=True)
            st.markdown("<label class='label'>Select Sender</label>", unsafe_allow_html=True)
            sender_option = st.selectbox(
                label="senderSelect",
                label_visibility="collapsed",
                options=["", "KCB", "Equity", "Cooperative Bank", "MPesa", "Other"],
                format_func=lambda x: "-- Select Sender --" if x == "" else x,
            )

            sender_other: Optional[str] = None
            if sender_option == "Other":
                sender_other = st.text_input(
                    label="senderOther",
                    label_visibility="collapsed",
                    placeholder="If Other, type here...",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button("🛡️ Analyze Message")

        st.markdown("</section>", unsafe_allow_html=True)

        if submitted:
            sender_value = (sender_other or "").strip() if sender_option == "Other" else sender_option.strip()
            sender_value = sender_value or None

            if not message.strip():
                analysis_error = "Please enter a message to analyze."
            else:
                with st.spinner("Analyzing..."):
                    try:
                        result = call_backend(message.strip(), sender_value)
                        st.session_state.analysis_result = result
                        st.session_state.badge_label = result.get("classification", "Result ready")
                    except requests.RequestException as exc:
                        st.session_state.analysis_result = None
                        st.session_state.badge_label = "Error"
                        analysis_error = f"Error calling the API: {exc}".strip()

    result = st.session_state.analysis_result
    badge_label = result.get("classification", st.session_state.badge_label) if result else st.session_state.badge_label
    badge_class = get_badge_class(badge_label)
    st.session_state.badge_label = badge_label

    with right_col:
        st.markdown("<section class='results-card'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='result-header'>
              <div>Result</div>
              <div id='scoreBadge' class='badge {badge_class}'>{sanitize(badge_label)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result:
            explanation = sanitize(result.get("explanation", "No explanation provided."))
            action = sanitize(result.get("recommended_action", "No action provided."))
            risk_score = result.get("risk_score")
            risk_value = "N/A"
            if isinstance(risk_score, (int, float)):
                risk_value = f"{risk_score:.2f}"
            elif risk_score is not None:
                risk_value = sanitize(str(risk_score))

            body_html = f"""
            <div class='result-body'>
              <div class='result-block'>
                <h3>Explanation</h3>
                <div class='result-text'>{explanation}</div>
              </div>
              <div class='result-block'>
                <h3>Recommended action</h3>
                <div class='result-text'>{action}</div>
              </div>
              <div class='result-block'>
                <h3>Risk score</h3>
                <div class='result-text'>{risk_value}</div>
              </div>
            </div>
            """
        else:
            body_html = """
            <div class='result-body'>
              <div class='placeholder'>Paste a message and click <strong>Analyze</strong> to start.</div>
            </div>
            """

        st.markdown(body_html, unsafe_allow_html=True)
        st.markdown(
            """
            <div class='meta-row'>
              <div class='meta'>Endpoint: <code>/analyze-message</code></div>
              <div class='meta'>Demo Theme: Cybersecurity (Dark)</div>
            </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

if analysis_error:
    st.warning(analysis_error)

st.markdown(
    """
    <footer class='app-footer'>
      <div>Built for hackathon demo — understands English / Kiswahili / Sheng</div>
      <div class='credits'>Primenova Technologies • MIT License</div>
    </footer>
    """,
    unsafe_allow_html=True,
)
