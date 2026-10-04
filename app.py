"""
app.py
Streamlit app for Group 8 - Customer Complaint Topic Modelling.

Run from PyCharm's TERMINAL (not the green Run button):
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
import sys
import os
import io
import re
import json
import html
import hashlib
import tempfile
from datetime import datetime
from fpdf import FPDF, XPos, YPos
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess_text
from topic_map import build_topic_map_html
from distribution import topic_counts, summarize
import speech

st.set_page_config(
    page_title="Customer Complaint Topic Explorer",
    page_icon="🗂️",
    layout="wide",
)

MODEL_OPTIONS = ["NMF", "LDA"]  # NMF first: it won on coherence and diversity
MODEL_CAPTIONS = {"NMF": "NMF (best model)", "LDA": "LDA"}
LABELS_PATH = "data/topic_labels.json"


def load_label_file():
    """Reads data/topic_labels.json into ({model: {idx: name}}, {model: {idx: style}}).
    A missing or broken file falls back to empty dicts, so topics show as 'Topic N'."""
    try:
        with open(LABELS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}
    names, styles = {}, {}
    for model_key in MODEL_OPTIONS:
        entries = raw.get(model_key, {}) or {}
        names[model_key] = {int(k): v.get("name", f"Topic {k}") for k, v in entries.items()}
        styles[model_key] = {int(k): v["style"] for k, v in entries.items() if v.get("style")}
    return names, styles


FILE_LABELS, TOPIC_STYLES = load_label_file()

# Session copy of the names, so edits in the "Name your topics" panel apply
# everywhere during a visit. Seeded from the saved file on first load.
if not isinstance(st.session_state.get("topic_labels"), dict) or \
        not all(m in st.session_state["topic_labels"] for m in MODEL_OPTIONS):
    st.session_state["topic_labels"] = {m: dict(FILE_LABELS[m]) for m in MODEL_OPTIONS}
st.session_state.setdefault("app_theme", "Day")

# --- Fixed brand palette: used only by exported PDF/PPTX reports, which always
# render in the standard brand look regardless of the viewer's in-app theme choice. ---
NAVY = "#1b3865"
SKY = "#45c8f3"
ORANGE = "#f47a21"
LIGHT_GRAY = "#f4f6f9"
GRAY = "#5e5e5f"

TOPIC_COLORS = [
    NAVY, ORANGE, SKY,
    "#5c7ba3", "#f4a35c", "#8fdcf7",
    "#0f2540", "#c25e14", "#2f95b8",
    "#a4b8cd", GRAY,
]
TOPIC_COLORMAPS = [
    LinearSegmentedColormap.from_list("t", ["#FFFFFF", c]) for c in TOPIC_COLORS
]

# ------------------------------------------------------------------------------
# Colour contrast helpers (WCAG 2.1). Used to guarantee readable text: any
# colour used for text is nudged darker or lighter until it clears the minimum
# contrast ratio against the background it sits on.
# ------------------------------------------------------------------------------
def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _luminance(hex_color):
    def channel(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(v) for v in _rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a, b):
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _mix(a, b, amount):
    ra, rb = _rgb(a), _rgb(b)
    return "#{:02x}{:02x}{:02x}".format(*(round(x + (y - x) * amount) for x, y in zip(ra, rb)))


def readable_on(color, background, min_ratio=4.5):
    """Returns `color`, darkened (on light backgrounds) or lightened (on dark ones)
    just enough to reach `min_ratio` contrast against `background`."""
    target = "#000000" if _luminance(background) > 0.18 else "#ffffff"
    step = 0.0
    candidate = color
    while contrast_ratio(candidate, background) < min_ratio and step < 1.0:
        step += 0.04
        candidate = _mix(color, target, step)
    return candidate


# ------------------------------------------------------------------------------
# Themes. Day = "Clear Sky" (cool off-white, white cards, deep navy text).
# Night = "Midnight" (deep navy surfaces, soft white text, sky-blue accent).
# Both built from the brand palette: navy #1b3865, sky #45c8f3, orange #f47a21.
# Only affects the live UI; PDF/PPTX exports always use the fixed brand look.
# ------------------------------------------------------------------------------
THEMES = {
    "Day": {
        "app_bg": "#f4f7fb",
        "sidebar_bg": "#ffffff",
        "card_bg": "#ffffff",
        "card_border": "#dbe3ee",
        "input_bg": "#ffffff",
        "input_border": "#c3cfdf",
        "code_bg": "#eef2f8",
        "text": "#14213d",
        "text_secondary": "#4a5870",
        "header_grad": "linear-gradient(120deg, #14213d 0%, #1b3865 50%, #1f5f94 100%)",
        "brand_icon_grad": "linear-gradient(135deg, #1b3865 0%, #2a7ab8 100%)",
        "accent": "#1b3865",          # selected nav, selected tab, primary buttons
        "accent_text": "#ffffff",
        "accent_soft": "#e8eef7",     # hover states
        "highlight": "#f47a21",       # decorative borders only, never text
        "wc_bg": "#ffffff",
        "grid_color": "#e3e9f2",
        "topic_colors": [
            "#1b3865", "#f47a21", "#2a9fd6", "#5c7ba3", "#e0892f",
            "#45c8f3", "#0f2540", "#c25e14", "#2f95b8", "#8aa1bd", "#5e5e5f",
        ],
    },
    "Night": {
        "app_bg": "#0b1626",
        "sidebar_bg": "#0f1d31",
        "card_bg": "#132640",
        "card_border": "#25406a",
        "input_bg": "#0f2036",
        "input_border": "#3a5782",
        "code_bg": "#0f2036",
        "text": "#eef3fa",
        "text_secondary": "#b9c7da",
        "header_grad": "linear-gradient(120deg, #0b1626 0%, #1b3865 55%, #1f5f94 100%)",
        "brand_icon_grad": "linear-gradient(135deg, #1b3865 0%, #45c8f3 100%)",
        "accent": "#45c8f3",
        "accent_text": "#0b1626",
        "accent_soft": "#1a3150",
        "highlight": "#f47a21",
        "wc_bg": "#132640",
        "grid_color": "#25406a",
        "topic_colors": [
            "#45c8f3", "#f47a21", "#8fdcf7", "#f4a35c", "#7fa6d6",
            "#ffc38a", "#5fb4d4", "#ffd1a3", "#a9c9e8", "#e8894a", "#dbe4f0",
        ],
    },
}

# --- Theme toggle: a single icon button showing the mode you'll switch TO.
# Rendered first so its value is known before the CSS below is built. ---
with st.sidebar:
    toggle_cols = st.columns([4, 1])
    with toggle_cols[1]:
        target_icon = "🌙" if st.session_state["app_theme"] == "Day" else "☀️"
        if st.button(target_icon, key="theme_toggle_btn", help="Switch day/night theme"):
            st.session_state["app_theme"] = (
                "Night" if st.session_state["app_theme"] == "Day" else "Day"
            )
            st.rerun()

selected_theme = st.session_state["app_theme"]
THEME = THEMES[selected_theme]
UI_TOPIC_COLORS = THEME["topic_colors"]                     # fills: bars, word clouds
UI_TOPIC_TEXT = [readable_on(c, THEME["app_bg"]) for c in UI_TOPIC_COLORS]  # topic names as text
T = THEME  # short alias for the stylesheet below

# --- Global CSS -----------------------------------------------------------------
# Every rule that sets a text colour also sets (or relies on) a background from the
# same theme, so light-on-light or dark-on-dark can't happen. Streamlit's own
# widgets (buttons, inputs, dropdowns, code, expanders, uploader, toolbar) are
# styled explicitly, because they otherwise keep light-mode backgrounds in Night.
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    /* ---------- Base surfaces and text ---------- */
    .stApp {{ background-color: {T["app_bg"]}; color: {T["text"]}; }}
    .stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{ color: {T["text"]}; }}
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] strong,
    .stApp [data-testid="stWidgetLabel"] p {{ color: {T["text"]} !important; }}
    h2, h3, h4 {{ font-weight: 700; }}
    .stApp hr {{ border-color: {T["card_border"]} !important; }}

    /* Streamlit draws captions at reduced opacity; restore full strength */
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] * {{
        color: {T["text_secondary"]} !important; opacity: 1 !important;
    }}

    /* Top toolbar (Deploy / menu) */
    .stApp header[data-testid="stHeader"] {{ background: {T["app_bg"]} !important; }}
    .stApp header[data-testid="stHeader"] button,
    .stApp header[data-testid="stHeader"] button * {{
        color: {T["text"]} !important; background-color: transparent !important;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: {T["sidebar_bg"]};
        border-right: 1px solid {T["card_border"]};
    }}
    section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {{ color: {T["text"]} !important; }}
    .sidebar-brand {{ display: flex; align-items: center; gap: 0.65rem; margin: 0.25rem 0 1.5rem 0; }}
    .sidebar-brand-icon {{
        width: 42px; height: 42px; border-radius: 11px; background: {T["brand_icon_grad"]};
        display: flex; align-items: center; justify-content: center; font-size: 1.35rem;
        flex-shrink: 0; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    }}
    .sidebar-brand-text {{ font-weight: 700; font-size: 1rem; line-height: 1.25; color: {T["text"]}; }}
    .sidebar-brand-text small {{
        display: block; font-weight: 500; font-size: 0.72rem; color: {T["text_secondary"]} !important;
        letter-spacing: 0.03em; text-transform: uppercase;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0.25rem; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding: 0.6rem 0.85rem; border-radius: 8px; width: 100%; transition: background-color 0.15s ease;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
        font-size: 1.02rem; color: {T["text"]} !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background-color: {T["accent_soft"]}; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{ background-color: {T["accent"]}; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
        color: {T["accent_text"]} !important; font-weight: 600;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] svg {{ display: none; }}
    /* Hide the round radio dot; :not(:has(p)) guarantees a page name is never hidden */
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child:not([data-testid]):not(:has(p)) {{
        display: none;
    }}

    /* ---------- Page banner (always white text on a dark gradient) ---------- */
    .app-header {{
        padding: 1.75rem 2rem; border-radius: 14px; background: {T["header_grad"]};
        margin-bottom: 1.5rem; box-shadow: 0 6px 18px rgba(20, 33, 61, 0.18);
    }}
    .stApp .app-header, .stApp .app-header * {{ color: #FFFFFF !important; }}
    .app-header-title {{ font-size: 1.9rem; font-weight: 700; line-height: 1.2; margin: 0; }}
    .app-header-subtitle {{ font-size: 0.98rem; opacity: 0.95; margin-top: 0.35rem; }}

    /* ---------- Buttons ---------- */
    .stApp button[data-testid^="stBaseButton-secondary"],
    .stApp [data-testid="stDownloadButton"] button,
    .stApp [data-testid="stFileUploaderDropzone"] button {{
        background-color: {T["card_bg"]} !important; color: {T["text"]} !important;
        border: 1px solid {T["input_border"]} !important; border-radius: 8px !important;
    }}
    .stApp button[data-testid^="stBaseButton-secondary"] *,
    .stApp [data-testid="stDownloadButton"] button *,
    .stApp [data-testid="stFileUploaderDropzone"] button * {{ color: {T["text"]} !important; }}
    .stApp button[data-testid^="stBaseButton-secondary"]:hover,
    .stApp [data-testid="stDownloadButton"] button:hover,
    .stApp [data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: {T["accent_soft"]} !important; border-color: {T["accent"]} !important;
    }}
    .stApp button[data-testid^="stBaseButton-primary"] {{
        background-color: {T["accent"]} !important; border: 1px solid {T["accent"]} !important;
        color: {T["accent_text"]} !important; border-radius: 8px !important;
    }}
    .stApp button[data-testid^="stBaseButton-primary"] * {{ color: {T["accent_text"]} !important; }}

    /* ---------- Inputs, text areas, dropdowns ---------- */
    .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
    .stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div {{
        background-color: {T["input_bg"]} !important; border-color: {T["input_border"]} !important;
    }}
    .stApp input, .stApp textarea {{
        background-color: {T["input_bg"]} !important; color: {T["text"]} !important;
        -webkit-text-fill-color: {T["text"]} !important; caret-color: {T["text"]};
    }}
    .stApp input::placeholder, .stApp textarea::placeholder {{
        color: {T["text_secondary"]} !important; -webkit-text-fill-color: {T["text_secondary"]} !important; opacity: 1;
    }}
    .stApp [data-baseweb="select"] *, .stApp [data-baseweb="select"] svg {{
        color: {T["text"]} !important; fill: {T["text"]} !important;
    }}
    /* Streamlit 1.62 widget wrappers (these carry their own white backgrounds) */
    .stApp [data-testid="stSelectbox"] div[role="group"],
    .stApp [data-testid="stTextAreaRootElement"],
    .stApp [data-testid="stTextInputRootElement"] {{
        background-color: {T["input_bg"]} !important;
        border: 1px solid {T["input_border"]} !important; border-radius: 8px !important;
    }}
    .stApp [data-testid="stSelectbox"] div[role="group"] input {{ border: none !important; }}
    .stApp [data-testid="stSelectbox"] button, .stApp [data-testid="stSelectbox"] svg {{
        color: {T["text"]} !important; fill: {T["text"]} !important; background: transparent !important;
    }}
    /* Progress bars: themed track and fill */
    .stApp [data-testid="stProgressBarTrack"] {{ background-color: {T["card_border"]} !important; }}
    .stApp [data-testid="stProgressBarTrack"] > div {{ background-color: {T["accent"]} !important; }}
    /* Dropdown option lists render outside .stApp, in a popover layer */
    [data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"] {{
        background-color: {T["input_bg"]} !important;
    }}
    [data-baseweb="popover"] li {{ background-color: {T["input_bg"]} !important; }}
    [data-baseweb="popover"] li, [data-baseweb="popover"] li * {{ color: {T["text"]} !important; }}
    [data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {{
        background-color: {T["accent_soft"]} !important;
    }}

    /* Open dropdown lists (Streamlit 1.62 renders these as role=listbox/option) */
    [role="listbox"] {{
        background-color: {T["input_bg"]} !important; border: 1px solid {T["input_border"]} !important;
    }}
    [role="listbox"] [role="option"], [role="listbox"] [role="option"] * {{ color: {T["text"]} !important; }}
    [role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][aria-selected="true"],
    [role="listbox"] [role="option"][data-focused] {{ background-color: {T["accent_soft"]} !important; }}

    /* ---------- Code, expanders, uploader, tables, alerts ---------- */
    .stApp [data-testid="stCode"] pre, .stApp [data-testid="stCode"] code {{
        background-color: {T["code_bg"]} !important;
    }}
    .stApp [data-testid="stCode"] pre *, .stApp [data-testid="stCode"] code {{ color: {T["text"]} !important; }}
    .stApp [data-testid="stExpander"] details {{
        background-color: {T["card_bg"]}; border: 1px solid {T["card_border"]} !important; border-radius: 10px;
    }}
    .stApp [data-testid="stExpander"] summary {{ background-color: {T["card_bg"]} !important; border-radius: 10px; }}
    .stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * {{
        color: {T["text"]} !important;
    }}
    .stApp [data-testid="stIconMaterial"] {{ color: {T["text_secondary"]} !important; }}
    .stApp [data-testid="stFileUploaderDropzone"] {{
        background-color: {T["input_bg"]} !important; border: 1px dashed {T["input_border"]} !important;
    }}
    .stApp [data-testid="stFileUploaderDropzone"] span, .stApp [data-testid="stFileUploaderDropzone"] small,
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] * {{ color: {T["text_secondary"]} !important; }}
    .stApp [data-testid="stFileUploaderFileName"], .stApp [data-testid="stFileUploaderFile"] * {{
        color: {T["text"]} !important;
    }}
    .stApp [data-testid="stDataFrame"] {{ border: 1px solid {T["card_border"]}; border-radius: 8px; }}
    .stApp [data-testid="stTable"] table, .stApp [data-testid="stTable"] th, .stApp [data-testid="stTable"] td {{
        color: {T["text"]} !important; background-color: {T["card_bg"]} !important;
        border-color: {T["card_border"]} !important;
    }}
    /* Alert boxes keep Streamlit's tinted background, which follows the page
       (light tint by day, dark tint by night), so their text follows the theme too */
    .stApp [data-testid="stAlertContainer"] p, .stApp [data-testid="stAlertContainer"] li,
    .stApp [data-testid="stAlertContainer"] strong, .stApp [data-testid="stAlertContainer"] span {{
        color: {T["text"]} !important;
    }}

    /* Recorder and uploaded-file chips keep their own light panels in both themes,
       so the text and icons inside them stay dark (checked against WCAG 4.5:1) */
    .stApp [data-testid="stAudioInput"] > div {{
        background-color: #ffffff !important; border: 1px solid {T["input_border"]}; border-radius: 8px;
    }}
    .stApp [data-testid="stAudioInputWaveformTimeCode"] {{ color: #14213d !important; opacity: 1 !important; }}
    .stApp [data-testid="stFileChip"] {{ background-color: #f4f7fb !important; }}
    .stApp [data-testid="stFileChip"], .stApp [data-testid="stFileChip"] * {{ color: #14213d !important; }}

    /* ---------- Cards and custom components ---------- */
    div[data-testid="stMetric"] {{
        background-color: {T["card_bg"]}; border: 1px solid {T["card_border"]};
        border-left: 4px solid {T["highlight"]}; border-radius: 10px; padding: 0.9rem 1rem;
    }}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] label * {{ color: {T["text_secondary"]} !important; }}
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{ color: {T["text"]} !important; }}
    div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {{ color: {T["text_secondary"]} !important; }}

    .pipeline-step {{
        background-color: {T["card_bg"]} !important; border: 1px solid {T["card_border"]};
        border-left: 4px solid {T["highlight"]} !important;
    }}
    .pipeline-step-title {{ color: {T["text"]} !important; }}
    .pipeline-step-desc {{ color: {T["text_secondary"]} !important; }}

    .style-badge {{
        display: inline-block; padding: 0.18rem 0.7rem; border-radius: 999px; font-size: 0.8rem;
        font-weight: 600; color: #FFFFFF !important; vertical-align: middle; margin: 0.15rem 0 0.9rem 0;
    }}
    .topic-title {{ font-size: 1.45rem; font-weight: 700; margin: 0.2rem 0 0.35rem 0; line-height: 1.25; }}
    .cloud-title {{ font-size: 1.05rem; font-weight: 700; margin: 0.4rem 0 0.2rem 0; line-height: 1.3; }}
    .complaint-quote {{
        background-color: {T["card_bg"]}; border: 1px solid {T["card_border"]}; border-radius: 10px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.7rem; font-size: 0.93rem; line-height: 1.55; color: {T["text"]};
    }}
    .complaint-meta {{ font-size: 0.8rem; color: {T["text_secondary"]} !important; margin-bottom: 0.35rem; }}

    /* Charts and word clouds (rendered as images) get soft rounded corners */
    .stApp [data-testid="stImage"] img, .stApp [data-testid="stImageContainer"] img {{ border-radius: 10px; }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {T["card_border"]}; }}
    .stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: 0.98rem; color: {T["text_secondary"]} !important; }}
    .stTabs [aria-selected="true"] p {{ color: {T["accent"] if selected_theme == "Night" else T["text"]} !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {T["accent"] if selected_theme == "Night" else T["highlight"]} !important; height: 3px; }}
    .stTabs [data-baseweb="tab-border"] {{ background-color: {T["card_border"]} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(title, subtitle):
    """Gradient banner at the top of every page.

    Uses plain <div>s rather than <h1>/<p>: Streamlit wraps heading text in an
    inner <span>, and the theme's global span colour was painting that span navy,
    which is why the title looked dark on the banner. The .app-header * rule in the
    stylesheet forces everything inside the banner to white as a second guard."""
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-title">{title}</div>
            <div class="app-header-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def themed_axes(fig, ax):
    """Applies the current theme's colors to a matplotlib fig/ax pair -
    background, tick labels, axis labels, title, and spines - so every
    custom chart matches whichever theme is active."""
    fig.patch.set_facecolor(THEME["card_bg"])
    ax.set_facecolor(THEME["card_bg"])
    ax.tick_params(colors=THEME["text_secondary"])
    ax.xaxis.label.set_color(THEME["text"])
    ax.yaxis.label.set_color(THEME["text"])
    ax.title.set_color(THEME["text"])
    for spine in ax.spines.values():
        spine.set_color(THEME["grid_color"])
    return fig, ax


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def pdf_safe(text):
    """fpdf2's built-in Helvetica font only supports Latin-1, but the app's text -
    including user-typed topic names - can contain em-dashes, curly quotes, or
    emoji. Map common punctuation to ASCII equivalents and fall back to '?' for
    anything else unsupported, rather than letting the PDF export crash."""
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...",
    }
    for uni_char, ascii_char in replacements.items():
        text = text.replace(uni_char, ascii_char)
    return text.encode("latin-1", "replace").decode("latin-1")


def get_topic_label(idx, model_choice="NMF"):
    """Returns the saved or edited name for a topic of the given model,
    falling back to 'Topic N'. Names are stored per model, because NMF's
    Topic 3 and LDA's Topic 3 are unrelated topics."""
    idx = int(idx)
    return st.session_state["topic_labels"].get(model_choice, {}).get(idx, f"Topic {idx}")


STYLE_COLORS = {
    style: readable_on(color, "#FFFFFF")   # darkened until white badge text passes 4.5:1
    for style, color in {
        "Complaint issue": "#2f5b93",
        "Legal template": "#f47a21",
        "Narrative account": "#2f95b8",
    }.items()
}


def get_topic_style(idx, model_choice="NMF"):
    return TOPIC_STYLES.get(model_choice, {}).get(int(idx))


def style_badge(style):
    """Small coloured pill showing how a topic's complaints tend to be written."""
    if not style:
        return ""
    color = STYLE_COLORS.get(style, "#5e5e5f")
    return f"<span class='style-badge' style='background-color:{color};'>{style}</span>"


def clean_redactions(text):
    """CFPB masks personal details as XXXX, XX/XX/XXXX and so on. Replace those
    runs with a readable marker when showing real complaints on screen."""
    text = re.sub(r"\bX{2,}(?:[/\-]X{2,})*\b", "[redacted]", str(text))
    text = re.sub(r"\{\$[\d,.]+\}", lambda m: m.group(0)[1:-1], text)
    return re.sub(r"\s+", " ", text).strip()


def blend_hex(start_hex, end_hex, amount):
    """Colour `amount` of the way from start_hex to end_hex (0 = start, 1 = end)."""
    a, b = hex_to_rgb(start_hex), hex_to_rgb(end_hex)
    mixed = [round(x + (y - x) * amount) for x, y in zip(a, b)]
    return "#{:02x}{:02x}{:02x}".format(*mixed)


@st.cache_data(show_spinner=False)
def wordcloud_png(freq_items, color_hex, bg_color, width=1000, height=420):
    """Cached word cloud as PNG bytes. freq_items is a tuple of (word, weight)
    pairs so Streamlit can hash it; the colormap runs from the page background
    into the topic's colour so clouds sit naturally on both themes."""
    strongest = readable_on(color_hex, bg_color, 3.0)
    faintest = readable_on(blend_hex(bg_color, strongest, 0.55), bg_color, 3.0)
    cmap = LinearSegmentedColormap.from_list("t", [faintest, strongest])
    wc = WordCloud(
        width=width, height=height, background_color=bg_color, colormap=cmap,
        max_words=100, prefer_horizontal=0.95, random_state=42,
    ).generate_from_frequencies(dict(freq_items))
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


def topic_top_terms(model, feature_names, topic_idx, n=30):
    weights = model.components_[topic_idx]
    top = weights.argsort()[::-1][:n]
    return tuple((str(feature_names[i]), float(weights[i])) for i in top)


def topic_distribution_chart(dist, model_choice, total_docs, height_per_bar=0.42):
    """Horizontal bars sorted by size, labelled with topic names, counts and shares.
    Easier to read than vertical bars with rotated names."""
    order = dist.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, height_per_bar * len(order) + 0.9))
    themed_axes(fig, ax)
    colors = [UI_TOPIC_COLORS[i % len(UI_TOPIC_COLORS)] for i in order.index]
    labels = [(lbl if lbl == f"Topic {i}" else f"{i}. {lbl}")
              for i, lbl in ((i, get_topic_label(i, model_choice)) for i in order.index)]
    bars = ax.barh(labels, order.values, color=colors)
    peak = max(order.values) if len(order) else 1
    for bar, value in zip(bars, order.values):
        share = value / total_docs if total_docs else 0
        ax.text(bar.get_width() + peak * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{value:,}  ({share:.0%})", va="center", fontsize=9, color=THEME["text_secondary"])
    ax.set_xlim(0, peak * 1.2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # whole complaints only
    ax.set_xlabel("Complaints")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def distribution_table(dist, model_choice, total_docs):
    """Topic counts as a DataFrame, used for the CSV download."""
    rows = []
    for idx, count in dist.sort_values(ascending=False).items():
        rows.append({
            "topic_number": int(idx),
            "topic_name": get_topic_label(idx, model_choice),
            "writing_style": get_topic_style(idx, model_choice) or "",
            "complaints": int(count),
            "share_pct": round(count / total_docs * 100, 2) if total_docs else 0,
        })
    return pd.DataFrame(rows)


def generate_executive_summary(dist_series, total_docs, model_choice):
    """Auto-written narrative paragraph summarizing the topic distribution -
    reusable in-app, in the PDF report, and in the slide deck."""
    top_idx = int(dist_series.idxmax())
    top_count = int(dist_series.max())
    top_pct = (top_count / total_docs * 100) if total_docs else 0
    n_topics = len(dist_series)
    label = get_topic_label(top_idx, model_choice)
    return (
        f"Across {total_docs:,} complaints analyzed with {model_choice}, the largest topic — "
        f"\"{label}\" — accounts for {top_pct:.0f}% of documents ({top_count:,} complaints). "
        f"The remaining {total_docs - top_count:,} complaints are distributed across the other "
        f"{n_topics - 1} topics, suggesting a mix of dominant and long-tail complaint themes."
    )


# A small built-in sample so the "Analyze & Report" page can be demoed instantly,
# without needing a live file upload during a presentation.
SAMPLE_COMPLAINTS = [
    "A debt collector has been calling me multiple times a day, even after I asked them to stop contacting me at work.",
    "I received a collection notice for a debt that is not mine, and I have already disputed this with the company twice.",
    "The collection agency threatened to sue me and garnish my wages if I did not pay immediately.",
    "I paid this debt in full last year but the collector is still reporting it as unpaid on my credit report.",
    "My credit report shows an account that I never opened, and I believe I am a victim of identity theft.",
    "I disputed an error on my credit report three months ago and it still has not been corrected.",
    "The company keeps calling my family members and neighbors about my debt, which feels like harassment.",
    "I asked for written validation of this debt and never received anything, yet they continue to call.",
    "My credit score dropped significantly after a collection account appeared that I do not recognize.",
    "The collector used abusive language and made threats during a phone call about a medical bill.",
    "I was never notified about this debt before it appeared on my credit report as a collection account.",
    "I have been trying to reach the company for weeks to correct an error in the amount owed.",
    "This account was already discharged in bankruptcy but is still being pursued by a collection agency.",
    "The debt collector contacted my employer directly, which I believe is against the rules.",
    "I keep receiving calls early in the morning and late at night about a debt I already settled.",
]


def build_pdf_report(new_df, assignments, topic_dist, model, feature_names, model_choice, source_name):
    """Builds a branded PDF summarizing topic assignments for an uploaded file.
    Returns raw PDF bytes, ready for st.download_button."""
    n_topics = model.components_.shape[0]
    dist = pd.Series(assignments).value_counts().reindex(range(n_topics), fill_value=0)
    summary_text = generate_executive_summary(dist, len(new_df), model_choice)

    # --- Chart: documents per topic, in brand colors, labeled with custom topic names ---
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bar_colors = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in dist.index]
    labels_x = [get_topic_label(i, model_choice) for i in dist.index]
    ax.bar(labels_x, dist.values, color=bar_colors)
    ax.set_ylabel("Documents")
    ax.set_title("Documents per Topic")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    fig.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        chart_path = os.path.join(tmpdir, "chart.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # --- Header ---
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(*hex_to_rgb(NAVY))
        pdf.cell(0, 12, "Customer Complaint Topic Report",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*hex_to_rgb(GRAY))
        pdf.cell(0, 8,
                 pdf_safe(f"Source file: {source_name}  |  Model: {model_choice}  |  "
                          f"Generated {datetime.now().strftime('%B %d, %Y %H:%M')}"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        pdf.set_draw_color(*hex_to_rgb(SKY))
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        # --- Summary ---
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*hex_to_rgb(NAVY))
        pdf.cell(0, 8, "Executive Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, pdf_safe(summary_text))
        pdf.ln(2)

        pdf.image(chart_path, x=15, w=180)
        pdf.ln(4)

        # --- Per-topic keyword breakdown ---
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*hex_to_rgb(NAVY))
        pdf.cell(0, 8, "Topic Keywords", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        for i in range(n_topics):
            weights = model.components_[i]
            top_idx = weights.argsort()[:-11:-1]
            words = ", ".join(feature_names[j] for j in top_idx)
            doc_count = int(dist.get(i, 0))

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*hex_to_rgb(TOPIC_COLORS[i % len(TOPIC_COLORS)]))
            pdf.multi_cell(0, 5.5, pdf_safe(f"{get_topic_label(i, model_choice)}  ({doc_count} documents)"),
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5.5, pdf_safe(words), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.5)

        return bytes(pdf.output())


def _pptx_slide_header(slide, title, color_rgb, prs):
    """Colored header bar + title text, reused across every content slide."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.9))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color_rgb
    bar.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.15), Inches(11.5), Inches(0.6))
    tb.text_frame.text = title
    tb.text_frame.paragraphs[0].font.size = Pt(26)
    tb.text_frame.paragraphs[0].font.bold = True
    tb.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)


def build_pptx_report(new_df, assignments, topic_dist, model, feature_names, model_choice, source_name):
    """Builds a branded slide deck (title, distribution chart, executive summary,
    one slide per topic) from the same analysis results as the PDF report."""
    n_topics = model.components_.shape[0]
    dist = pd.Series(assignments).value_counts().reindex(range(n_topics), fill_value=0)
    summary_text = generate_executive_summary(dist, len(new_df), model_choice)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bar_colors = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in dist.index]
    labels_x = [get_topic_label(i, model_choice) for i in dist.index]
    ax.bar(labels_x, dist.values, color=bar_colors)
    ax.set_ylabel("Documents")
    ax.set_title("Documents per Topic")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        chart_path = os.path.join(tmpdir, "chart.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        blank_layout = prs.slide_layouts[6]

        navy_rgb = RGBColor(*hex_to_rgb(NAVY))
        gray_rgb = RGBColor(*hex_to_rgb(GRAY))

        # --- Title slide ---
        slide = prs.slides.add_slide(blank_layout)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = navy_rgb
        bg.line.fill.background()

        title_tb = slide.shapes.add_textbox(Inches(0.8), Inches(2.7), Inches(11.5), Inches(1.5))
        title_tb.text_frame.text = "Customer Complaint Topic Report"
        title_tb.text_frame.paragraphs[0].font.size = Pt(40)
        title_tb.text_frame.paragraphs[0].font.bold = True
        title_tb.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)

        sub_tb = slide.shapes.add_textbox(Inches(0.8), Inches(4.0), Inches(11.5), Inches(0.8))
        sub_tb.text_frame.text = (
            f"Source: {source_name}  ·  Model: {model_choice}  ·  "
            f"{datetime.now().strftime('%B %d, %Y')}"
        )
        sub_tb.text_frame.paragraphs[0].font.size = Pt(16)
        sub_tb.text_frame.paragraphs[0].font.color.rgb = RGBColor(*hex_to_rgb(SKY))

        # --- Executive summary slide ---
        slide_sum = prs.slides.add_slide(blank_layout)
        _pptx_slide_header(slide_sum, "Executive Summary", navy_rgb, prs)
        sum_tb = slide_sum.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(4.5))
        sum_tb.text_frame.word_wrap = True
        sum_tb.text_frame.text = summary_text
        sum_tb.text_frame.paragraphs[0].font.size = Pt(20)
        sum_tb.text_frame.paragraphs[0].font.color.rgb = RGBColor(40, 40, 40)

        # --- Distribution chart slide ---
        slide_dist = prs.slides.add_slide(blank_layout)
        _pptx_slide_header(slide_dist, "Topic Distribution", navy_rgb, prs)
        slide_dist.shapes.add_picture(chart_path, Inches(0.9), Inches(1.3), width=Inches(11.5))

        # --- One slide per topic ---
        for i in range(n_topics):
            weights = model.components_[i]
            top_idx = weights.argsort()[:-11:-1]
            words = ", ".join(feature_names[j] for j in top_idx)
            doc_count = int(dist.get(i, 0))

            slide_k = prs.slides.add_slide(blank_layout)
            color = RGBColor(*hex_to_rgb(TOPIC_COLORS[i % len(TOPIC_COLORS)]))
            _pptx_slide_header(slide_k, get_topic_label(i, model_choice), color, prs)

            count_tb = slide_k.shapes.add_textbox(Inches(0.8), Inches(1.25), Inches(11.5), Inches(0.6))
            count_tb.text_frame.text = f"{doc_count} documents"
            count_tb.text_frame.paragraphs[0].font.size = Pt(18)
            count_tb.text_frame.paragraphs[0].font.color.rgb = gray_rgb

            words_tb = slide_k.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(11.5), Inches(3.8))
            words_tb.text_frame.word_wrap = True
            words_tb.text_frame.text = words
            words_tb.text_frame.paragraphs[0].font.size = Pt(24)
            words_tb.text_frame.paragraphs[0].font.color.rgb = RGBColor(40, 40, 40)

        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()


@st.cache_data
def load_data():
    df = pd.read_csv("data/complaints_clean.csv")
    return df


@st.cache_resource
def load_models():
    tfidf_vectorizer = joblib.load("data/tfidf_vectorizer.joblib")
    count_vectorizer = joblib.load("data/count_vectorizer.joblib")
    lda_model = joblib.load("data/lda_model.joblib")
    nmf_model = joblib.load("data/nmf_model.joblib")
    lda_doc_topic = np.load("data/lda_doc_topic.npy")
    nmf_doc_topic = np.load("data/nmf_doc_topic.npy")
    return tfidf_vectorizer, count_vectorizer, lda_model, nmf_model, lda_doc_topic, nmf_doc_topic


def get_top_words(model, feature_names, n_top=10):
    topics = []
    for topic_weights in model.components_:
        top_indices = topic_weights.argsort()[: -n_top - 1 : -1]
        topics.append([feature_names[i] for i in top_indices])
    return topics


@st.cache_data(show_spinner=False)
def read_prebuilt_map(path, modified_time):
    """modified_time is part of the cache key, so a rebuilt file is picked up."""
    with open(path, encoding="utf-8") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def build_topic_map_live(_model, feature_names, doc_topic, _doc_term_matrix, model_choice):
    """Fallback when no pre-built map exists. Underscore arguments are left out of
    Streamlit's cache key because models and sparse matrices don't hash reliably."""
    return build_topic_map_html(_model, feature_names, doc_topic, _doc_term_matrix)


def get_topic_map_html(model_choice, model, feature_names, doc_topic, vectorizer, texts):
    """Returns (html, source) where source is 'prebuilt' or 'live'."""
    path = f"data/topic_map_{model_choice.lower()}.html"
    if os.path.exists(path):
        return read_prebuilt_map(path, os.path.getmtime(path)), "prebuilt"
    html = build_topic_map_live(model, feature_names, doc_topic, vectorizer.transform(texts), model_choice)
    return html, "live"


@st.cache_data(show_spinner=False)
def corpus_word_counts(_df, n=200):
    """Most frequent cleaned words across the corpus. Cached once per session,
    since the corpus doesn't change while the app runs."""
    counts = _df["clean_text"].astype(str).str.split().explode().value_counts()
    return counts.head(n)


@st.cache_data(show_spinner=False)
def corpus_summary(_df):
    lengths = _df["clean_text"].astype(str).str.split().str.len()
    vocab = _df["clean_text"].astype(str).str.split().explode().nunique()
    return len(_df), float(lengths.mean()), int(vocab), lengths


# --- Sidebar navigation ---
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">📋</div>
        <div class="sidebar-brand-text">
            Complaint Topic<br>Modelling
            <small>Group 8 · MSBA610</small>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Voice Complaints helpers ----------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_speech_model():
    """Whisper speech model, loaded once per server and shared by every visitor."""
    return speech.load_model()


def get_speech_model():
    """Returns (model, None) or (None, message). Never raises, so the page can explain problems."""
    try:
        return load_speech_model(), None
    except ImportError:
        return None, ("Voice features need the faster-whisper package. Run "
                      "`pip install -r requirements.txt`, then restart the app.")
    except Exception as err:
        return None, ("The speech model couldn't be loaded. It downloads automatically the first time "
                      "it's used, so check the internet connection and try again. "
                      f"(Details: {type(err).__name__})")


def classify_text(text, model_choice, models):
    """Classify one complaint with the chosen model. result['ok'] is False when none of
    its words are in the model's vocabulary."""
    tfidf_vectorizer, count_vectorizer, lda_model, nmf_model, _, _ = models
    model, vectorizer = (lda_model, count_vectorizer) if model_choice == "LDA" else (nmf_model, tfidf_vectorizer)
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    if vec.nnz == 0:
        return {"ok": False, "cleaned": cleaned}
    raw = model.transform(vec)[0]
    total = raw.sum()
    shares = raw / total if total > 0 else raw
    ranked = [int(i) for i in np.argsort(shares)[::-1][:3]]
    return {"ok": True, "cleaned": cleaned, "shares": shares, "ranked": ranked, "best": ranked[0],
            "model": model, "vectorizer": vectorizer}


def show_classification(result, model_choice):
    """Same display as Try It Yourself: best match, writing style, top terms, top three topics."""
    if not result.get("ok"):
        st.warning("None of these words appear in the model's vocabulary, so the complaint can't be "
                   "placed in a topic. Try a longer recording that describes the problem.")
        return
    best, shares = result["best"], result["shares"]
    feature_names = result["vectorizer"].get_feature_names_out()
    color = UI_TOPIC_TEXT[best % len(UI_TOPIC_TEXT)]
    st.markdown(
        f"<div class='topic-title' style='color:{color};'>Best match: Topic {best}, "
        f"{get_topic_label(best, model_choice)}</div>{style_badge(get_topic_style(best, model_choice))}",
        unsafe_allow_html=True,
    )
    st.caption("Top terms for this topic: " + ", ".join(w for w, _ in topic_top_terms(result["model"], feature_names, best, n=8)))
    for idx in result["ranked"]:
        st.progress(float(min(max(shares[idx], 0.0), 1.0)), text=f"{idx}. {get_topic_label(idx, model_choice)}: {shares[idx]:.0%}")
    vocab = set(feature_names)
    matched = [w for w in result["cleaned"].split() if w in vocab]
    if matched:
        with st.expander("Words the model recognised"):
            st.write(", ".join(dict.fromkeys(matched)))


def fmt_duration(seconds):
    seconds = int(round(seconds or 0))
    return f"{seconds // 60}:{seconds % 60:02d}"


PAGE_NAMES = ["How It Works", "Corpus Overview", "Preprocessing Demo", "Topic Explorer", "Model Comparison", "Try It Yourself", "Voice Complaints", "Analyze & Report"]
PAGE_ICONS = {
    "How It Works": "🧭",
    "Corpus Overview": "📊",
    "Preprocessing Demo": "🧹",
    "Topic Explorer": "🔍",
    "Model Comparison": "⚖️",
    "Try It Yourself": "✍️",
    "Voice Complaints": "🎙️",
    "Analyze & Report": "📤",
}
nav_choice = st.sidebar.radio(
    "Navigate",
    [f"{PAGE_ICONS[name]}  {name}" for name in PAGE_NAMES],
    label_visibility="collapsed",
)
page = nav_choice.split("  ", 1)[1]

data_ready = os.path.exists("data/complaints_clean.csv")
models_ready = os.path.exists("data/lda_model.joblib")

# ============================================================
# PAGE: How It Works (viewable even before the pipeline has been run)
# ============================================================
if page == "How It Works":
    page_header("How It Works", "The pipeline behind this app, in five steps.")

    st.markdown(
        """
        <style>
        .pipeline-step {
            display: flex;
            gap: 1rem;
            padding: 1rem 1.2rem;
            border-radius: 10px;
            background-color: #f4f6f9;
            border-left: 4px solid #45c8f3;
            margin-bottom: 0.85rem;
        }
        .pipeline-step-icon {
            font-size: 1.6rem;
            flex-shrink: 0;
        }
        .pipeline-step-title {
            font-weight: 700;
            color: #1b3865;
            margin-bottom: 0.15rem;
        }
        .pipeline-step-desc {
            color: #3a3a3a;
            font-size: 0.92rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        ("🧹", "Clean & Preprocess",
         "Raw complaint text is stripped of redacted placeholders, dollar amounts, URLs, and "
         "punctuation, then tokenized, stripped of stopwords, and lemmatized, which turns plurals "
         "into their base form ('calls' becomes 'call')."),
        ("🔢", "Represent as Numbers",
         "Cleaned text is converted into two numeric forms: TF-IDF weighted vectors for NMF, "
         "and raw word counts for LDA — matching how each algorithm is designed to work."),
        ("🧩", "Discover Topics",
         "Two unsupervised models — Latent Dirichlet Allocation (LDA) and Non-negative Matrix "
         "Factorization (NMF) — each find a set of latent topics purely from word co-occurrence "
         "patterns, with no labels required."),
        ("📏", "Evaluate & Tune",
         "Topic coherence (do a topic's top words actually relate to each other?) and topic "
         "diversity (are topics distinct, not redundant?) are measured across a range of topic "
         "counts to pick the best-performing model and number of topics."),
        ("🗂️", "Explore & Deploy",
         "The winning model powers this app — browse topics, test new text, or upload a file "
         "for live analysis and a downloadable report, all running the model trained during "
         "development."),
    ]
    for icon, title, desc in steps:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <div class="pipeline-step-icon">{icon}</div>
                <div>
                    <div class="pipeline-step-title">{title}</div>
                    <div class="pipeline-step-desc">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Use the sidebar to explore each stage: **Corpus Overview** and **Preprocessing Demo** "
        "cover steps 1-2, **Topic Explorer** and **Model Comparison** cover steps 3-4, and "
        "**Try It Yourself**, **Voice Complaints** and **Analyze & Report** cover step 5."
    )
    st.stop()

if not data_ready:
    st.warning(
        "No processed data found yet. Run `python src/preprocessing.py` from the terminal first, "
        "then `python src/modeling.py` to train the topic models."
    )
    st.stop()

df = load_data()

# ============================================================
# PAGE: Corpus Overview
# ============================================================
if page == "Corpus Overview":
    page_header("Corpus Overview", "A first look at the complaint dataset before any modelling.")

    n_docs, avg_words, vocab_size, lengths = corpus_summary(df)
    word_counts = corpus_word_counts(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total complaints", f"{n_docs:,}")
    col2.metric("Avg. words per complaint", f"{avg_words:.0f}")
    col3.metric("Vocabulary size", f"{vocab_size:,}")

    st.subheader("Complaint length distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    themed_axes(fig, ax)
    ax.hist(lengths, bins=40, color=UI_TOPIC_COLORS[0])
    ax.axvline(lengths.median(), color=UI_TOPIC_COLORS[1], linestyle="--", linewidth=1.5)
    ax.text(lengths.median(), ax.get_ylim()[1] * 0.92, f"  median {lengths.median():.0f} words",
            color=UI_TOPIC_COLORS[1], fontsize=9)
    ax.set_xlabel("Word count (after cleaning)")
    ax.set_ylabel("Number of complaints")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top words in the cleaned corpus")
        top_words = word_counts.head(15)
        fig_w, ax_w = plt.subplots(figsize=(6, 4.2))
        themed_axes(fig_w, ax_w)
        ax_w.barh(top_words.index[::-1], top_words.values[::-1], color=UI_TOPIC_COLORS[1])
        ax_w.set_xlabel("Occurrences")
        ax_w.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig_w)

    with col_b:
        st.subheader("☁️ Word Cloud")
        st.image(wordcloud_png(
            tuple((str(w), float(c)) for w, c in word_counts.items()),
            UI_TOPIC_COLORS[2], THEME["wc_bg"],
        ), width="stretch")

# ============================================================
# PAGE: Preprocessing Demo
# ============================================================
elif page == "Preprocessing Demo":
    page_header("Preprocessing Demo", "See the cleaning pipeline applied to raw text, live.")

    sample = st.text_area(
        "Complaint text",
        value="I called XXXX regarding my $500.00 charge and they never responded to my email at test@example.com!!",
        height=120,
    )
    if sample:
        st.subheader("Cleaned output")
        st.code(preprocess_text(sample), language=None)

# ============================================================
# PAGE: Topic Explorer
# ============================================================
elif page == "Topic Explorer":
    page_header("Topic Explorer", "Browse each discovered topic, its vocabulary, and the complaints behind it.")

    if not models_ready:
        st.warning("Run `python src/modeling.py` from the terminal first to train the models.")
        st.stop()

    tfidf_vectorizer, count_vectorizer, lda_model, nmf_model, lda_doc_topic, nmf_doc_topic = load_models()

    model_choice = st.selectbox(
        "Choose a model", MODEL_OPTIONS, format_func=lambda m: MODEL_CAPTIONS[m], key="explorer_model",
    )

    if model_choice == "LDA":
        model, vectorizer, doc_topic = lda_model, count_vectorizer, lda_doc_topic
    else:
        model, vectorizer, doc_topic = nmf_model, tfidf_vectorizer, nmf_doc_topic
    feature_names = vectorizer.get_feature_names_out()

    n_topics = model.components_.shape[0]
    rows_match = doc_topic.shape[0] == len(df)
    assignments = np.argmax(doc_topic, axis=1)
    dist = pd.Series(assignments).value_counts().reindex(range(n_topics), fill_value=0)
    total_docs = int(dist.sum())

    # ---------------- Name your topics ----------------
    with st.expander(f"✏️ Name your {model_choice} topics", expanded=False):
        st.caption(
            "Names apply everywhere in the app straight away, including the PDF and slide-deck "
            "exports. To keep them for every visitor, save them to data/topic_labels.json and "
            "commit that file to GitHub."
        )
        label_cols = st.columns(2)
        for i in range(n_topics):
            with label_cols[i % 2]:
                new_label = st.text_input(
                    f"Topic {i}", value=get_topic_label(i, model_choice),
                    key=f"label_{model_choice}_{i}",
                )
                st.session_state["topic_labels"][model_choice][i] = new_label.strip() or f"Topic {i}"

        export = {}
        for m in MODEL_OPTIONS:
            n_m = (nmf_model if m == "NMF" else lda_model).components_.shape[0]
            export[m] = {}
            for i in range(n_m):
                name = get_topic_label(i, m)
                style = get_topic_style(i, m)
                if name != f"Topic {i}" or style:
                    export[m][str(i)] = {"name": name, **({"style": style} if style else {})}
        export_json = json.dumps(export, indent=2)

        save_col, dl_col = st.columns(2)
        with save_col:
            if st.button("💾 Save names to data/topic_labels.json", width="stretch"):
                try:
                    with open(LABELS_PATH, "w", encoding="utf-8") as f:
                        f.write(export_json)
                    st.success("Saved. Commit data/topic_labels.json to keep these names on the deployed app.")
                except OSError as err:
                    st.error(f"Couldn't write the file ({err}). Use the download button instead.")
        with dl_col:
            st.download_button(
                "⬇️ Download names (JSON)", data=export_json, file_name="topic_labels.json",
                mime="application/json", width="stretch",
            )

    tab_overview, tab_deep, tab_clouds, tab_map = st.tabs(
        ["📊 Overview", "🔎 Topic deep dive", "☁️ Word clouds", "🗺️ Topic map"]
    )

    # ---------------- Overview ----------------
    with tab_overview:
        st.info(generate_executive_summary(dist, total_docs, model_choice))

        st.subheader("Complaints per topic")
        st.caption("Each complaint is counted under the topic it scores highest on.")
        st.pyplot(topic_distribution_chart(dist, model_choice, total_docs))

        dist_df = distribution_table(dist, model_choice, total_docs)
        st.download_button(
            "⬇️ Download topic distribution (CSV)",
            data=dist_df.to_csv(index=False).encode("utf-8"),
            file_name=f"topic_distribution_{model_choice.lower()}.csv",
            mime="text/csv",
        )

        styled = [s for s in (get_topic_style(i, model_choice) for i in range(n_topics)) if s]
        if styled:
            st.subheader("How complaints are written")
            st.caption(
                "Some topics group complaints by the issue raised. Others group them by writing "
                "style: formal dispute-letter templates versus first-person accounts."
            )
            style_counts = {}
            for i in range(n_topics):
                s = get_topic_style(i, model_choice)
                if s:
                    style_counts[s] = style_counts.get(s, 0) + int(dist[i])
            style_cols = st.columns(len(style_counts))
            for col, (s, count) in zip(style_cols, sorted(style_counts.items(), key=lambda kv: -kv[1])):
                with col:
                    st.markdown(style_badge(s), unsafe_allow_html=True)
                    st.metric(s, f"{count / total_docs:.0%}", f"{count:,} complaints", delta_color="off",
                              label_visibility="collapsed")

    # ---------------- Topic deep dive ----------------
    with tab_deep:
        order_by_size = dist.sort_values(ascending=False).index.tolist()
        option_labels = {i: f"{i}. {get_topic_label(i, model_choice)}  ({dist[i]:,} complaints)"
                         for i in order_by_size}
        chosen = st.selectbox(
            "Pick a topic",
            order_by_size,
            format_func=option_labels.get,
            key=f"deep_topic_{model_choice}",
        )
        color = UI_TOPIC_COLORS[chosen % len(UI_TOPIC_COLORS)]
        title_color = UI_TOPIC_TEXT[chosen % len(UI_TOPIC_TEXT)]
        rank = order_by_size.index(chosen) + 1

        st.markdown(
            f"<div class='topic-title' style='color:{title_color};'>Topic {chosen}: "
            f"{get_topic_label(chosen, model_choice)}</div>"
            f"{style_badge(get_topic_style(chosen, model_choice))}",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Complaints", f"{dist[chosen]:,}")
        m2.metric("Share of corpus", f"{dist[chosen] / total_docs:.1%}" if total_docs else "-")
        m3.metric("Size rank", f"{rank} of {n_topics}")

        terms = topic_top_terms(model, feature_names, chosen, n=30)
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Top terms by weight**")
            top12 = terms[:12][::-1]
            fig_t, ax_t = plt.subplots(figsize=(6, 4.6))
            themed_axes(fig_t, ax_t)
            ax_t.barh([w for w, _ in top12], [v for _, v in top12], color=color)
            ax_t.set_xlabel("Weight in topic")
            ax_t.spines[["top", "right"]].set_visible(False)
            fig_t.tight_layout()
            st.pyplot(fig_t)
        with right:
            st.markdown("**Word cloud**")
            st.image(wordcloud_png(terms, color, THEME["wc_bg"], width=800, height=620), width="stretch")

        st.markdown("**Most representative complaints**")
        if not rows_match:
            st.caption(
                f"Complaint examples are unavailable: the saved model has {doc_topic.shape[0]:,} rows "
                f"but the corpus has {len(df):,}. Re-run modeling.py so they match."
            )
        else:
            text_col = "raw_text" if "raw_text" in df.columns else "clean_text"
            weights = doc_topic[:, chosen]
            row_totals = doc_topic.sum(axis=1)
            candidates = np.argsort(weights)[::-1][:3]
            st.caption("The three complaints that score highest on this topic, with personal details redacted by the CFPB.")
            for n, d in enumerate(candidates, start=1):
                share = weights[d] / row_totals[d] if row_totals[d] > 0 else 0
                text = clean_redactions(df.iloc[d][text_col])
                shown = text if len(text) <= 700 else text[:700].rsplit(" ", 1)[0] + " …"
                st.markdown(
                    f"<div class='complaint-quote'><div class='complaint-meta'>Example {n}: "
                    f"{share:.0%} of this complaint's topic weight falls on this topic</div>{shown}</div>",
                    unsafe_allow_html=True,
                )

    # ---------------- Word clouds ----------------
    with tab_clouds:
        st.caption("Word size reflects how strongly that word defines the topic.")
        for row_start in range(0, n_topics, 3):
            cols = st.columns(3)
            for offset, col in enumerate(cols):
                topic_idx = row_start + offset
                if topic_idx >= n_topics:
                    continue
                t_color = UI_TOPIC_COLORS[topic_idx % len(UI_TOPIC_COLORS)]
                t_terms = topic_top_terms(model, feature_names, topic_idx, n=30)
                with col:
                    st.markdown(
                        f"<div class='cloud-title' style='color:{UI_TOPIC_TEXT[topic_idx % len(UI_TOPIC_TEXT)]};'>{topic_idx}. "
                        f"{get_topic_label(topic_idx, model_choice)}</div>",
                        unsafe_allow_html=True,
                    )
                    st.image(wordcloud_png(t_terms, t_color, THEME["wc_bg"], width=700, height=420), width="stretch")
                    st.caption(", ".join(w for w, _ in t_terms[:5]))

    # ---------------- Topic map ----------------
    with tab_map:
        st.caption(
            "Bubble size shows how common a topic is; distance shows how related topics are. "
            "Click a bubble to see its top terms. Bubble numbers match the topic numbers used in this app."
        )
        legend = ", ".join(f"{i} = {get_topic_label(i, model_choice)}" for i in range(n_topics))
        st.caption(f"Key: {legend}")
        try:
            with st.spinner("Loading the interactive map..."):
                vis_html, source = get_topic_map_html(
                    model_choice, model, feature_names, doc_topic, vectorizer, df["clean_text"].astype(str),
                )
            st.iframe(vis_html, height=700)
            if source == "live":
                st.caption("Built live. Run `python src/build_topic_maps.py` to pre-build it for faster loading.")
        except Exception:
            st.info(
                "The interactive map couldn't be generated for this model. The overview, deep dive "
                "and word clouds still show every topic in full. Try switching model, or run "
                "`python src/build_topic_maps.py` locally and commit the saved map."
            )

# ============================================================
# PAGE: Model Comparison
# ============================================================
elif page == "Model Comparison":
    page_header("Model Comparison", "LDA vs NMF, side by side on coherence and diversity.")

    sweep_path = "data/n_topics_sweep.csv"
    if os.path.exists(sweep_path):
        sweep_df = pd.read_csv(sweep_path)
        fig_c, ax_c = plt.subplots(figsize=(8, 3.5))
        themed_axes(fig_c, ax_c)
        ax_c.plot(sweep_df["n_topics"], sweep_df["lda_coherence"], marker="o",
                  color=UI_TOPIC_COLORS[1], label="LDA", linewidth=2)
        ax_c.plot(sweep_df["n_topics"], sweep_df["nmf_coherence"], marker="o",
                  color=UI_TOPIC_COLORS[0], label="NMF", linewidth=2)
        ax_c.set_xlabel("Number of topics")
        ax_c.set_ylabel("Coherence (c_v)")
        legend = ax_c.legend(facecolor=THEME["card_bg"], edgecolor=THEME["grid_color"])
        for text in legend.get_texts():
            text.set_color(THEME["text"])
        ax_c.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig_c)

        best_lda = sweep_df.loc[sweep_df["lda_coherence"].idxmax()]
        best_nmf = sweep_df.loc[sweep_df["nmf_coherence"].idxmax()]

        col1, col2 = st.columns(2)
        col1.metric("Best LDA coherence", f"{best_lda['lda_coherence']:.3f}", f"at n_topics={int(best_lda['n_topics'])}")
        col2.metric("Best NMF coherence", f"{best_nmf['nmf_coherence']:.3f}", f"at n_topics={int(best_nmf['n_topics'])}")
    else:
        st.info("Run `python src/evaluation.py` first to generate data/n_topics_sweep.csv.")

    st.subheader("All four evaluation criteria")
    st.caption("The project brief asks for coherence, diversity, interpretability and distribution. "
               "The two models found different topics, so each is judged on its own 11.")

    rows = []      # (criterion, what it checks, LDA value, NMF value, higher_is_better, formatter)
    metrics_path = "data/final_model_metrics.csv"
    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path).set_index("Metric")
        rows.append(("Coherence (c_v)", "Do a topic's top words appear together in real complaints?",
                     float(m.loc["Topic Coherence (c_v)", "LDA"]), float(m.loc["Topic Coherence (c_v)", "NMF"]), True, "{:.3f}"))
        rows.append(("Diversity", "Share of top words unique to one topic",
                     float(m.loc["Topic Diversity", "LDA"]), float(m.loc["Topic Diversity", "NMF"]), True, "{:.3f}"))
    interp_path = "data/interpretability_scores.csv"
    if os.path.exists(interp_path):
        ip = pd.read_csv(interp_path).set_index("model")
        rows.append(("Interpretability (1 to 5)", f"Blind group rating, {int(ip['raters'].max())} raters",
                     float(ip.loc["LDA", "overall"]), float(ip.loc["NMF", "overall"]), True, "{:.2f}"))
    else:
        rows.append(("Interpretability (1 to 5)", "Blind group rating", None, None, True, "{:.2f}"))
    dist = {}
    if models_ready:
        _, _, _, _, lda_doc_topic_cmp, nmf_doc_topic_cmp = load_models()
        for name, dt in [("LDA", lda_doc_topic_cmp), ("NMF", nmf_doc_topic_cmp)]:
            counts = topic_counts(dt); dist[name] = (counts, summarize(counts))
        rows.append(("Distribution evenness (0 to 1)", "How evenly complaints spread across topics",
                     dist["LDA"][1]["evenness"], dist["NMF"][1]["evenness"], True, "{:.3f}"))
        rows.append(("Largest topic", "Share of complaints in the biggest topic",
                     dist["LDA"][1]["largest_share"], dist["NMF"][1]["largest_share"], None, "{:.1%}"))
        rows.append(("Topics under 2%", "Topics that hold almost no complaints",
                     dist["LDA"][1]["topics_under_2pct"], dist["NMF"][1]["topics_under_2pct"], None, "{:d}"))

    def show(v, f):
        return "Not measured yet" if v is None else f.format(v)

    def leader(lda, nmf, higher):
        if higher is None or lda is None or nmf is None or abs(lda - nmf) < 1e-9:
            return ""
        return "NMF" if (nmf > lda) == higher else "LDA"

    table_rows = [{"Criterion": c, "What it checks": w, "LDA": show(l, f), "NMF": show(n, f), "Ahead": leader(l, n, h)}
                  for c, w, l, n, h, f in rows]
    st.table(pd.DataFrame(table_rows).set_index("Criterion"))

    leads = {"NMF": [], "LDA": []}
    for c, w, l, n, h, f in rows:
        who = leader(l, n, h)
        if who:
            leads[who].append(c.split(" (")[0].lower())
    parts = []
    for who in ["NMF", "LDA"]:
        if leads[who]:
            items = leads[who]
            listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
            parts.append(f"{who} is ahead on {listed}.")
    if parts:
        st.success(" ".join(parts))
    if not os.path.exists(interp_path):
        st.caption("Interpretability needs the group's ratings: run `python src/make_rating_sheet.py`, have each "
                   "member fill in a copy, then run `python src/score_ratings.py`.")
    st.caption("Evenness is reported rather than treated as a strict win. Real complaint themes are not equally "
               "common, so a perfectly even spread isn't the goal. The warning signs are a topic holding almost "
               "nothing, or one topic holding most complaints.")

    if dist:
        st.subheader("How the complaints spread")
        col_l, col_n = st.columns(2)
        for col, name in [(col_l, "LDA"), (col_n, "NMF")]:
            with col:
                st.markdown(f"**{name}**")
                st.pyplot(topic_distribution_chart(dist[name][0], name, dist[name][1]["complaints"]))

# ============================================================
# PAGE: Try It Yourself
# ============================================================
elif page == "Try It Yourself":
    page_header("Try It Yourself", "Paste a complaint, or pick an example, and see which topics it belongs to.")

    if not models_ready:
        st.warning("Run `python src/modeling.py` from the terminal first to train the models.")
        st.stop()

    tfidf_vectorizer, count_vectorizer, lda_model, nmf_model, lda_doc_topic, nmf_doc_topic = load_models()

    TRY_EXAMPLES = {
        "Repeated calls": "They call me five or six times a day about a debt, including at work, even after I told them in writing to stop calling.",
        "Identity theft": "There is a collection account on my credit report that I never opened. I believe I am a victim of identity theft and I filed a police report.",
        "Validation request": "I sent a letter asking the collector to validate this alleged debt and provide proof from the original creditor, but they never responded and keep reporting it.",
    }

    def use_example(text):
        st.session_state["try_text"] = text

    st.session_state.setdefault("try_text", "")
    st.caption("Try an example:")
    ex_cols = st.columns(len(TRY_EXAMPLES))
    for col, (label, text) in zip(ex_cols, TRY_EXAMPLES.items()):
        col.button(label, on_click=use_example, args=(text,), width="stretch", key=f"ex_{label}")

    user_text = st.text_area("Complaint text", key="try_text", height=150,
                             placeholder="Describe a debt collection problem in a few sentences...")
    model_choice = st.selectbox("Model", MODEL_OPTIONS, format_func=lambda m: MODEL_CAPTIONS[m], key="try_model")

    if st.button("Classify complaint", type="primary") and user_text.strip():
        cleaned = preprocess_text(user_text)
        if model_choice == "LDA":
            model, vectorizer = lda_model, count_vectorizer
        else:
            model, vectorizer = nmf_model, tfidf_vectorizer

        vec = vectorizer.transform([cleaned])
        if vec.nnz == 0:
            st.warning(
                "None of the words in this text appear in the model's vocabulary, so it can't be "
                "placed in a topic. Try a longer complaint that describes the problem."
            )
            st.stop()

        raw_scores = model.transform(vec)[0]
        total = raw_scores.sum()
        shares = raw_scores / total if total > 0 else raw_scores
        ranked = np.argsort(shares)[::-1][:3]
        feature_names = vectorizer.get_feature_names_out()

        best = int(ranked[0])
        best_color = UI_TOPIC_TEXT[best % len(UI_TOPIC_TEXT)]
        st.markdown(
            f"<div class='topic-title' style='color:{best_color};'>Best match: Topic {best}, "
            f"{get_topic_label(best, model_choice)}</div>"
            f"{style_badge(get_topic_style(best, model_choice))}",
            unsafe_allow_html=True,
        )
        st.caption("Top terms for this topic: " + ", ".join(w for w, _ in topic_top_terms(model, feature_names, best, n=8)))

        st.subheader("Topic breakdown")
        for idx in ranked:
            idx = int(idx)
            st.progress(
                float(min(max(shares[idx], 0.0), 1.0)),
                text=f"{idx}. {get_topic_label(idx, model_choice)}: {shares[idx]:.0%}",
            )

        vocab = set(feature_names)
        matched = [w for w in cleaned.split() if w in vocab]
        if matched:
            with st.expander("Words the model recognised"):
                st.write(", ".join(dict.fromkeys(matched)))

# ============================================================
# PAGE: Voice Complaints
# ============================================================
elif page == "Voice Complaints":
    page_header("Voice Complaints", "Record or upload spoken complaints, turn them into text, and see which topics they fall under.")

    if not models_ready:
        st.warning("Run `python src/modeling.py` from the terminal first to train the models.")
        st.stop()

    models = load_models()
    model_choice = st.selectbox("Model", MODEL_OPTIONS, format_func=lambda m: MODEL_CAPTIONS[m], key="voice_model")
    st.caption("Speech is turned into text by OpenAI's Whisper model (English, base size), running inside this app. "
               "Recordings are not saved. The first transcription after the app wakes up can take up to a "
               "minute while the speech model downloads.")

    tab_rec, tab_up = st.tabs(["🎙️ Record a complaint", "📁 Upload recordings"])

    # ---------------- Record ----------------
    with tab_rec:
        st.markdown("Click the microphone, speak for 10 to 30 seconds, then click it again to stop. "
                    "Two or three sentences classify more reliably than one.")
        recording = st.audio_input("Record a complaint", key="voice_recording")
        if recording is not None:
            audio_bytes = recording.getvalue()
            signature = hashlib.md5(audio_bytes).hexdigest()
            ready = True
            if st.session_state.get("voice_signature") != signature:      # a new recording: transcribe it once
                whisper, problem = get_speech_model()
                if problem:
                    st.error(problem); ready = False
                else:
                    with st.spinner("Turning speech into text..."):
                        try:
                            text, seconds = speech.transcribe(whisper, io.BytesIO(audio_bytes))
                            st.session_state["voice_signature"] = signature
                            st.session_state["voice_transcript"] = text
                            st.session_state["voice_seconds"] = seconds
                        except Exception as err:
                            st.error(f"The recording couldn't be read ({type(err).__name__}). Please record again.")
                            ready = False
            if ready:
                transcript = st.text_area("Transcript", key="voice_transcript", height=120,
                                          help="Fix any misheard words, then press Ctrl+Enter to classify again.")
                st.caption(f"Recording length {fmt_duration(st.session_state.get('voice_seconds'))}. "
                           "Edit the transcript and press Ctrl+Enter to classify again.")
                if transcript.strip():
                    show_classification(classify_text(transcript, model_choice, models), model_choice)
                else:
                    st.warning("No speech was recognised. Try again, a little closer to the microphone.")

    # ---------------- Upload ----------------
    with tab_up:
        st.markdown("Upload one or more recordings of customer complaints, such as saved calls or voice notes. "
                    "Each one is transcribed and given a topic.")
        files = st.file_uploader("Recordings", type=["wav", "mp3", "m4a", "ogg", "oga", "opus", "flac", "webm", "aac", "mp4"],
                                 accept_multiple_files=True, key="voice_files")
        st.caption("Works best when the recording is mostly the customer speaking. On two-person calls, the "
                   "agent's words are transcribed too, which can blur the topic. Longer recordings take longer to transcribe.")
        if files and len(files) > 20:
            st.warning("That's a lot of recordings at once. Large batches can be slow on the free hosting tier, so consider splitting them.")
        if files and st.button(f"Transcribe and classify {len(files)} recording{'s' if len(files) != 1 else ''}",
                               type="primary", key="voice_run"):
            whisper, problem = get_speech_model()
            if problem:
                st.error(problem)
            else:
                rows = []
                progress = st.progress(0.0, text="Starting...")
                for i, f in enumerate(files):
                    progress.progress(i / len(files), text=f"Transcribing {f.name} ({i + 1} of {len(files)})")
                    note = ""
                    try:
                        text, seconds = speech.transcribe(whisper, io.BytesIO(f.getvalue()))
                    except Exception as err:
                        text, seconds, note = "", 0.0, f"Couldn't read this file ({type(err).__name__})"
                    result = classify_text(text, model_choice, models) if text.strip() else {"ok": False}
                    if not note and not text.strip():
                        note = "No speech recognised"
                    elif not note and not result.get("ok"):
                        note = "No words the model recognises"
                    best = result["best"] if result.get("ok") else None
                    rows.append({"file": f.name, "seconds": seconds, "transcript": text, "topic": best,
                                 "confidence": float(result["shares"][best]) if best is not None else None,
                                 "note": note, "result": result})
                progress.progress(1.0, text=f"Done: {len(files)} recording{'s' if len(files) != 1 else ''} processed")
                st.session_state["voice_batch"] = {"model": model_choice, "rows": rows}

        batch = st.session_state.get("voice_batch")
        if batch:
            used, rows = batch["model"], batch["rows"]
            done = [r for r in rows if r["topic"] is not None]
            st.subheader("Results")
            if used != model_choice:
                st.caption(f"These results used {used}. Run them again to use {model_choice}.")
            def style_of(r):
                return (get_topic_style(r["topic"], used) or "") if r["topic"] is not None else ""
            table = pd.DataFrame([{
                "Recording": r["file"], "Length": fmt_duration(r["seconds"]),
                "Topic": f"{r['topic']}. {get_topic_label(r['topic'], used)}" if r["topic"] is not None else "",
                "Style": style_of(r), "Confidence": r["confidence"], "Transcript": r["transcript"], "Note": r["note"],
            } for r in rows])
            st.dataframe(table, width="stretch", hide_index=True, column_config={
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                "Transcript": st.column_config.TextColumn("Transcript", width="large"),
            })
            export = pd.DataFrame([{
                "recording": r["file"], "length_seconds": round(r["seconds"], 1), "transcript": r["transcript"],
                "topic_number": r["topic"], "topic_name": get_topic_label(r["topic"], used) if r["topic"] is not None else "",
                "writing_style": style_of(r), "confidence": round(r["confidence"], 3) if r["confidence"] is not None else None,
                "model": used, "note": r["note"],
            } for r in rows])
            st.download_button("⬇️ Download results (CSV)", data=export.to_csv(index=False).encode("utf-8"),
                               file_name="voice_complaint_topics.csv", mime="text/csv", key="voice_csv")
            if len(done) >= 2:
                st.subheader("Topics in these recordings")
                counts = pd.Series([r["topic"] for r in done]).value_counts()
                st.pyplot(topic_distribution_chart(counts, used, len(done)))
            st.subheader("Each recording")
            for r in rows:
                label = (f"{r['file']}: {get_topic_label(r['topic'], used)}" if r["topic"] is not None
                         else f"{r['file']}: {r['note']}")
                with st.expander(label):
                    if r["transcript"]:
                        st.markdown(f"<div class='complaint-quote'>{html.escape(r['transcript'])}</div>", unsafe_allow_html=True)
                    if r["result"].get("ok"):
                        show_classification(r["result"], used)
                    elif r["note"]:
                        st.caption(r["note"])

# ============================================================
# PAGE: Analyze & Report
# ============================================================
elif page == "Analyze & Report":
    page_header("Analyze & Report", "Upload a new file (or try a sample), run it through the topic models, and download a report.")

    if not models_ready:
        st.warning("Run `python src/modeling.py` from the terminal first to train the models.")
        st.stop()

    def run_upload_analysis(raw_texts, base_df, model_choice, source_name):
        tfidf_vectorizer, count_vectorizer, lda_model, nmf_model, _, _ = load_models()
        cleaned = [preprocess_text(t) for t in raw_texts]

        if model_choice == "NMF":
            vecs = tfidf_vectorizer.transform(cleaned)
            topic_dist = nmf_model.transform(vecs)
        else:
            vecs = count_vectorizer.transform(cleaned)
            topic_dist = lda_model.transform(vecs)

        assignments = np.argmax(topic_dist, axis=1)
        row_sums = topic_dist.sum(axis=1)
        shares = np.divide(topic_dist, row_sums[:, None], out=np.zeros_like(topic_dist), where=row_sums[:, None] > 0)
        result_df = base_df.copy()
        result_df["assigned_topic"] = [get_topic_label(i, model_choice) for i in assignments]
        result_df["confidence"] = shares[np.arange(len(assignments)), assignments].round(3)

        st.session_state["upload_analysis"] = {
            "df": result_df,
            "assignments": assignments,
            "topic_dist": topic_dist,
            "model_choice": model_choice,
            "source_name": source_name,
        }
        st.session_state.pop("report_bytes", None)
        st.session_state.pop("deck_bytes", None)

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a CSV or plain-text file",
            type=["csv", "txt"],
            help="CSV: pick which column holds the complaint text. TXT: one complaint per line.",
        )
    with col_sample:
        st.markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
        sample_clicked = st.button("🔁 Try a Sample", width="stretch")

    model_choice = st.selectbox("Model to use for topic assignment", MODEL_OPTIONS,
                                format_func=lambda m: MODEL_CAPTIONS[m], key="upload_model_choice")

    if sample_clicked:
        with st.spinner("Running the sample through the topic model..."):
            sample_df = pd.DataFrame({"text": SAMPLE_COMPLAINTS})
            run_upload_analysis(SAMPLE_COMPLAINTS, sample_df, model_choice, "sample_complaints.csv")

    elif uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            upload_df = pd.read_csv(uploaded_file)
            text_col = st.selectbox("Which column contains the complaint text?", upload_df.columns)
            raw_texts = upload_df[text_col].astype(str).tolist()
        else:
            raw_texts = [
                line.strip() for line in uploaded_file.read().decode("utf-8").splitlines()
                if line.strip()
            ]
            upload_df = pd.DataFrame({"text": raw_texts})

        st.success(f"Loaded {len(raw_texts)} documents from **{uploaded_file.name}**.")

        if st.button("Run Topic Analysis", type="primary"):
            with st.spinner("Cleaning, preprocessing, and assigning topics..."):
                run_upload_analysis(raw_texts, upload_df, model_choice, uploaded_file.name)

    if "upload_analysis" in st.session_state:
        results = st.session_state["upload_analysis"]
        result_df = results["df"]
        assignments = results["assignments"]
        model_choice = results["model_choice"]

        tfidf_vectorizer, count_vectorizer, lda_model, nmf_model, _, _ = load_models()
        if model_choice == "NMF":
            model, feature_names = nmf_model, tfidf_vectorizer.get_feature_names_out()
        else:
            model, feature_names = lda_model, count_vectorizer.get_feature_names_out()

        dist = pd.Series(assignments).value_counts().sort_index()

        st.divider()
        st.info(generate_executive_summary(dist, len(result_df), model_choice))

        st.subheader("Topic distribution for this file")
        st.pyplot(topic_distribution_chart(dist, model_choice, len(result_df)))

        st.subheader("Assignments")
        st.dataframe(
            result_df,
            width="stretch",
            height=min(420, 38 + 35 * len(result_df)),
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "confidence", help="Share of the complaint's topic weight on its assigned topic",
                    min_value=0.0, max_value=1.0, format="%.2f",
                ),
            },
        )
        st.download_button(
            "⬇️ Download all assignments (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="topic_assignments.csv",
            mime="text/csv",
        )

        st.subheader("Download report")
        st.caption("Both formats include the same distribution chart, executive summary, and topic keywords.")

        col_pdf, col_pptx = st.columns(2)
        with col_pdf:
            if st.button("📄 Generate PDF Report", width="stretch"):
                with st.spinner("Building PDF..."):
                    st.session_state["report_bytes"] = build_pdf_report(
                        result_df, assignments, results["topic_dist"],
                        model, feature_names, model_choice, results["source_name"],
                    )
            if "report_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Download PDF",
                    data=st.session_state["report_bytes"],
                    file_name="complaint_topic_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

        with col_pptx:
            if st.button("🖼️ Generate Slide Deck", width="stretch"):
                with st.spinner("Building slide deck..."):
                    st.session_state["deck_bytes"] = build_pptx_report(
                        result_df, assignments, results["topic_dist"],
                        model, feature_names, model_choice, results["source_name"],
                    )
            if "deck_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Download Slides (PPTX)",
                    data=st.session_state["deck_bytes"],
                    file_name="complaint_topic_deck.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    width="stretch",
                )
