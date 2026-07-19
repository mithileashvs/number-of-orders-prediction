import streamlit as st
import pandas as pd
import os
import io
import re
import html

from utils import load_data, preprocess_data, create_eda
from model import train_models

REQUIRED_COLUMNS = {"Order Date", "Order ID", "Sales", "Quantity", "Discount", "Profit"}

# ==========================================================
# FLEXIBLE COLUMN MATCHING FOR UPLOADED FILES
# ==========================================================
# Judges' CSVs won't always use our exact header names or number
# formats. This maps common real-world variants onto the canonical
# schema the pipeline expects, and cleans currency/percent formatting
# before handing off to preprocess_data().

COLUMN_ALIASES = {
    "Order Date": [
        "order date", "orderdate", "date", "order_date", "orddate",
        "purchase date", "transaction date", "sale date", "invoice date",
    ],
    "Order ID": [
        "order id", "orderid", "order_id", "order no", "order number",
        "orderno", "invoice id", "invoice no", "invoice number",
        "transaction id", "transactionid", "order code",
    ],
    "Sales": [
        "sales", "sale", "revenue", "amount", "sale amount",
        "total sales", "total amount", "order value", "sales amount",
        "net sales", "gross sales",
    ],
    "Quantity": [
        "quantity", "qty", "quantity ordered", "units", "unit count",
        "no of units", "units sold", "item quantity",
    ],
    "Discount": [
        "discount", "disc", "discount rate", "discount %",
        "discount percent", "discount pct", "discountratio",
    ],
    "Profit": [
        "profit", "margin", "net profit", "gross profit", "earnings",
        "profit amount",
    ],
}


def _norm_key(text: str) -> str:
    """Lowercase and strip everything but letters/numbers, so 'Order_Date',
    'order date', and 'OrderDate' all collapse to the same key."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


_ALIAS_LOOKUP = {}
for _canonical, _aliases in COLUMN_ALIASES.items():
    _ALIAS_LOOKUP[_norm_key(_canonical)] = _canonical
    for _alias in _aliases:
        _ALIAS_LOOKUP[_norm_key(_alias)] = _canonical


def normalize_uploaded_columns(df: pd.DataFrame):
    """Rename columns to the canonical schema wherever a confident alias
    match exists, and clean common formatting issues (currency symbols,
    thousands separators, percent signs) in the numeric columns.

    Returns (cleaned_df, rename_map) where rename_map only contains
    columns that were actually renamed (original -> canonical).
    """
    rename_map = {}
    for col in df.columns:
        canonical = _ALIAS_LOOKUP.get(_norm_key(col))
        if canonical and canonical not in rename_map.values():
            rename_map[col] = canonical

    df = df.rename(columns=rename_map)

    # Strip currency symbols / thousands separators on non-numeric fields.
    # (Checked with is_numeric_dtype rather than `dtype == object` so this
    # works across pandas versions, including pandas >=2.x/3.x where plain
    # text columns get a dedicated string dtype instead of `object`.)
    for col in ("Sales", "Profit", "Quantity"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            cleaned = df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
            df[col] = pd.to_numeric(cleaned.replace("", None), errors="coerce")

    # Discount may arrive as "20%" (-> 0.20) or already as a 0-1 fraction.
    if "Discount" in df.columns and not pd.api.types.is_numeric_dtype(df["Discount"]):
        as_str = df["Discount"].astype(str).str.strip()
        has_percent = as_str.str.contains("%", na=False)
        cleaned = pd.to_numeric(
            as_str.str.replace(r"[^0-9.\-]", "", regex=True).replace("", None),
            errors="coerce",
        )
        df["Discount"] = cleaned.where(~has_percent, cleaned / 100)

    renamed_only = {k: v for k, v in rename_map.items() if k != v}
    return df, renamed_only

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Number of Orders Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# DESIGN TOKENS
# ==========================================================
# Theme: "Night Freight" — a late-shift logistics command center.
# Deep ink-navy field, radar-teal for movement/in-transit signals,
# warehouse-amber for volume/orders, soft coral for variance/alerts.

INK        = "#080C16"
INK_2      = "#111A2E"
GLASS      = "rgba(255,255,255,0.055)"
GLASS_BRD  = "rgba(255,255,255,0.10)"
TEXT       = "#EAEEF7"
MUTED      = "#8C96AF"
TEAL       = "#2FD9C4"
AMBER      = "#F2B156"
CORAL      = "#F2708A"
VIOLET     = "#8B87F0"

ACCENTS = [TEAL, AMBER, VIOLET, CORAL]

# ==========================================================
# GLOBAL STYLE INJECTION
# ==========================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {{
        --noop-ink: {INK};
        --noop-ink2: {INK_2};
        --noop-glass: {GLASS};
        --noop-glass-brd: {GLASS_BRD};
        --noop-text: {TEXT};
        --noop-muted: {MUTED};
        --noop-teal: {TEAL};
        --noop-amber: {AMBER};
        --noop-coral: {CORAL};
        --noop-violet: {VIOLET};
    }}

    #MainMenu, footer {{ visibility: hidden; }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}
    [data-testid="stToolbar"] {{
        visibility: hidden;
    }}

    /* Sidebar is permanent — remove all controls that could hide it */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}
    [data-testid="stSidebar"] {{
        min-width: 300px !important;
        max-width: 300px !important;
        width: 300px !important;
        transform: none !important;
        visibility: visible !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        width: 300px !important;
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: var(--noop-text);
    }}

    .stApp {{
        background:
            radial-gradient(1100px 620px at 8% -8%, rgba(47,217,196,0.14), transparent 60%),
            radial-gradient(900px 560px at 100% 8%, rgba(242,177,86,0.10), transparent 55%),
            radial-gradient(800px 700px at 50% 115%, rgba(139,135,240,0.08), transparent 60%),
            linear-gradient(160deg, var(--noop-ink) 0%, var(--noop-ink2) 100%);
        background-attachment: fixed;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }}

    /* ---------------- Sidebar ---------------- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(17,26,46,0.92), rgba(8,12,22,0.96));
        border-right: 1px solid var(--noop-glass-brd);
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.4rem;
    }}

    .noop-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0 0.4rem 1.1rem 0.4rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--noop-glass-brd);
    }}
    .noop-brand-icon {{
        width: 38px; height: 38px;
        border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px;
        background: linear-gradient(145deg, rgba(47,217,196,0.25), rgba(139,135,240,0.15));
        border: 1px solid var(--noop-glass-brd);
    }}
    .noop-brand-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 15px;
        line-height: 1.15;
        color: var(--noop-text);
    }}
    .noop-brand-sub {{
        font-size: 10.5px;
        letter-spacing: 1.2px;
        color: var(--noop-muted);
        text-transform: uppercase;
    }}

    .noop-nav-eyebrow {{
        font-size: 10.5px;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: var(--noop-muted);
        margin: 0.2rem 0 0.6rem 0.4rem;
    }}

    .noop-nav-active {{
        display: flex; align-items: center; justify-content: flex-start; gap: 10px;
        padding: 0.62rem 0.9rem;
        margin-bottom: 0.35rem;
        border-radius: 12px;
        background: linear-gradient(120deg, rgba(47,217,196,0.16), rgba(139,135,240,0.10));
        border: 1px solid rgba(47,217,196,0.35);
        font-size: 14px;
        font-weight: 500;
        color: var(--noop-text);
        box-shadow: 0 0 0 1px rgba(47,217,196,0.05), 0 6px 18px rgba(47,217,196,0.08);
    }}

    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        text-align: left;
        justify-content: flex-start !important;
        background: transparent;
        border: 1px solid transparent;
        color: var(--noop-muted);
        font-weight: 500;
        font-size: 14px;
        padding: 0.62rem 0.9rem;
        border-radius: 12px;
        margin-bottom: 0.35rem;
        transition: all 0.18s ease;
    }}
    section[data-testid="stSidebar"] .stButton > button > div {{
        width: 100%;
        justify-content: flex-start !important;
    }}
    section[data-testid="stSidebar"] .stButton > button p {{
        text-align: left !important;
        width: 100%;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255,255,255,0.05);
        border-color: var(--noop-glass-brd);
        color: var(--noop-text);
    }}
    section[data-testid="stSidebar"] .stButton > button:focus:not(:active) {{
        color: var(--noop-text);
    }}

    /* ---------------- Sidebar dataset uploader ---------------- */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
        background: rgba(255,255,255,0.04);
        border: 1px dashed var(--noop-glass-brd);
        border-radius: 12px;
        padding: 0.6rem;
    }}
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: none !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
        font-size: 12px !important;
    }}
    .noop-data-badge {{
        display: block;
        padding: 6px 10px;
        border-radius: 10px;
        font-size: 11px;
        line-height: 1.5;
        margin: 0.5rem 0 0.7rem 0;
        border: 1px solid;
    }}
    .noop-data-badge b {{
        font-weight: 600;
    }}
    .noop-data-badge-sub {{
        font-size: 10px;
        opacity: 0.7;
        margin-top: 1px;
        cursor: help;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .noop-data-badge.uploaded {{
        background: rgba(47,217,196,0.10);
        border-color: rgba(47,217,196,0.32);
        color: var(--noop-teal);
    }}
    .noop-data-badge.default {{
        background: rgba(255,255,255,0.04);
        border-color: var(--noop-glass-brd);
        color: var(--noop-muted);
    }}
    .noop-data-badge.error {{
        background: rgba(242,112,138,0.10);
        border-color: rgba(242,112,138,0.35);
        color: var(--noop-coral);
    }}

    .noop-sidebar-footer {{
        position: relative;
        margin-top: 1.4rem;
        padding-top: 1rem;
        border-top: 1px solid var(--noop-glass-brd);
        font-size: 11px;
        color: var(--noop-muted);
        line-height: 1.6;
    }}
    .noop-sidebar-footer b {{ color: var(--noop-text); }}

    /* ---------------- Headers ---------------- */
    .noop-eyebrow {{
        display: inline-block;
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--noop-teal);
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .noop-page-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 34px;
        line-height: 1.2;
        color: var(--noop-text);
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }}
    .noop-page-sub {{
        font-size: 14.5px;
        color: var(--noop-muted);
        margin-bottom: 1.6rem;
        max-width: 640px;
    }}
    .noop-divider {{
        height: 1px;
        border: none;
        margin: 1.6rem 0;
        background: linear-gradient(90deg, rgba(47,217,196,0.35), rgba(255,255,255,0.03) 60%);
    }}

    /* ---------------- Metric cards ---------------- */
    .noop-metric-card {{
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
        border: 1px solid var(--noop-glass-brd);
        border-radius: 20px;
        padding: 20px 20px 18px 20px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 10px 34px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.06);
        animation: noopFadeUp 0.55s cubic-bezier(.2,.7,.3,1) both;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
    }}
    .noop-metric-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(255,255,255,0.2);
        box-shadow: 0 16px 42px rgba(0,0,0,0.48), inset 0 1px 0 rgba(255,255,255,0.09);
    }}
    .noop-icon-wrap {{
        position: relative;
        width: 42px; height: 42px;
        border-radius: 13px;
        display: flex; align-items: center; justify-content: center;
        font-size: 19px;
        margin-bottom: 14px;
    }}
    .noop-ping {{
        position: absolute; inset: 0;
        border-radius: 13px;
        animation: noopPing 2.6s cubic-bezier(0,0,0.2,1) infinite;
    }}
    .noop-metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 600;
        color: var(--noop-text);
        letter-spacing: -0.5px;
        line-height: 1.1;
    }}
    .noop-metric-label {{
        font-family: 'Inter', sans-serif;
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        color: var(--noop-muted);
        margin-top: 7px;
    }}

    @keyframes noopFadeUp {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes noopPing {{
        0%   {{ transform: scale(0.75); opacity: 0.55; }}
        70%  {{ transform: scale(1.9); opacity: 0; }}
        100% {{ opacity: 0; }}
    }}

    /* ---------------- Glass panel (bordered containers) ---------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.012));
        border: 1px solid var(--noop-glass-brd) !important;
        border-radius: 20px !important;
        padding: 4px 6px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.30);
    }}

    .noop-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(47,217,196,0.12);
        border: 1px solid rgba(47,217,196,0.32);
        color: var(--noop-teal);
        font-size: 12.5px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}

    .noop-tag {{
        display: inline-block;
        padding: 5px 11px;
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--noop-glass-brd);
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: var(--noop-text);
        margin: 3px 6px 3px 0;
    }}

    /* dataframes / images */
    [data-testid="stDataFrame"], .stImage img {{
        border-radius: 14px !important;
        overflow: hidden;
        border: 1px solid var(--noop-glass-brd);
    }}

    ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.12); border-radius: 8px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.2); }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# COMPONENT HELPERS
# ==========================================================

def page_header(eyebrow: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div>
            <div class="noop-eyebrow">{eyebrow}</div>
            <div class="noop-page-title">{title}</div>
            <div class="noop-page-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(icon: str, label: str, value: str, accent: str, delay: float = 0.0):
    st.markdown(
        f"""
        <div class="noop-metric-card" style="animation-delay:{delay}s;">
            <div class="noop-icon-wrap" style="background:{accent}22; border:1px solid {accent}40;">
                <span class="noop-ping" style="background:{accent}33;"></span>
                <span style="position:relative; z-index:1;">{icon}</span>
            </div>
            <div class="noop-metric-value">{value}</div>
            <div class="noop-metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown("<hr class='noop-divider'/>", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR — BRANDING
# ==========================================================

st.sidebar.markdown(
    """
    <div class="noop-brand">
        <div class="noop-brand-icon">📦</div>
        <div>
            <div class="noop-brand-title">Order Forecast</div>
            <div class="noop-brand-sub">Control Center</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# SIDEBAR — DATASET UPLOAD
# ==========================================================
# Judges / reviewers can drop in their own Superstore-format CSV here.
# If nothing is uploaded, or the file doesn't match the expected schema,
# the app falls back to the bundled sample dataset.

st.sidebar.markdown("<div class='noop-nav-eyebrow'>Dataset</div>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload your own CSV to verify",
    type=["csv"],
    help=(
        "Optional. Upload a transaction-level CSV. Column names are matched "
        "flexibly — 'order_date', 'OrderDate', and 'Purchase Date' all work — "
        f"but the file needs equivalents of: {', '.join(sorted(REQUIRED_COLUMNS))}. "
        "Currency symbols, commas, and % signs in numeric columns are handled "
        "automatically."
    ),
    label_visibility="collapsed",
)

data_error = None
rename_map = {}

if uploaded_file is not None:
    try:
        candidate_df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), encoding="latin1")
        candidate_df, rename_map = normalize_uploaded_columns(candidate_df)
        missing = REQUIRED_COLUMNS - set(candidate_df.columns)
        if missing:
            data_error = f"Missing columns: {', '.join(sorted(missing))}"
            raw_df = load_data()
            data_source = "default"
        else:
            raw_df = candidate_df
            data_source = "uploaded"
    except Exception as exc:
        data_error = f"Could not read that file ({exc})"
        raw_df = load_data()
        data_source = "default"
else:
    raw_df = load_data()
    data_source = "default"

try:
    df = preprocess_data(raw_df)
except Exception as exc:
    if data_source == "uploaded":
        data_error = f"Could not process that file ({exc})"
    raw_df = load_data()
    data_source = "default"
    rename_map = {}
    df = preprocess_data(raw_df)

if data_error:
    st.sidebar.markdown(
        f"<div class='noop-data-badge error'>⚠️ {data_error} — showing sample data instead.</div>",
        unsafe_allow_html=True,
    )
elif data_source == "uploaded":
    sub_line = ""
    if rename_map:
        pairs_title = "; ".join(f"{orig} -> {canon}" for orig, canon in rename_map.items())
        n = len(rename_map)
        sub_line = (
            f"<div class='noop-data-badge-sub' title=\"{html.escape(pairs_title)}\">"
            f"🔗 {n} column{'s' if n != 1 else ''} auto-matched</div>"
        )
    st.sidebar.markdown(
        f"<div class='noop-data-badge uploaded'>📤 <b>{uploaded_file.name}</b> · "
        f"{len(df):,} rows{sub_line}</div>",
        unsafe_allow_html=True,
    )

# ==========================================================
# PIPELINE — regenerate EDA charts and retrain every model on
# whichever dataset is currently active (default or uploaded).
# ==========================================================

with st.spinner("Analyzing dataset and training models..."):
    create_eda(df)
    ml_results = train_models(df)

best_model_name = max(ml_results, key=lambda name: ml_results[name]["R2"])
best_r2 = ml_results[best_model_name]["R2"]

# ==========================================================
# SIDEBAR — NAVIGATION
# ==========================================================

PAGES = [
    ("home", "🏠", "Home"),
    ("dataset", "📂", "Dataset"),
    ("eda", "📊", "EDA"),
    ("models", "🤖", "Model Comparison"),
    ("prediction", "📈", "Prediction"),
    ("about", "ℹ️", "About"),
]

if "noop_page" not in st.session_state:
    st.session_state.noop_page = "home"

st.sidebar.markdown("<div class='noop-nav-eyebrow'>Navigate</div>", unsafe_allow_html=True)

for key, icon, label in PAGES:
    if st.session_state.noop_page == key:
        st.sidebar.markdown(
            f"<div class='noop-nav-active'>{icon}&nbsp;&nbsp;&nbsp;{label}</div>",
            unsafe_allow_html=True,
        )
    else:
        if st.sidebar.button(f"{icon}   {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.noop_page = key
            st.rerun()

st.sidebar.markdown(
    """
    <div class="noop-sidebar-footer">
        <b>MITHILEASH V S</b><br/>
        SPARKIIT ML Internship · 2026<br/>
        mithileashvs@gmail.com
    </div>
    """,
    unsafe_allow_html=True,
)

page_key = st.session_state.noop_page

# ==========================================================
# HOME PAGE
# ==========================================================

if page_key == "home":

    page_header(
        "Overview",
        "Number of Orders Prediction",
        "Forecasting daily customer order volume from historical Superstore sales activity using regression models.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("📊", "Records", f"{len(df):,}", TEAL, 0.00)
    with c2:
        metric_card("🧮", "Models Trained", f"{len(ml_results)}", AMBER, 0.08)
    with c3:
        metric_card("🏆", "Best Model", best_model_name, VIOLET, 0.16)
    with c4:
        metric_card("📈", "R² Score", f"{best_r2:.3f}", CORAL, 0.24)

    divider()

    left, right = st.columns([2, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                """
                <div style="padding: 1.2rem 1.3rem 0.4rem 1.3rem;">
                    <span class="noop-badge">✦ Machine Learning Internship Project</span>
                    <h4 style="font-family:'Space Grotesk',sans-serif; margin-top:1.1rem; margin-bottom:0.6rem;">
                        Project Overview
                    </h4>
                    <p style="color:#C6CCDC; font-size:14.5px; line-height:1.7;">
                        This project predicts <b>daily customer order counts</b> from historical
                        <b>Superstore sales data</b> using machine learning regression models.
                        The raw transaction log was aggregated by <b>Order Date</b> into a daily
                        order-count series, then used to train and evaluate several models to find
                        the strongest predictor of demand.
                    </p>
                    <h4 style="font-family:'Space Grotesk',sans-serif; margin-top:1.4rem; margin-bottom:0.7rem;">
                        Project Workflow
                    </h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            steps = [
                "Load Dataset", "Clean Data", "Exploratory Data Analysis",
                "Feature Engineering", "Train Multiple Models",
                "Evaluate Performance", "Select Best Model", "Visualize Predictions",
            ]
            tags_html = "".join(f"<span class='noop-tag'>{s}</span>" for s in steps)
            st.markdown(f"<div style='padding: 0 1.3rem 1.3rem 1.3rem;'>{tags_html}</div>", unsafe_allow_html=True)

        st.write("")
        st.write("**Project Topic:** Number of Orders Prediction")
        st.write("**Full Name:** MITHILEASH V S")
        st.write("**Registered Email:** mithileashvs@gmail.com")

    with right:
        with st.container(border=True):
            st.markdown(
                """
                <div style="padding: 1.2rem 1.3rem;">
                    <h4 style="font-family:'Space Grotesk',sans-serif; margin-top:0; margin-bottom:0.9rem;">
                        🛠 Technologies Used
                    </h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            tech = ["Python", "Streamlit", "Pandas", "Scikit-learn", "XGBoost", "Matplotlib"]
            tech_html = "".join(f"<span class='noop-tag'>{t}</span>" for t in tech)
            st.markdown(f"<div style='padding: 0 1.3rem 1.3rem 1.3rem;'>{tech_html}</div>", unsafe_allow_html=True)

# ==========================================================
# DATASET PAGE
# ==========================================================

elif page_key == "dataset":

    page_header(
        "Data",
        "Dataset Overview",
        "A snapshot of the aggregated daily order dataset used to train every model.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("🧾", "Rows", f"{df.shape[0]:,}", TEAL, 0.0)
    with c2:
        metric_card("📐", "Columns", f"{df.shape[1]}", AMBER, 0.08)
    with c3:
        metric_card("⚠️", "Missing Values", f"{int(df.isnull().sum().sum())}", CORAL, 0.16)

    divider()

    with st.container(border=True):
        st.markdown("<div style='padding: 1.1rem 1.2rem 0.4rem 1.2rem;'><h4 style=\"font-family:'Space Grotesk',sans-serif; margin:0;\">Dataset Preview</h4></div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=350)
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("<div style='padding: 1.1rem 1.2rem 0.2rem 1.2rem;'><h4 style=\"font-family:'Space Grotesk',sans-serif; margin:0;\">Statistical Summary</h4></div>", unsafe_allow_html=True)
            st.dataframe(df.describe(), use_container_width=True)
            st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown("<div style='padding: 1.1rem 1.2rem 0.2rem 1.2rem;'><h4 style=\"font-family:'Space Grotesk',sans-serif; margin:0;\">Column Names</h4></div>", unsafe_allow_html=True)
            cols_html = "".join(f"<span class='noop-tag'>{c}</span>" for c in df.columns.tolist())
            st.markdown(f"<div style='padding: 0 1.2rem 1rem 1.2rem;'>{cols_html}</div>", unsafe_allow_html=True)
            st.markdown("<div style='padding: 0.2rem 1.2rem 0 1.2rem;'><h4 style=\"font-family:'Space Grotesk',sans-serif; margin:0 0 0.5rem 0;\">Data Types</h4></div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(df.dtypes, columns=["Type"]), use_container_width=True)
            st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

# ==========================================================
# EDA PAGE
# ==========================================================

elif page_key == "eda":

    page_header(
        "Analysis",
        "Exploratory Data Analysis",
        "Visualizations generated during preprocessing, exploring seasonality, correlation, and drivers of order volume.",
    )

    left, right = st.columns(2, gap="large")

    with left:
        if os.path.exists("outputs/correlation_heatmap.png"):
            with st.container(border=True):
                st.markdown("<div style='padding:0.9rem 1rem 0 1rem;'><span class='noop-eyebrow'>Signal</span></div>", unsafe_allow_html=True)
                st.image("outputs/correlation_heatmap.png", caption="Correlation Heatmap", use_container_width=True)
            st.write("")
        if os.path.exists("outputs/monthly_orders.png"):
            with st.container(border=True):
                st.markdown("<div style='padding:0.9rem 1rem 0 1rem;'><span class='noop-eyebrow'>Seasonality</span></div>", unsafe_allow_html=True)
                st.image("outputs/monthly_orders.png", caption="Average Monthly Orders", use_container_width=True)

    with right:
        if os.path.exists("outputs/orders_over_time.png"):
            with st.container(border=True):
                st.markdown("<div style='padding:0.9rem 1rem 0 1rem;'><span class='noop-eyebrow'>Trend</span></div>", unsafe_allow_html=True)
                st.image("outputs/orders_over_time.png", caption="Orders Over Time", use_container_width=True)
            st.write("")
        if os.path.exists("outputs/feature_importance.png"):
            with st.container(border=True):
                st.markdown("<div style='padding:0.9rem 1rem 0 1rem;'><span class='noop-eyebrow'>Drivers</span></div>", unsafe_allow_html=True)
                st.image("outputs/feature_importance.png", caption="Feature Importance", use_container_width=True)

# ==========================================================
# MODEL COMPARISON
# ==========================================================

elif page_key == "models":

    page_header(
        "Benchmark",
        "Machine Learning Model Comparison",
        "Three regression models were trained on the same feature set and evaluated head-to-head.",
    )

    results = ml_results

    st.markdown(
        "<span class='noop-badge'>✓ Training completed successfully</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    result_df = pd.DataFrame(results).T

    with st.container(border=True):
        st.markdown("<div style='padding: 1.1rem 1.2rem 0.2rem 1.2rem;'><h4 style=\"font-family:'Space Grotesk',sans-serif; margin:0;\">Model Performance</h4></div>", unsafe_allow_html=True)
        st.dataframe(result_df, use_container_width=True)
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("🧮", "Models Trained", f"{len(results)}", TEAL, 0.0)
    with c2:
        metric_card("🏆", "Best Model", best_model_name, AMBER, 0.08)
    with c3:
        metric_card("📐", "Best R²", f"{best_r2:.3f}", VIOLET, 0.16)

    st.write("")

    with st.container(border=True):
        st.markdown(
            """
            <div style="padding: 1.1rem 1.3rem;">
                <h4 style="font-family:'Space Grotesk',sans-serif; margin-top:0;">Evaluation Metrics</h4>
                <p style="color:#C6CCDC; font-size:14px; line-height:1.8; margin-bottom:0;">
                    <b>MAE</b> — Mean Absolute Error. Lower is better.<br/>
                    <b>RMSE</b> — Root Mean Squared Error. Lower is better.<br/>
                    <b>R² Score</b> — Variance explained. Higher is better.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page_key == "prediction":

    page_header(
        "Results",
        "Prediction Results",
        "How the best-performing model's forecasts compare against actual daily order counts.",
    )

    st.markdown(f"<span class='noop-badge'>🏆 Best performing model: {best_model_name}</span>", unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("⚙️", "Algorithm", best_model_name, TEAL, 0.0)
    with c2:
        metric_card("📐", "R² Score", f"{best_r2:.3f}", AMBER, 0.08)
    with c3:
        metric_card("✅", "Status", "Ready", VIOLET, 0.16)

    divider()

    with st.container(border=True):
        st.markdown("<div style='padding: 1.1rem 1.2rem 0.2rem 1.2rem;'><h4 style=\"font-family:'Space Grotesk',sans-serif; margin:0;\">Actual vs Predicted Orders</h4></div>", unsafe_allow_html=True)
        if os.path.exists("outputs/actual_vs_predicted.png"):
            st.image("outputs/actual_vs_predicted.png", use_container_width=True)
        else:
            st.warning("Prediction graph not found.")
        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

    st.write("")

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="padding: 1.1rem 1.3rem;">
                <h4 style="font-family:'Space Grotesk',sans-serif; margin-top:0;">Prediction Summary</h4>
                <p style="color:#C6CCDC; font-size:14.5px; line-height:1.75; margin-bottom:0;">
                    The {best_model_name} model produced the highest prediction accuracy of the
                    three candidates, with an R² of {best_r2:.3f}. The scatter plot compares
                    actual daily orders with predicted values — the closer the points sit to the
                    ideal trend line, the better the model captures the relationship between the
                    engineered features and daily order counts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page_key == "about":

    page_header(
        "Reference",
        "About This Project",
        "Workflow, algorithms, tooling, and authorship behind the Number of Orders Prediction app.",
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                """
                <div style="padding: 1.2rem 1.3rem;">
                    <p style="color:#C6CCDC; font-size:14.5px; line-height:1.75;">
                        This machine learning application predicts the number of customer orders
                        using historical Superstore sales data, aggregated into a daily time series.
                    </p>
                    <h4 style="font-family:'Space Grotesk',sans-serif; margin-bottom:0.6rem;">Workflow</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            steps = [
                "Load Dataset", "Clean Data", "Feature Engineering",
                "Train Machine Learning Models", "Compare Model Performance",
                "Select Best Model", "Predict Daily Orders",
            ]
            for i, s in enumerate(steps, start=1):
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:12px; padding: 7px 1.3rem;">
                        <div style="font-family:'JetBrains Mono',monospace; font-size:12px; color:{TEAL};
                                    width:26px; height:26px; border-radius:8px; display:flex;
                                    align-items:center; justify-content:center;
                                    background:rgba(47,217,196,0.12); border:1px solid rgba(47,217,196,0.3);">
                            {i:02d}
                        </div>
                        <div style="font-size:14px; color:#DBE0EC;">{s}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown(
                """
                <div style="padding: 1.2rem 1.3rem;">
                    <h4 style="font-family:'Space Grotesk',sans-serif; margin-top:0;">Algorithms Used</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            algos = ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"]
            algo_html = "".join(f"<span class='noop-tag'>{a}</span>" for a in algos)
            st.markdown(f"<div style='padding: 0 1.3rem 1.2rem 1.3rem;'>{algo_html}</div>", unsafe_allow_html=True)

        st.write("")

        with st.container(border=True):
            st.markdown(
                """
                <div style="padding: 1.2rem 1.3rem;">
                    <h4 style="font-family:'Space Grotesk',sans-serif; margin-top:0;">Python Libraries</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            libs = ["Pandas", "NumPy", "Scikit-Learn", "XGBoost", "Matplotlib", "Streamlit"]
            lib_html = "".join(f"<span class='noop-tag'>{l}</span>" for l in libs)
            st.markdown(f"<div style='padding: 0 1.3rem 1.2rem 1.3rem;'>{lib_html}</div>", unsafe_allow_html=True)

        st.write("")

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="padding: 1.2rem 1.3rem;">
                    <h4 style="font-family:'Space Grotesk',sans-serif; margin-top:0; margin-bottom:0.5rem;">Developed By</h4>
                    <p style="font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:16px; margin:0; color:{TEAL};">
                        MITHILEASH V S
                    </p>
                    <p style="color: var(--noop-muted); font-size:12.5px; margin:2px 0 0 0;">
                        SPARKIIT Machine Learning Internship · 2026
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==========================================================
# FOOTER
# ==========================================================

divider()

st.markdown(
    """
    <div style="text-align:center; color:var(--noop-muted); font-size:12px; padding-bottom: 1rem;">
        © 2026 &nbsp;·&nbsp; Number of Orders Prediction &nbsp;·&nbsp; Machine Learning Internship Project
    </div>
    """,
    unsafe_allow_html=True,
)