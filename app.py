# -----------------------------
# Asset Dashboard (Streamlit) — All Assets
# -----------------------------
# HOW TO RUN:
#   streamlit run app.py
# -----------------------------
import io
import os
import base64
import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
import altair as alt
from pathlib import Path
import urllib.parse  # for mailto links & web deeplinks
from typing import Optional, Union

# For building .eml (HTML email) files
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import html as _html

# ML deps
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import hashlib

# Page setup
st.set_page_config(page_title="Asset Dashboard", layout="wide")

# ===== Global styles for a cleaner look =====
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.hr { border: 0; border-top: 1px solid rgba(180,180,180,.25); margin: 0.75rem 0 1rem 0; }
[data-testid="stDataFrame"] thead tr { position: sticky; top: 0; z-index: 1; }
.stExpander { border: 1px solid rgba(150,150,150,.25); border-radius: 8px; }
.stExpander > div > div { padding-top: .25rem; }
.action-bar { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; margin: .25rem 0 .75rem 0; }
.card { border: 1px solid rgba(150,150,150,.25); border-radius: 10px; padding: .75rem .9rem; background: rgba(120,120,120,.05); margin-bottom: .75rem; }
[data-testid="stDataFrame"] tbody tr td { padding-top: 6px; padding-bottom: 6px; }
h4, h5 { margin-bottom: .3rem; }
.section-caption { color: #888; font-size: .9rem; margin-top: -.35rem; }
.app-topbar { display: flex; align-items: center; gap: 12px; margin: 4px 0 12px 0; }
.app-logo { height: 44px; width: auto; object-fit: contain; transform: translateY(2px); }
.app-title { margin: 0; line-height: 1.1; font-weight: 700; }
@media (min-width: 900px)  { .app-title { font-size: 2.0rem; } }
@media (min-width: 1200px) { .app-title { font-size: 2.2rem; } }
</style>
""", unsafe_allow_html=True)

# ===== Simple UI helpers =====
def section_header(title: str, caption: Optional[str] = None):
    st.markdown(f"### {title}")
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="hr">', unsafe_allow_html=True)

def card():
    from contextlib import contextmanager
    @contextmanager
    def _card():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        yield
        st.markdown('</div>', unsafe_allow_html=True)
    return _card()

# --- Logo + title header ---
def resolve_logo(filename: str) -> Optional[Path]:
    here = Path(__file__).parent.resolve()
    cwd = Path.cwd().resolve()
    for p in [
        here / filename,
        here / "assets" / filename,
        here / "static" / filename,
        cwd / filename,
        cwd / "assets" / filename,
        cwd / "static" / filename,
    ]:
        if p.exists():
            return p
    return None

def to_b64(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None

_theme = st.get_option("theme.base")
is_dark = (_theme == "dark")
LIGHT_LOGO = resolve_logo("assetslogo_light.png")
DARK_LOGO  = resolve_logo("assetslogo_dark.png")
logo_path = DARK_LOGO if (is_dark and DARK_LOGO) else LIGHT_LOGO
logo_src  = None
if logo_path:
    b64 = to_b64(logo_path)
    if b64:
        logo_src = f"data:image/png;base64,{b64}"

if logo_src:
    st.markdown(
        f"""
        <div class="app-topbar">
          <img src="{logo_src}" alt="logo" class="app-logo" />
          <h1 class="app-title">📦 Asset Dashboard (All Assets)</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="app-topbar">
          <h1 class="app-title">📦 Asset Dashboard (All Assets)</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Defaults
DEFAULT_EXCEL_PATH = "assetlist.xlsx"
DEFAULT_SHEET_NAME = "Sheet1"

# Auto-detection candidates
USER_CANDIDATES = ["USER","ASSIGNEE","ASSIGNED_TO","OWNER","USERNAME","EMPLOYEE","EMAIL","EMPNAME","EMPID"]
AGE_CANDIDATES = ["AGE","ASSET_AGE_YEARS"]
PURCHASE_DATE_CANDIDATES = ["PURCHASE_DATE","BUY_DATE","PO_DATE","INVOICE_DATE","PROCUREMENT_DATE","PURDATE"]
ASSET_ID_CANDIDATES = ["ASSETNO","ASSET_NO","ASSET_NUMBER","ASSET_ID"]
SERIAL_ID_CANDIDATES = ["SRNO","SERIAL","SERIAL_NO","SERIALNUMBER"]
EMAIL_CANDIDATES = ["EMAIL","MAIL","USER_EMAIL","WORK_EMAIL","E_MAIL"]

PREFERRED_DISPLAY_COLUMNS = [
    "ASSETNO","SRNO","MAKE","MODEL",
    "EMPNAME","EMPID",
    "SUBCAT",
    "CITY","COUNTRY",
    "SUBLOC","LEVEL","SEATNO","LOCNAME","STATUS","AGE","USER","ITEMTYPE"
]

# ---------- Helpers: normalization and detection ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.upper()
    )
    return df

def normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s.replace({"nan": np.nan, "": np.nan})
            df[col] = s
    return df

def standardize_value_case(df: pd.DataFrame, columns: list[str], mode: str = "upper") -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            s = df[col]
            mask = s.notna()
            if mode == "upper":
                df.loc[mask, col] = s[mask].astype(str).str.strip().str.upper()
            elif mode == "lower":
                df.loc[mask, col] = s[mask].astype(str).str.strip().str.lower()
            elif mode == "title":
                df.loc[mask, col] = s[mask].astype(str).str.strip().str.title()
    return df

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def guess_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None

def compute_age_from_purchase_date(df: pd.DataFrame, purchase_col: str, out_col: str = "AGE") -> pd.DataFrame:
    if purchase_col not in df.columns:
        return df
    df[purchase_col] = pd.to_datetime(df[purchase_col], errors="coerce")
    today = pd.Timestamp.today()
    age_years = (today - df[purchase_col]).dt.days / 365.25
    df[out_col] = np.floor(age_years).astype("Int64")
    return df

# ---------- Filter utilities ----------
def options_for(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).str.strip().unique().tolist()
    return sorted([v for v in vals if v != ""])

def apply_list_filter(df_in: pd.DataFrame, col: str, selected: list) -> pd.DataFrame:
    if not selected or col not in df_in.columns:
        return df_in
    series = df_in[col]
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
        series = series.astype(str)
    right = [str(x).strip() for x in selected]
    return df_in[series.isin(right)]

def keyword_search(df_in: pd.DataFrame, query: str, cols: list[str]) -> pd.DataFrame:
    """
    NaN-safe keyword search across selected columns.
    - Replaces NaN with empty string before searching
    - Avoids accidental matches on the literal string "nan"
    """
    if not query or not str(query).strip():
        return df_in
    q = str(query).strip()
    cols_to_search = [c for c in cols if c in df_in.columns] or list(df_in.columns)
    mask = None
    for c in cols_to_search:
        series = df_in[c]
        if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
            series = series.astype(str)
        series = series.where(series.notna(), "").astype(str).str.strip()
        series = series.replace("nan", "", regex=False)
        col_mask = series.str.contains(q, case=False, na=False)
        mask = col_mask if mask is None else (mask | col_mask)
    return df_in[mask] if mask is not None else df_in

def users_with_multiple_assets(df: pd.DataFrame, user_col: Optional[str], asset_id_col: Optional[str]) -> pd.DataFrame:
    if (not user_col) or (not asset_id_col) or (user_col not in df.columns) or (asset_id_col not in df.columns):
        return pd.DataFrame(columns=[user_col or "USER", "ASSET_COUNT"])
    counts = (
        df.groupby(user_col, dropna=False)[asset_id_col]
        .nunique()
        .reset_index(name="ASSET_COUNT")
    )
    return counts[counts["ASSET_COUNT"] > 1].sort_values("ASSET_COUNT", ascending=False)

def filter_age_threshold(df: pd.DataFrame, age_col: Optional[str], threshold: int) -> pd.DataFrame:
    if not age_col or age_col not in df.columns:
        return df.head(0)
    return df[df[age_col].fillna(-1) >= threshold]

def export_df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# ---------- Email helpers ----------
def _first_non_null(series: pd.Series) -> Optional[str]:
    if series is None:
        return None
    vals = pd.Series(series).dropna().astype(str).str.strip()
    return vals.iloc[0] if not vals.empty else None

def make_box_table(df_user: pd.DataFrame, columns: list[str], max_width: int = 30) -> str:
    cols = [c for c in columns if c in df_user.columns]
    if not cols:
        return "(no details available)"
    rows = [[("" if pd.isna(v) else str(v)) for v in r]
            for r in df_user[cols].itertuples(index=False, name=None)]
    widths = []
    for i, c in enumerate(cols):
        content_lens = [len(r[i]) for r in rows] if rows else [0]
        w = min(max([len(c)] + content_lens), max_width)
        widths.append(w)
    def trunc(v, w): return v if len(v) <= w else (v[:w-1] + "…")
    def pad(v, w): return v + (" " * (w - len(v)))
    top = "┌" + "┬".join("─" * w for w in widths) + "┐"
    sep = "├" + "┼".join("─" * w for w in widths) + "┤"
    bot = "└" + "┴".join("─" * w for w in widths) + "┘"
    header_cells = [pad(trunc(c, w), w) for c, w in zip(cols, widths)]
    header = "│" + "│".join(header_cells) + "│"
    body_lines = []
    for r in rows:
        cells = [pad(trunc(v, w), w) for v, w in zip(r, widths)]
        body_lines.append("│" + "│".join(cells) + "│")
    return "\n".join([top, header, sep] + body_lines + [bot])

def make_html_table(df_user: pd.DataFrame, columns: list[str]) -> str:
    cols = [c for c in columns if c in df_user.columns]
    if not cols:
        return "<p>(no details available)</p>"
    def esc(x):
        return _html.escape("" if pd.isna(x) else str(x))
    thead = "<tr>" + "".join(
        f"<th style='text-align:left;padding:6px 8px;border:1px solid #ccc;background:#f7f7f7'>{esc(c)}</th>"
        for c in cols
    ) + "</tr>"
    rows_html = []
    for _, r in df_user[cols].iterrows():
        tds = "".join(
            f"<td style='text-align:left;padding:6px 8px;border:1px solid #ccc'>{esc(r[c])}</td>"
            for c in cols
        )
        rows_html.append(f"<tr>{tds}</tr>")
    tbody = "\n".join(rows_html)
    table = f"""
    <table style="border-collapse:collapse;border:1px solid #ccc;font-family:Segoe UI,Arial,sans-serif;font-size:12.5px">
      <thead>{thead}</thead>
      <tbody>{tbody}</tbody>
    </table>
    """
    return table

def build_mailto_link(to_addr: Optional[str], subject: str, body: str) -> str:
    if to_addr is None:
        to_addr = ""
    body = body.replace("\r\n", "\n").replace("\n", "\r\n")
    params = []
    if subject:
        params.append("subject=" + urllib.parse.quote(subject, safe=""))
    if body:
        params.append("body=" + urllib.parse.quote(body, safe=""))
    query = "&".join(params)
    return f"mailto:{to_addr}?{query}" if query else f"mailto:{to_addr}"

def build_outlook_web_compose(to_addr: Optional[str], subject: str, body_text: str) -> str:
    base = "https://outlook.office.com/mail/deeplink/compose"
    params = { "to": to_addr or "", "subject": subject, "body": body_text }
    return base + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)

def build_eml(to_addr: Optional[str], subject: str, html_body: str, text_fallback: str = "") -> bytes:
    msg = MIMEMultipart("alternative")
    if to_addr:
        msg["To"] = to_addr
    msg["Subject"] = subject
    if not text_fallback:
        text_fallback = "Please view this email in an HTML-capable client."
    part_text = MIMEText(text_fallback, "plain", "utf-8")
    part_html = MIMEText(html_body, "html", "utf-8")
    msg.attach(part_text)
    msg.attach(part_html)
    return msg.as_bytes()

# ====== Policy + ML helpers ======
BUSINESS_FAMILIES = [
    "LATITUDE","PRECISION","THINKPAD","THINKBOOK","ELITEBOOK","ZBOOK","PROBOOK",
    "SURFACE PRO","SURFACE LAPTOP","SURFACE BOOK","SURFACE GO","PORTEGE","DYNABOOK"
]
CONSUMER_FAMILIES = [
    "XPS","ASPIRE","VOSTRO","VIVOBOOK","ZENBOOK","ROG","NITRO","MACBOOK","INSPIRON","SATELLITE","AERO","YOGA"
]

def infer_family(model_str: str) -> str:
    if not isinstance(model_str, str): return "UNKNOWN"
    up = model_str.upper()
    for t in sorted(BUSINESS_FAMILIES + CONSUMER_FAMILIES, key=len, reverse=True):
        if t in up: return t
    return up.split()[0] if up.strip() else "UNKNOWN"

def infer_grade(family: str) -> str:
    if family in BUSINESS_FAMILIES: return "BUSINESS"
    if family in CONSUMER_FAMILIES: return "CONSUMER"
    return "UNKNOWN"

def typical_lifespan_years(family: str, make: str) -> float:
    f = (family or "").upper()
    m = (make or "").upper()
    if "SURFACE" in f or "MICROSOFT" in m: return 5.8
    if "MACBOOK" in f or "APPLE" in m:     return 6.0
    if f in BUSINESS_FAMILIES:             return 5.0
    if f in CONSUMER_FAMILIES:             return 4.0
    return 5.0

def normalize_itemtype(x: Union[str, float, None]) -> str:
    if pd.isna(x): return ""
    return str(x).strip().upper()

def is_laptop_row(row: pd.Series) -> bool:
    subcat = normalize_itemtype(row.get("SUBCAT"))
    model  = str(row.get("MODEL") or "")
    fam    = infer_family(model)
    return any(k in subcat for k in ["LAPTOP","NOTEBOOK","MACBOOK"]) or (fam in BUSINESS_FAMILIES or fam in CONSUMER_FAMILIES)

def immediate_replacement_flag(row: pd.Series) -> int:
    """
    Immediate replacement if:
      - AGE >= 5 years
      - ITEMTYPE == 'Client Service Equipment'
    """
    age_ok = False
    if "AGE" in row and pd.notna(row["AGE"]):
        try:
            age_ok = float(row["AGE"]) >= 5.0
        except Exception:
            age_ok = False
    itype = normalize_itemtype(row.get("ITEMTYPE"))
    is_service_equip = (itype == "CLIENT SERVICE EQUIPMENT")
    return int(age_ok and is_service_equip)

def warranty_predicted_date(row):
    exp = pd.to_datetime(row.get("EXPDATE"), errors="coerce")
    if pd.isna(exp):
        return None
    return exp.normalize()

def upper_bound_years_for_row(row: pd.Series) -> float:
    itype = normalize_itemtype(row.get("ITEMTYPE"))
    if itype == "CLIENT SERVICE EQUIPMENT":
        return 5.0
    if is_laptop_row(row):
        return 5.0
    fam = infer_family(str(row.get("MODEL") or ""))
    make = str(row.get("MAKE") or "")
    return typical_lifespan_years(fam, make)

def upper_bound_predicted_date(row: pd.Series) -> Optional[pd.Timestamp]:
    ub_years = upper_bound_years_for_row(row)
    pur = pd.to_datetime(row.get("PURDATE"), errors="coerce")
    today = pd.Timestamp.today().normalize()
    if pd.notna(pur):
        return (pur + pd.DateOffset(days=int(ub_years * 365.25))).normalize()
    age_val = row.get("AGE")
    if pd.notna(age_val):
        try:
            remaining_years = max(0.0, ub_years - float(age_val))
        except Exception:
            remaining_years = ub_years
        return (today + pd.DateOffset(days=int(remaining_years * 365.25))).normalize()
    return None

def predicted_replacement_date(row: pd.Series) -> Optional[pd.Timestamp]:
    w = warranty_predicted_date(row)
    u = upper_bound_predicted_date(row)
    candidates = [d for d in [w, u] if d is not None and pd.notna(d)]
    return min(candidates) if candidates else None

def remaining_days_until(dt_target: Optional[pd.Timestamp]) -> Optional[float]:
    if dt_target is None or pd.isna(dt_target):
        return None
    return (dt_target.normalize() - pd.Timestamp.today().normalize()).days

def compute_rule_replacement_date(row: pd.Series) -> Optional[pd.Timestamp]:
    pur = pd.to_datetime(row.get("PURDATE"), errors="coerce")
    exp = pd.to_datetime(row.get("EXPDATE"), errors="coerce")
    fam = infer_family(str(row.get("MODEL","") or ""))
    make = str(row.get("MAKE","") or "")
    typical = typical_lifespan_years(fam, make)
    today = pd.Timestamp.today().normalize()
    age = row.get("AGE")

    if pd.notna(pur):
        dt1 = (pur + pd.DateOffset(days=int(typical * 365.25))).normalize()
    elif pd.notna(age):
        remaining_years = max(0.0, typical - float(age))
        dt1 = (today + pd.DateOffset(days=int(remaining_years * 365.25))).normalize()
    else:
        dt1 = None

    dt2 = (exp - pd.Timedelta(days=90)).normalize() if pd.notna(exp) else None
    cands = [d for d in [dt1, dt2] if d is not None]
    return min(cands) if cands else None

def engineer_ml_features(df_in: pd.DataFrame):
    df = df_in.copy()
    df["PURDATE"] = pd.to_datetime(df["PURDATE"], errors="coerce") if "PURDATE" in df.columns else pd.NaT
    df["EXPDATE"] = pd.to_datetime(df["EXPDATE"], errors="coerce") if "EXPDATE" in df.columns else pd.NaT
    if "MODEL" not in df.columns: df["MODEL"] = np.nan
    df["family"] = df["MODEL"].apply(infer_family)
    df["grade"]  = df["family"].apply(infer_grade)
    if "AGE" not in df.columns or df["AGE"].isna().all():
        today = pd.Timestamp.today().normalize()
        df["AGE"] = ((today - df["PURDATE"]).dt.days / 365.25).round(2)
    today = pd.Timestamp.today().normalize()
    df["DAYS_TO_WARRANTY_END"] = (df["EXPDATE"] - today).dt.days
    df["IN_WARRANTY"] = (df["DAYS_TO_WARRANTY_END"] >= 0).astype("Int64").fillna(0).astype(int)
    df["RULE_REPLACE_DATE"] = df.apply(compute_rule_replacement_date, axis=1)
    H = 180
    df["RULE_REPLACE_180D"] = (pd.to_datetime(df["RULE_REPLACE_DATE"]) - today).dt.days.le(H).fillna(False).astype(int)

    cat_cols = [c for c in ["MAKE","MODEL","family","grade","SUBCAT","ITEMTYPE","CITY","COUNTRY","STATUS","BU","PROJECT","PRJNAME"] if c in df.columns]
    num_cols = [c for c in ["AGE","DAYS_TO_WARRANTY_END","IN_WARRANTY"] if c in df.columns]
    return df, cat_cols, num_cols

@st.cache_resource(show_spinner=False)
def train_bootstrap_model(df_feats: pd.DataFrame, cat_cols: list[str], num_cols: list[str], target_col="RULE_REPLACE_180D"):
    dfm = df_feats.dropna(subset=[target_col])
    if dfm.empty:
        raise ValueError("No rows available to train.")

    if num_cols:
        dfm[num_cols] = dfm[num_cols].fillna(0)

    X = dfm[cat_cols + num_cols]
    y = dfm[target_col].astype(int)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    models = {
        "logreg": Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))]),
        "rf":     Pipeline([("prep", pre), ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced_subsample", random_state=42))]),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scores = {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:,1]
        auc = roc_auc_score(y_te, p)
        ap  = average_precision_score(y_te, p)
        pr, rc, f1, _ = precision_recall_fscore_support(y_te, (p>=0.5).astype(int), average="binary", zero_division=0)
        scores[name] = {"auc": auc, "ap": ap, "precision": pr, "recall": rc, "f1": f1, "model": pipe}

    best_name = max(scores, key=lambda k: (scores[k]["ap"], scores[k]["auc"]))
    return best_name, scores[best_name]["model"], scores

# =========================
# SINGLE SOURCE OF TRUTH: Data loading (UPLOAD or PATH), cached by bytes+sheet
# =========================
st.sidebar.header("📄 Data Source")

_EXCEL_PATH_KEY = "excel_path"
_ASSET_UPLOAD_KEY = "asset_file"

if _EXCEL_PATH_KEY not in st.session_state:
    st.session_state[_EXCEL_PATH_KEY] = DEFAULT_EXCEL_PATH

def _on_upload_change():
    st.cache_data.clear()
    for k in ["_asset_model"]:
        st.session_state.pop(k, None)
    uploaded_now = st.session_state.get(_ASSET_UPLOAD_KEY)
    if uploaded_now is not None:
        st.session_state[_EXCEL_PATH_KEY] = ""  # auto-clear path on upload
    st.rerun()

uploaded = st.sidebar.file_uploader(
    "Upload an Excel file (.xlsx)",
    type=["xlsx"],
    help="Uploading a file automatically clears the path input below.",
    key=_ASSET_UPLOAD_KEY,
    on_change=_on_upload_change,
)

# Optional: clear upload button
if st.session_state.get(_ASSET_UPLOAD_KEY) is not None:
    if st.sidebar.button("🧹 Clear uploaded file"):
        st.session_state[_ASSET_UPLOAD_KEY] = None
        st.cache_data.clear()
        if not st.session_state.get(_EXCEL_PATH_KEY):
            st.session_state[_EXCEL_PATH_KEY] = DEFAULT_EXCEL_PATH
        st.rerun()

st.sidebar.text_input(
    "Or provide Excel file path",
    key=_EXCEL_PATH_KEY,
)

def read_excel_sheets_from_bytes(file_bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl").sheet_names

@st.cache_data(show_spinner=True)
def load_data_fixed(file_bytes: bytes, sheet_name: str):
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="openpyxl")
    df = normalize_columns(df)

    user_col = guess_column(df, USER_CANDIDATES)
    age_col = guess_column(df, AGE_CANDIDATES)
    purchase_col = guess_column(df, PURCHASE_DATE_CANDIDATES)
    asset_id_col = guess_column(df, ASSET_ID_CANDIDATES)
    serial_id_col = guess_column(df, SERIAL_ID_CANDIDATES)
    email_col = guess_column(df, EMAIL_CANDIDATES)

    if user_col is None:
        if "EMPNAME" in df.columns:
            user_col = "EMPNAME"
        elif "EMPID" in df.columns:
            user_col = "EMPID"

    if age_col is None and purchase_col is not None:
        df = compute_age_from_purchase_date(df, purchase_col, out_col="AGE")
        age_col = "AGE"

    if age_col and age_col in df.columns:
        df[age_col] = to_numeric_safe(df[age_col])

    text_cols_to_clean = [
        "MAKE","MODEL","CITY","COUNTRY","SUBLOC","LEVEL","LOCNAME","SEATNO","STATUS",
        "USER","ASSIGNEE","ASSIGNED_TO","OWNER","CAT","EMPNAME","EMPID","SUBCAT","ITEMTYPE"
    ]
    if email_col and email_col in df.columns:
        text_cols_to_clean.append(email_col)
    text_cols_to_clean = [c for c in text_cols_to_clean if c in df.columns]
    df = normalize_text_columns(df, text_cols_to_clean)

    categorical_value_cols = ["MAKE","MODEL","SUBCAT","ITEMTYPE","CITY","COUNTRY","STATUS","LOCNAME","SUBLOC","LEVEL","CAT"]
    df = standardize_value_case(df, [c for c in categorical_value_cols if c in df.columns], mode="upper")

    return {
        "df": df,
        "user_col": user_col,
        "age_col": age_col,
        "purchase_col": purchase_col,
        "asset_id_col": asset_id_col,
        "serial_id_col": serial_id_col,
        "email_col": email_col,
    }

# Determine source & read sheet names (UPLOAD takes priority)
file_bytes = None
file_label = None
sheet_names = []

try:
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_label = uploaded.name
        sheet_names = read_excel_sheets_from_bytes(file_bytes)
    else:
        excel_path = st.session_state[_EXCEL_PATH_KEY].strip()
        if excel_path and os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                file_bytes = f.read()
            file_label = os.path.basename(excel_path)
            sheet_names = read_excel_sheets_from_bytes(file_bytes)
        elif excel_path:
            st.sidebar.warning("The provided path does not exist. Upload a file or correct the path.")
        else:
            st.sidebar.info("Upload a file or provide a valid path to continue.")
except Exception as e:
    st.sidebar.error(f"Could not read sheet names: {e}")

# Sheet selection
if sheet_names:
    sheet = st.sidebar.selectbox("Sheet", options=sheet_names, index=0)
else:
    sheet_in = st.sidebar.text_input("Sheet name or index", value=str(DEFAULT_SHEET_NAME))
    sheet = int(sheet_in) if sheet_in.isdigit() else sheet_in

# Final load (stop app if no dataset)
if file_bytes is not None:
    try:
        data = load_data_fixed(file_bytes, sheet)
        df = data["df"]
        user_col = data["user_col"]
        age_col = data["age_col"]
        purchase_col = data["purchase_col"]
        asset_id_col = data["asset_id_col"]
        serial_id_col = data["serial_id_col"]
        email_col = data["email_col"]

        md5 = hashlib.md5(file_bytes).hexdigest()[:8]
        st.sidebar.success(f"Loaded {len(df):,} rows")
        with st.sidebar.expander("Dataset info", expanded=False):
            st.write({
                "file": file_label or "(bytes)",
                "sheet": sheet,
                "md5": md5,
                "rows": int(len(df)),
                "cols": list(df.columns)[:12] + (["…"] if len(df.columns) > 12 else []),
            })
    except Exception as e:
        st.sidebar.error(f"Failed to load Excel: {e}")
        st.stop()
else:
    st.stop()

# ---------- Column mapping overrides ----------
with st.sidebar.expander("🔧 Column mapping (override auto-detect)", expanded=False):
    colnames = list(df.columns)

    user_override = st.selectbox(
        "User column (for 'Users with multiple assets' & grouped view)",
        options=["(auto)"] + colnames,
        index=0,
        help="Pick the column representing the person. Common: EMPNAME or USER."
    )
    asset_override = st.selectbox(
        "Asset ID column",
        options=["(auto)"] + colnames,
        index=0,
        help="Pick the unique device identifier. Common: ASSETNO or SRNO or SERIAL."
    )
    email_override = st.selectbox(
        "Email column (optional)",
        options=["(auto)"] + colnames,
        index=0,
        help="Pick the column containing the employee's email. Used for the per-user email button."
    )

    if user_override != "(auto)":
        user_col = user_override
    else:
        if user_col is None:
            if "EMPNAME" in df.columns:
                user_col = "EMPNAME"
            elif "EMPID" in df.columns:
                user_col = "EMPID"

    if asset_override != "(auto)":
        asset_id_col = asset_override
    else:
        if (asset_id_col is None) or (asset_id_col not in df.columns):
            for candidate in ["ASSETNO","SRNO","SERIAL","SERIAL_NO","SERIALNUMBER","ASSET_NO","ASSET_NUMBER","ASSET_ID"]:
                if candidate in df.columns:
                    asset_id_col = candidate
                    break
            else:
                asset_id_col = None

    if email_override != "(auto)":
        email_col = email_override

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total assets", f"{len(df):,}")
with c2:
    st.metric("Asset types (SUBCAT)", f"{df['SUBCAT'].nunique() if 'SUBCAT' in df.columns else 0:,}")
with c3:
    st.metric("Countries", f"{df['COUNTRY'].nunique() if 'COUNTRY' in df.columns else 0:,}")
with c4:
    if 'AGE' in df.columns and df['AGE'].notna().any():
        st.metric("Avg Age (yrs)", f"{float(df['AGE'].dropna().mean()):.1f}")
    else:
        st.metric("Avg Age (yrs)", "—")

# ---------- Tabs (w/ ML + Emails) ----------
tab_table, tab_charts, tab_summary, tab_ml, tab_emails = st.tabs([
    "📋 Table", "📊 Charts", "ℹ️ Summary", "🤖 ML Predictions", "✉️ Emails to be Sent"
])

# ---------- Sidebar: Filters & Actions ----------
st.sidebar.header("🎛️ Filters & Actions")

subcat_options = options_for(df, "SUBCAT")
subcat_sel = st.sidebar.multiselect("SUBCAT (Asset type)", subcat_options, help="Select one or more asset types.")

action = st.sidebar.radio(
    "Choose action",
    [
        "All assets",
        "All users → all assets (grouped)",
        "Users with multiple assets",
        "Assets aged ≥ N years",
        "Immediate replacement (AGE ≥ 5 & Client Service Equipment)"
    ],
    index=0
)

with st.sidebar.expander("Attribute filters", expanded=False):
    df_after_subcat = df[df["SUBCAT"].isin(subcat_sel)] if ("SUBCAT" in df.columns and subcat_sel) else df

    def mf(col): return st.multiselect(col, options_for(df_after_subcat, col)) if col in df_after_subcat.columns else []
    itemtype_sel = mf("ITEMTYPE")
    after_itemtype = df_after_subcat[df_after_subcat["ITEMTYPE"].isin(itemtype_sel)] if itemtype_sel else df_after_subcat

    country_sel = st.multiselect("COUNTRY", options_for(after_itemtype, "COUNTRY"))
    after_country = after_itemtype[after_itemtype["COUNTRY"].isin(country_sel)] if country_sel else after_itemtype

    city_sel    = st.multiselect("CITY",    options_for(after_country, "CITY"))
    make_sel    = st.multiselect("MAKE",    options_for(after_country, "MAKE"))
    model_sel   = st.multiselect("MODEL",   options_for(after_country, "MODEL"))
    subloc_sel  = st.multiselect("SUBLOC (Building)", options_for(after_country, "SUBLOC"))
    level_sel   = st.multiselect("LEVEL",   options_for(after_country, "LEVEL"))
    locname_sel = st.multiselect("LOCNAME (Office)",  options_for(after_country, "LOCNAME"))
    seatno_sel  = st.multiselect("SEATNO",  options_for(after_country, "SEATNO"))

LIKELY_SEARCH_COLS = [
    "ASSETNO", "ASSET_NO", "ASSET_NUMBER",
    "SRNO", "SERIAL", "SERIAL_NO", "SERIALNUMBER",
    "USER", "ASSIGNEE", "ASSIGNED_TO", "OWNER", "EMAIL", "USERNAME",
    "EMPNAME", "EMPID",
    "SUBCAT", "ITEMTYPE",
    "MAKE", "MODEL", "STATUS", "CITY", "COUNTRY", "LOCNAME", "SUBLOC", "SEATNO"
]
search_cols_default = [c for c in LIKELY_SEARCH_COLS if c in df.columns] or list(df.columns)

search_cols = st.sidebar.multiselect(
    "Search across columns",
    options=list(df.columns),
    default=search_cols_default,
    help="Select which columns to search. If none are selected, the app will search across all columns."
)

query = st.sidebar.text_input("Keyword search", placeholder="Type a keyword (e.g., model, user, city...)")
ascending = st.sidebar.checkbox("Ascending", value=True)

# ---------- Apply filters & search ----------
filtered = df.copy()
if subcat_sel and "SUBCAT" in filtered.columns:
    filtered = filtered[filtered["SUBCAT"].isin(subcat_sel)]
for col, vals in [
    ("ITEMTYPE", itemtype_sel),
    ("COUNTRY", country_sel),
    ("CITY", city_sel),
    ("MAKE", make_sel),
    ("MODEL", model_sel),
    ("SUBLOC", subloc_sel),
    ("LEVEL", level_sel),
    ("LOCNAME", locname_sel),
    ("SEATNO", seatno_sel),
]:
    if vals and col in filtered.columns:
        filtered = filtered[filtered[col].isin(vals)]

filtered = keyword_search(filtered, query, search_cols)
st.caption(f"Filtered rows (ALL assets): {len(filtered):,} / {len(df):,}")

# ==== Policy columns on the current filtered view ====
work = filtered.copy()
work["WARRANTY_PREDICTED_DATE"] = work.apply(warranty_predicted_date, axis=1)
work["UPPER_BOUND_DATE"]       = work.apply(upper_bound_predicted_date, axis=1)
work["PREDICTED_REPLACE_DATE"] = work.apply(predicted_replacement_date, axis=1)
work["REMAINING_DAYS"]         = work["PREDICTED_REPLACE_DATE"].apply(remaining_days_until)
work["IMMEDIATE_REPLACE"]      = work.apply(immediate_replacement_flag, axis=1)

owner_display_col = data["user_col"] if (data["user_col"] and data["user_col"] in work.columns) else ("EMPNAME" if "EMPNAME" in work.columns else None)

# ---------- Tabs Logic ----------
if action == "All assets":
    with tab_table:
        section_header("All Assets (after filters & search)")
    result = work

elif action == "All users → all assets (grouped)":
    with tab_table:
        section_header("All users → all assets (grouped)", "Only users with ≥2 assets are shown below. Use filters to narrow scope.")
    if not user_col or user_col not in filtered.columns:
        with tab_table: st.warning("User column not detected. Map it in the sidebar.")
        result = filtered.head(0)
    elif not asset_id_col or asset_id_col not in filtered.columns:
        with tab_table: st.warning("Asset ID column not detected. Map it in the sidebar.")
        result = filtered.head(0)
    else:
        summary = (
            filtered.groupby(user_col, dropna=False)[asset_id_col]
            .nunique()
            .reset_index(name="ASSET_COUNT")
            .sort_values(["ASSET_COUNT", user_col], ascending=[False, True])
        )
        view_summary = summary.query("ASSET_COUNT >= 2").copy()

        with tab_table:
            with card():
                c1, c2, c3 = st.columns([1,1,2])
                with c1: st.metric("Users (≥2 assets)", f"{len(view_summary):,}")
                with c2: st.metric("Rows in scope", f"{int(filtered[filtered[user_col].isin(view_summary[user_col])].shape[0]):,}")
                with c3: st.caption("Tip: Adjust filters in the sidebar to focus by SUBCAT, COUNTRY, etc.")

            with card():
                st.write("**Summary (user → asset count)**")
                st.dataframe(view_summary.reset_index(drop=True), use_container_width=True, height=220)

            df_sorted = filtered.sort_values([user_col, asset_id_col], na_position="last")
            cols_to_show_default = [c for c in PREFERRED_DISPLAY_COLUMNS if c in df_sorted.columns] or list(df_sorted.columns)
            base_cols_for_email = [asset_id_col, serial_id_col, "MAKE", "MODEL", "SUBCAT", "STATUS", "CITY", "COUNTRY", "LOCNAME"]
            base_cols_for_email = [c for c in base_cols_for_email if c and c in df_sorted.columns]

            st.markdown("#### Details by user (≥2 assets)")
            st.caption("Each user has two tabs: **Assets** (full table) and **Email** (.eml download).")

            for i, (_, row) in enumerate(view_summary.iterrows()):
                u = row[user_col]
                label = u if pd.notna(u) else "(missing user)"
                user_assets = df_sorted[df_sorted[user_col] == u]

                with st.expander(f"{label} — {len(user_assets)} asset(s)", expanded=False):
                    tabs_user = st.tabs(["Assets", "Email"])

                    with tabs_user[0]:
                        with card():
                            cols_to_show = [c for c in PREFERRED_DISPLAY_COLUMNS if c in user_assets.columns] or list(user_assets.columns)
                            st.dataframe(user_assets[cols_to_show], use_container_width=True, height=280)

                    with tabs_user[1]:
                        with card():
                            col_toggle = st.toggle("Show extended columns (email table)", value=False, key=f"toggle_cols_{i}")
                            email_cols = (cols_to_show if col_toggle else base_cols_for_email) or cols_to_show_default

                            recipient = None
                            if email_col and (email_col in user_assets.columns):
                                recipient = _first_non_null(user_assets[email_col])

                            default_subject = "Asset review — items assigned to you"
                            default_greeting = f"Hi {label},"
                            box_table = make_box_table(user_assets, email_cols)
                            default_body = (
                                f"{default_greeting}\n\n"
                                "Please return any assets that you are not using from the following assets under your name:\n\n"
                                f"{box_table}\n\n"
                                "If any device is in active use, no action is needed.\n"
                                "Otherwise, please return unused items to IT. Thank you!\n\n"
                                "— IT Asset Management"
                            )

                            subject_val = st.text_input("Email subject", value=default_subject, key=f"subject_group_{i}")
                            body_val = st.text_area("Email body (plain text; box table copies cleanly into Outlook)",
                                                    value=default_body, height=220, key=f"body_group_{i}")

                            mailto_url = build_mailto_link(recipient, subject_val, body_val)
                            html_table = make_html_table(user_assets, email_cols)
                            html_body = f"""
                            <p>Hi {_html.escape(str(label))},</p>
                            <p>Please return any assets that you are not using from the following assets under your name:</p>
                            {html_table}
                            <p>If any device is in active use, no action is needed.<br>
                            Otherwise, please return unused items to IT. Thank you!</p>
                            <p>— IT Asset Management</p>
                            """
                            preview_text = f"To: {recipient or '<add recipient>'}\nSubject: {subject_val}\n\n{body_val}"
                            eml_bytes = build_eml(recipient, subject_val, html_body, text_fallback=preview_text)

                            st.markdown('<div class="action-bar">', unsafe_allow_html=True)
                            st.download_button(
                                "⬇️ Download email (.eml) — open in Outlook to edit & send",
                                data=eml_bytes,
                                file_name=f"assets_{(str(label) or 'user').replace(' ','_')}.eml",
                                mime="message/rfc822",
                            )
                            st.link_button("Open in Mail (quick draft)", mailto_url)
                            st.markdown('</div>', unsafe_allow_html=True)

                            st.markdown("**Copy-ready preview (solid table):**")
                            st.code(preview_text, language="text")

            result = df_sorted[df_sorted[user_col].isin(view_summary[user_col])]

elif action == "Users with multiple assets":
    with tab_table:
        section_header("Users with multiple assets", "Shows a flat table of all rows for users who own ≥2 unique assets.")
    if not user_col or user_col not in filtered.columns:
        with tab_table: st.warning("User column not detected. Map it in the sidebar.")
        result = filtered.head(0)
    elif not asset_id_col or asset_id_col not in filtered.columns:
        with tab_table: st.warning("Asset ID column not detected. Map it in the sidebar.")
        result = filtered.head(0)
    else:
        counts = (filtered.groupby(user_col, dropna=False)[asset_id_col].nunique().reset_index(name="ASSET_COUNT"))
        multi_users = counts.loc[counts["ASSET_COUNT"] >= 2, user_col].dropna()
        result = filtered[filtered[user_col].isin(multi_users)].copy()
        sort_cols = [c for c in [user_col, asset_id_col] if c in result.columns]
        if sort_cols: result = result.sort_values(sort_cols, na_position="last", kind="mergesort").reset_index(drop=True)
        with tab_table:
            with card():
                c1, c2 = st.columns([1,2])
                with c1: st.metric("Users (≥2 assets)", f"{multi_users.nunique():,}")
                with c2: st.metric("Rows shown", f"{len(result):,}")
                st.caption("Use filters in the sidebar to refine by type, geo, status, etc.")

elif action == "Assets aged ≥ N years":
    threshold = st.sidebar.number_input("Age threshold (years)", min_value=0, max_value=50, value=5, step=1)
    with tab_table:
        section_header(f"Assets aged ≥ {threshold} years")
    result = filter_age_threshold(filtered, age_col, threshold)
    if result.empty and (not age_col or age_col not in df.columns):
        st.warning("No AGE column detected. Add an AGE column or a purchase date so the app can compute AGE.")

elif action == "Immediate replacement (AGE ≥ 5 & Client Service Equipment)":
    with tab_table:
        section_header("Immediate replacement — Policy view",
                       "Assets with AGE ≥ 5 years AND ITEMTYPE == 'Client Service Equipment'.")
    result = work[work["IMMEDIATE_REPLACE"] == 1].copy()
    sort_cols = [c for c in ["REMAINING_DAYS", "PREDICTED_REPLACE_DATE", owner_display_col] if c in result.columns]
    if sort_cols: result = result.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    with tab_table:
        c1, c2 = st.columns([1,2])
        with c1: st.metric("Assets needing replacement now", f"{len(result):,}")
        with c2: st.caption("Tip: Export this list and trigger your email workflow.")
else:
    result = work

# ---------- TABLE TAB: per-asset panel + display & export ----------
with tab_table:
    section_header("Asset Details + Predictions", "Pick an asset to see owner and the two prediction lines.")
    selector_id_col = None
    for cand in [asset_id_col, "ASSETNO", "SRNO", "SERIAL", "ASSET_ID", "ASSET_NUMBER"]:
        if cand and cand in work.columns:
            selector_id_col = cand
            break

    if selector_id_col:
        asset_ids = work[selector_id_col].dropna().astype(str).unique().tolist()
        if asset_ids:
            asset_pick = st.selectbox(f"Select an asset ({selector_id_col})", sorted(asset_ids))
            row_sel = work[work[selector_id_col].astype(str) == asset_pick].head(1)
            if not row_sel.empty:
                r = row_sel.iloc[0]
                owner_val = r.get(owner_display_col) if owner_display_col else None
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Final predicted date (earliest)", str(r.get("PREDICTED_REPLACE_DATE") or "—"))
                with c2:
                    rd_val = r.get("REMAINING_DAYS")
                    st.metric("Remaining days", "—" if rd_val is None or pd.isna(rd_val) else f"{int(rd_val)}")
                with c3:
                    irep = r.get("IMMEDIATE_REPLACE")
                    irep = 0 if pd.isna(irep) else int(irep)
                    st.metric("Immediate replacement?", "Yes ✅" if irep == 1 else "No")

                wline = r.get("WARRANTY_PREDICTED_DATE")
                uline = r.get("UPPER_BOUND_DATE")
                wline = None if pd.isna(wline) else wline
                uline = None if pd.isna(uline) else uline
                st.write(f"**According to warranty** (EXPDATE − 90 days): {wline if wline is not None else '—'}")
                st.write(f"**According to upper bound** (~5 years): {uline if uline is not None else '—'}")

                st.write("**Current owner**:", owner_val or "—")
                detail_cols = [c for c in [
                    selector_id_col, "MAKE","MODEL","SUBCAT","ITEMTYPE","CITY","COUNTRY","STATUS","AGE",
                    "EXPDATE","PURDATE","WARRANTY_PREDICTED_DATE","UPPER_BOUND_DATE",
                    "PREDICTED_REPLACE_DATE","REMAINING_DAYS","IMMEDIATE_REPLACE",
                    owner_display_col
                ] if c in work.columns]
                st.dataframe(row_sel[detail_cols], use_container_width=True, height=220)
    else:
        st.info("No suitable asset identifier column found (try mapping ASSETNO/SRNO/SERIAL in the sidebar).")

    # Results table
    st.markdown("### Results")
    st.caption("This table reflects the current action and filters.")
    cols_to_show = [c for c in (PREFERRED_DISPLAY_COLUMNS + ["WARRANTY_PREDICTED_DATE","UPPER_BOUND_DATE","PREDICTED_REPLACE_DATE","REMAINING_DAYS","IMMEDIATE_REPLACE"]) if c in result.columns] or list(result.columns)
    st.write(f"Rows: {len(result):,}")
    st.dataframe(result[cols_to_show], use_container_width=True, height=560)

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    safe_action = action.replace(" ", "_").replace("≥", "ge").replace("→", "to").replace("&", "and")
    export_name = f"assets_{safe_action}_{timestamp}.xlsx"
    if len(result):
        xbytes = export_df_to_excel_bytes(result[cols_to_show])
        st.download_button(
            label="⬇️ Download current table as Excel",
            data=xbytes,
            file_name=export_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------- CHARTS TAB ----------
with tab_charts:
    section_header("Key Visuals")
    c1, c2 = st.columns(2)

    with c1:
        if "SUBCAT" in filtered.columns and not filtered["SUBCAT"].dropna().empty:
            st.write("Top Asset Types (SUBCAT)")
            top_subcat = filtered["SUBCAT"].dropna().value_counts().head(10).reset_index()
            top_subcat.columns = ["SUBCAT", "COUNT"]
            chart = alt.Chart(top_subcat).mark_bar().encode(
                x=alt.X("COUNT:Q", title="Count"),
                y=alt.Y("SUBCAT:N", sort='-x', title="Asset Type (SUBCAT)")
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

        if "STATUS" in filtered.columns and not filtered["STATUS"].dropna().empty:
            st.write("Assets by Status")
            status_ct = filtered["STATUS"].dropna().value_counts().reset_index()
            status_ct.columns = ["STATUS", "COUNT"]
            chart = alt.Chart(status_ct).mark_bar().encode(
                x=alt.X("STATUS:N", sort='-y', title="Status"),
                y=alt.Y("COUNT:Q", title="Count")
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

        if "ITEMTYPE" in filtered.columns and not filtered["ITEMTYPE"].dropna().empty:
            st.write("Top Item Types")
            itemtype_ct = filtered["ITEMTYPE"].dropna().value_counts().head(10).reset_index()
            itemtype_ct.columns = ["ITEMTYPE", "COUNT"]
            chart = alt.Chart(itemtype_ct).mark_bar().encode(
                x=alt.X("COUNT:Q", title="Count"),
                y=alt.Y("ITEMTYPE:N", sort='-x', title="Item Type")
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

    with c2:
        if "MAKE" in filtered.columns and not filtered["MAKE"].dropna().empty:
            st.write("Top 10 Makes")
            top_make = filtered["MAKE"].dropna().value_counts().head(10).reset_index()
            top_make.columns = ["MAKE", "COUNT"]
            chart = alt.Chart(top_make).mark_bar().encode(
                x=alt.X("COUNT:Q", title="Count"),
                y=alt.Y("MAKE:N", sort='-x', title="Make")
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

        if "AGE" in filtered.columns and filtered["AGE"].notna().any():
            st.write("Age Distribution (years)")
            age_df = pd.DataFrame({"AGE": filtered["AGE"].dropna().astype(float)})
            chart = alt.Chart(age_df).mark_bar().encode(
                x=alt.X("AGE:Q", bin=alt.Bin(maxbins=20), title="Age (years)"),
                y=alt.Y("count()", title="Count")
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

# ---------- SUMMARY TAB ----------
with tab_summary:
    section_header("Data Quality & Summary")

    missing_name = int(filtered["EMPNAME"].isna().sum()) if "EMPNAME" in filtered.columns else 0
    missing_id   = int(filtered["EMPID"].isna().sum())   if "EMPID" in filtered.columns else 0

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Rows (filtered)", f"{len(filtered):,}")
    with c2: st.metric("Missing EMPNAME", f"{missing_name:,}")
    with c3: st.metric("Missing EMPID", f"{missing_id:,}")

    st.markdown("---")

    cc1, cc2 = st.columns(2)
    with cc1:
        if "COUNTRY" in filtered.columns and not filtered["COUNTRY"].dropna().empty:
            st.write("Top Countries")
            top_countries = filtered["COUNTRY"].dropna().value_counts().head(10).reset_index()
            top_countries.columns = ["COUNTRY", "COUNT"]
            st.dataframe(top_countries, use_container_width=True, height=280)
    with cc2:
        if "CITY" in filtered.columns and not filtered["CITY"].dropna().empty:
            st.write("Top Cities")
            top_cities = filtered["CITY"].dropna().value_counts().head(10).reset_index()
            top_cities.columns = ["CITY", "COUNT"]
            st.dataframe(top_cities, use_container_width=True, height=280)

    if missing_name or missing_id:
        st.info(
            "Tip: Consider enriching missing employee data by joining with an HR master on EMPID/EMAIL. "
            "If you want, I can add an optional merge step to backfill EMPNAME/EMPID."
        )

# ---------- ML TAB ----------
with tab_ml:
    section_header("Near‑Refresh Predictions (ML)")

    target_df = filtered.copy()
    df_feats, cat_cols, num_cols = engineer_ml_features(target_df)

    if num_cols:
        df_feats[num_cols] = df_feats[num_cols].fillna(0)

    if st.button("Train model (bootstrap)", type="primary"):
        try:
            best_name, model, metrics = train_bootstrap_model(df_feats, cat_cols, num_cols, target_col="RULE_REPLACE_180D")
            st.success(f"Selected model: {best_name}")
            with st.expander("Validation metrics", expanded=True):
                for k, v in metrics.items():
                    st.write(f"{k}: AUC={v['auc']:.3f} | AP={v['ap']:.3f} | P/R/F1={v['precision']:.3f}/{v['recall']:.3f}/{v['f1']:.3f}")
            st.session_state["_asset_model"] = (model, cat_cols, num_cols)
        except Exception as ex:
            st.error(f"Training failed: {ex}")

    if "_asset_model" in st.session_state:
        model, cat_cols, num_cols = st.session_state["_asset_model"]
        try:
            feats_for_scoring = df_feats[cat_cols + num_cols].copy()
            if num_cols:
                feats_for_scoring[num_cols] = feats_for_scoring[num_cols].fillna(0)
            probs = model.predict_proba(feats_for_scoring)[:,1]
            preds = (probs >= 0.5).astype(int)
        except Exception as ex:
            st.error(f"Scoring failed: {ex}")
            st.stop()

        out = target_df.copy()
        out["RULE_REPLACE_DATE"]    = df_feats["RULE_REPLACE_DATE"]
        out["RULE_REPLACE_180D"]    = df_feats["RULE_REPLACE_180D"]
        out["ML_PROB_REPLACE_180D"] = probs.round(4)
        out["ML_PREDICT_180D"]      = preds

        st.markdown("#### Ranked assets by ML probability (next 180 days)")
        cols_rank = [c for c in ["ASSETNO","SRNO","MAKE","MODEL","ITEMTYPE","CITY","COUNTRY","STATUS","AGE","RULE_REPLACE_DATE","ML_PROB_REPLACE_180D","ML_PREDICT_180D"] if c in out.columns]
        st.dataframe(out.sort_values("ML_PROB_REPLACE_180D", ascending=False)[cols_rank].reset_index(drop=True),
                     use_container_width=True, height=420)

        xbytes_ml = export_df_to_excel_bytes(out)
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
        st.download_button(
            label="⬇️ Download predictions (Excel)",
            data=xbytes_ml,
            file_name=f"assets_ml_predictions_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.markdown("---")
        st.caption("Optional: combine policy and ML to broaden the immediate replacement list.")
        use_ml = st.toggle("Include ML‑flagged assets (prob ≥ 0.60) in policy view", value=False)
        if use_ml:
            work_ml = work.copy()
            df_feats_all, cc, nn = engineer_ml_features(work_ml)
            if nn:
                df_feats_all[nn] = df_feats_all[nn].fillna(0)
            work_ml["_ML_PROB"] = model.predict_proba(df_feats_all[cc + nn])[:,1]
            ml_mask = work_ml["_ML_PROB"] >= 0.60
            combined = work_ml[(work_ml["IMMEDIATE_REPLACE"] == 1) | (ml_mask)].copy()
            cols_to_show = [c for c in [
                "ASSETNO","SRNO","MAKE","MODEL","ITEMTYPE","SUBCAT","CITY","COUNTRY","STATUS","AGE",
                owner_display_col,"PREDICTED_REPLACE_DATE","REMAINING_DAYS","_ML_PROB","IMMEDIATE_REPLACE"
            ] if c in combined.columns] or list(combined.columns)
            st.dataframe(combined.sort_values(["IMMEDIATE_REPLACE","_ML_PROB"], ascending=[False,False])[cols_to_show],
                         use_container_width=True, height=300)

# ---------- EMAILS TAB ----------
with tab_emails:
    section_header("Emails to be Sent", "Generate .eml drafts for two workflows. Uses the current filters above.")

    # ---------------------------
    # A) All users → all assets (grouped; users with ≥2 unique assets)
    # ---------------------------
    st.markdown("## 👥 All users → all assets (grouped)")
    if not user_col or user_col not in filtered.columns:
        st.warning("User column not detected. Use the 'Column mapping' overrides in the sidebar to select EMPNAME/USER/etc.")
    elif not asset_id_col or asset_id_col not in filtered.columns:
        st.warning("Asset ID column not detected. Use the 'Column mapping' overrides in the sidebar to select ASSETNO/SRNO/SERIAL.")
    else:
        candidate_email_cols = [c for c in [email_col, "EMAIL","WORK_EMAIL","USER_EMAIL","E_MAIL"] if c and c in filtered.columns]
        if not candidate_email_cols:
            candidate_email_cols = [c for c in filtered.columns if "MAIL" in c.upper() or "EMAIL" in c.upper()]
        if not candidate_email_cols:
            st.info("No email-like column found in the current data. Map the email column in the sidebar (🔧 Column mapping).")
        else:
            pick_email_col = st.selectbox("Recipient column", options=candidate_email_cols, index=0, key="emails_tab_group_emailcol")

            counts = (filtered.groupby(user_col, dropna=False)[asset_id_col]
                      .nunique()
                      .reset_index(name="ASSET_COUNT"))
            view_users = counts.loc[counts["ASSET_COUNT"] >= 2, user_col].dropna()
            df_group_scope = filtered[filtered[user_col].isin(view_users)].copy()

            st.caption(f"In scope users (≥2 assets): **{view_users.nunique():,}**")
            name_like_cols = [c for c in ["EMPNAME","USER","ASSIGNEE","ASSIGNED_TO","OWNER","EMPID"] if c in df_group_scope.columns]
            name_col = st.selectbox("Name column (for {NAME})", options=["(none)"] + name_like_cols, index=0, key="emails_tab_group_namecol")

            default_subject = "Asset review — items assigned to you"
            default_body = (
                "Hi {NAME},\n\n"
                "Please review the assets under your name and return any that are not in use:\n\n"
                "{ASSET_TABLE}\n\n"
                "If any device is in active use, no action is needed.\n"
                "Otherwise, please return unused items to IT. Thank you!\n\n"
                "— IT Asset Management"
            )
            subj_val = st.text_input("Email subject", value=default_subject, key="emails_tab_group_subject")
            body_val = st.text_area(
                "Email body (plain text). Use {NAME} and {ASSET_TABLE} placeholders.",
                value=default_body, height=160, key="emails_tab_group_body"
            )

            default_asset_cols = [c for c in ["ASSETNO","SRNO","MAKE","MODEL","SUBCAT","STATUS","CITY","COUNTRY","LOCNAME"] if c in df_group_scope.columns]
            asset_cols_for_email = st.multiselect(
                "Columns to include in the asset table",
                options=list(df_group_scope.columns),
                default=default_asset_cols,
                key="emails_tab_group_cols"
            )

            df_mail = df_group_scope.dropna(subset=[pick_email_col]).copy()
            if not df_mail.empty:
                df_mail[pick_email_col] = df_mail[pick_email_col].astype(str).str.strip()
                df_mail = df_mail[df_mail[pick_email_col] != ""]
            recipients = (
                df_mail.groupby(user_col, dropna=False)[pick_email_col]
                .apply(lambda s: _first_non_null(s))
                .reset_index(name="_RECIP")
            )
            recipients = recipients.dropna(subset=["_RECIP"])
            st.caption(f"Unique recipients (by user): **{len(recipients):,}**")

            def _name_for_group(df_g):
                if name_col and name_col in df_g.columns and name_col != "(none)":
                    v = _first_non_null(df_g[name_col])
                    return v if v else "there"
                return "there"

            import zipfile
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for _, rec in recipients.iterrows():
                    u = rec[user_col]
                    to_addr = rec["_RECIP"]
                    df_g = df_mail[df_mail[user_col] == u].copy()

                    ascii_table = make_box_table(df_g, asset_cols_for_email)
                    rendered_body = body_val.format(NAME=_html.escape(str(_name_for_group(df_g))), ASSET_TABLE=ascii_table)

                    html_table = make_html_table(df_g, asset_cols_for_email)
                    html_body = f"""
                    <p>Hi {_html.escape(str(_name_for_group(df_g)))} ,</p>
                    <p>Please review the assets under your name and return any that are not in use:</p>
                    {html_table}
                    <p>If any device is in active use, no action is needed.<br>
                    Otherwise, please return unused items to IT. Thank you!</p>
                    <p>— IT Asset Management</p>
                    """.strip()

                    eml_bytes = build_eml(to_addr, subj_val, html_body, text_fallback=rendered_body)
                    safe_user = "user" if pd.isna(u) else str(u).strip().replace(" ", "_")
                    safe_to = str(to_addr).replace("@", "_at_").replace(".", "_")
                    zf.writestr(f"grouped_{safe_user}_{safe_to}.eml", eml_bytes)

            st.download_button(
                "⬇️ Download emails (All users ≥2 assets) — ZIP of .eml",
                data=zip_buf.getvalue(),
                file_name=f"emails_grouped_{dt.datetime.now():%Y%m%d_%H%M}.zip",
                mime="application/zip",
                use_container_width=True
            )

    st.markdown("---")

    # ---------------------------
    # B) Immediate replacement emails (AGE ≥ 5 & Client Service Equipment)
    # ---------------------------
    st.markdown("## ⚠️ Immediate Replacement (AGE ≥ 5 & Client Service Equipment)")
    imp = work[work["IMMEDIATE_REPLACE"] == 1].copy()

    if imp.empty:
        st.info("No assets currently meet the immediate replacement policy (AGE ≥ 5 & Client Service Equipment) under current filters.")
    else:
        st.caption(f"Assets flagged for immediate replacement: **{len(imp):,}**")

        candidate_email_cols_imp = [c for c in [email_col, "EMAIL","WORK_EMAIL","USER_EMAIL","E_MAIL"] if c and c in imp.columns]
        if not candidate_email_cols_imp:
            candidate_email_cols_imp = [c for c in imp.columns if "MAIL" in c.upper() or "EMAIL" in c.upper()]
        if not candidate_email_cols_imp:
            st.info("No email-like column found for immediate replacement set. Map the email column in the sidebar (🔧 Column mapping).")
        else:
            pick_email_col_imp = st.selectbox("Recipient column (Immediate Replacement)", options=candidate_email_cols_imp, index=0, key="emails_tab_imp_emailcol")

            name_like_cols_imp = [c for c in ["EMPNAME","USER","ASSIGNEE","ASSIGNED_TO","OWNER","EMPID"] if c in imp.columns]
            name_col_imp = st.selectbox("Name column (for {NAME})", options=["(none)"] + name_like_cols_imp, index=0, key="emails_tab_imp_namecol")

            default_subject_imp = "Immediate replacement required — asset(s) assigned to you"
            default_body_imp = (
                "Hi {NAME},\n\n"
                "The following asset(s) assigned to you exceed our policy (AGE ≥ 5 & Client Service Equipment) and require immediate replacement/return:\n\n"
                "{ASSET_TABLE}\n\n"
                "Please coordinate with IT to return or replace these devices.\n\n"
                "— IT Asset Management"
            )
            subj_val_imp = st.text_input("Email subject (Immediate Replacement)", value=default_subject_imp, key="emails_tab_imp_subject")
            body_val_imp = st.text_area(
                "Email body (plain text). Use {NAME} and {ASSET_TABLE} placeholders.",
                value=default_body_imp, height=160, key="emails_tab_imp_body"
            )

            default_cols_imp = [c for c in [
                "ASSETNO","SRNO","MAKE","MODEL","ITEMTYPE","SUBCAT","STATUS","CITY","COUNTRY",
                "AGE","PREDICTED_REPLACE_DATE","REMAINING_DAYS"
            ] if c in imp.columns]
            asset_cols_for_email_imp = st.multiselect(
                "Columns to include in the asset table (Immediate Replacement)",
                options=list(imp.columns),
                default=default_cols_imp,
                key="emails_tab_imp_cols"
            )

            df_imp_mail = imp.dropna(subset=[pick_email_col_imp]).copy()
            if not df_imp_mail.empty:
                df_imp_mail[pick_email_col_imp] = df_imp_mail[pick_email_col_imp].astype(str).str.strip()
                df_imp_mail = df_imp_mail[df_imp_mail[pick_email_col_imp] != ""]
            recips_imp = df_imp_mail[pick_email_col_imp].unique().tolist()
            st.caption(f"Recipients (Immediate Replacement): **{len(recips_imp):,}**")

            def _name_for_group_imp(df_g):
                if name_col_imp and name_col_imp in df_g.columns and name_col_imp != "(none)":
                    v = _first_non_null(df_g[name_col_imp])
                    return v if v else "there"
                return "there"

            import zipfile
            zip_buf_imp = io.BytesIO()
            with zipfile.ZipFile(zip_buf_imp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for em in recips_imp:
                    df_g = df_imp_mail[df_imp_mail[pick_email_col_imp] == em].copy()
                    ascii_table = make_box_table(df_g, asset_cols_for_email_imp)
                    rendered_body = body_val_imp.format(NAME=_html.escape(str(_name_for_group_imp(df_g))), ASSET_TABLE=ascii_table)

                    html_table = make_html_table(df_g, asset_cols_for_email_imp)
                    html_body = f"""
                    <p>Hi {_html.escape(str(_name_for_group_imp(df_g)))} ,</p>
                    <p>The following asset(s) assigned to you exceed our policy (AGE ≥ 5 & Client Service Equipment) and require immediate replacement/return:</p>
                    {html_table}
                    <p>Please coordinate with IT to return or replace these devices.</p>
                    <p>— IT Asset Management</p>
                    """.strip()

                    eml_bytes = build_eml(em, subj_val_imp, html_body, text_fallback=rendered_body)
                    safe_em = em.replace("@", "_at_").replace(".", "_")
                    zf.writestr(f"immediate_replacement_{safe_em}.eml", eml_bytes)

            st.download_button(
                "⬇️ Download emails (Immediate Replacement) — ZIP of .eml",
                data=zip_buf_imp.getvalue(),
                file_name=f"emails_immediate_replacement_{dt.datetime.now():%Y%m%d_%H%M}.zip",
                mime="application/zip",
                use_container_width=True
            )
