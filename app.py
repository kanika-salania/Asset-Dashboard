# -----------------------------
# Asset Dashboard (Streamlit) — All Assets
# -----------------------------
# HOW TO RUN:
#   python -m streamlit run app.py
#
# Load via file uploader (recommended) or by typing a local/server path.
# Filters and explores all assets; asset type filter driven by SUBCAT column.
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
import urllib.parse  # for mailto links

# Page setup
st.set_page_config(page_title="Asset Dashboard", layout="wide")

# --- Clean, perfectly aligned logo + title header ---
def resolve_logo(filename: str) -> Path | None:
    """Look for the file next to app.py and in ./assets or ./static."""
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

def to_b64(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None

# Theme & logo pick (theme.base is "dark" or "light")
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

# Styles to align logo and title
st.markdown("""
<style>
/* Make the header a single-row flex container */
.app-topbar {
  display: flex;
  align-items: center;               /* vertical centering */
  gap: 12px;                         /* space between logo and title */
  margin: 4px 0 12px 0;              /* compact spacing */
}

/* Control the logo size and optical alignment */
.app-logo {
  height: 44px;                      /* adjust to taste: 36–48px works well */
  width: auto;
  flex: 0 0 auto;
  object-fit: contain;
  transform: translateY(2px);        /* tiny nudge to align with text baseline */
}

/* Title reset so it doesn't push down */
.app-title {
  margin: 0;
  line-height: 1.1;
  font-weight: 700;
}

/* Optional: responsive scaling of the title on larger screens */
@media (min-width: 900px)  { .app-title { font-size: 2.0rem; } }
@media (min-width: 1200px) { .app-title { font-size: 2.2rem; } }
</style>
""", unsafe_allow_html=True)

# Render single flex row with <img> + <h1>
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

# Defaults (can be overridden in the sidebar)
DEFAULT_EXCEL_PATH = "assetlist.xlsx"
DEFAULT_SHEET_NAME = "Sheet1"

# Auto-detection candidates
USER_CANDIDATES = ["USER", "ASSIGNEE", "ASSIGNED_TO", "OWNER", "USERNAME", "EMPLOYEE", "EMAIL", "EMPNAME", "EMPID"]
AGE_CANDIDATES = ["AGE", "ASSET_AGE_YEARS"]
PURCHASE_DATE_CANDIDATES = ["PURCHASE_DATE", "BUY_DATE", "PO_DATE", "INVOICE_DATE", "PROCUREMENT_DATE"]
ASSET_ID_CANDIDATES = ["ASSETNO", "ASSET_NO", "ASSET_NUMBER", "ASSET_ID"]
SERIAL_ID_CANDIDATES = ["SRNO", "SERIAL", "SERIAL_NO", "SERIALNUMBER"]
EMAIL_CANDIDATES = ["EMAIL", "MAIL", "USER_EMAIL", "WORK_EMAIL", "E_MAIL"]  # NEW

# Preferred Results table order (includes EMPNAME & EMPID)
PREFERRED_DISPLAY_COLUMNS = [
    "ASSETNO", "SRNO", "MAKE", "MODEL",
    "EMPNAME", "EMPID",
    "SUBCAT",  # asset type
    "CITY", "COUNTRY",
    "SUBLOC", "LEVEL", "SEATNO", "LOCNAME", "STATUS", "AGE", "USER", "ITEMTYPE"
]

# ---------- Helpers: normalization and detection ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case headers, collapse spaces to underscores, trim whitespace."""
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.upper()
    )
    return df

def normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Clean text-like columns for reliable filtering/search.
    Keep missing as np.nan so the UI shows 'NaN' consistently.
    """
    for col in columns:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s.replace({"nan": np.nan, "": np.nan})
            df[col] = s
    return df

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def guess_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
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

# ---------- Data load (cached) ----------
@st.cache_data(show_spinner=True)
def load_data(path_or_buffer, sheet):
    """
    Load Excel from a filesystem path or an uploaded file-like object.
    Normalize headers, detect columns, compute age if possible, and clean text.
    """
    df = pd.read_excel(path_or_buffer, sheet_name=sheet, engine="openpyxl")
    df = normalize_columns(df)

    # Auto-detect key columns
    user_col = guess_column(df, USER_CANDIDATES)
    age_col = guess_column(df, AGE_CANDIDATES)
    purchase_col = guess_column(df, PURCHASE_DATE_CANDIDATES)
    asset_id_col = guess_column(df, ASSET_ID_CANDIDATES)
    serial_id_col = guess_column(df, SERIAL_ID_CANDIDATES)
    email_col = guess_column(df, EMAIL_CANDIDATES)  # NEW

    # Fallback for user_col to EMPNAME/EMPID if no user-like column detected
    if user_col is None:
        if "EMPNAME" in df.columns:
            user_col = "EMPNAME"
        elif "EMPID" in df.columns:
            user_col = "EMPID"

    # Compute AGE from purchase date if needed
    if age_col is None and purchase_col is not None:
        df = compute_age_from_purchase_date(df, purchase_col, out_col="AGE")
        age_col = "AGE"

    # Ensure numeric AGE if present
    if age_col and age_col in df.columns:
        df[age_col] = to_numeric_safe(df[age_col])

    # Clean text columns (including EMPNAME/EMPID, SUBCAT, ITEMTYPE, EMAIL)
    text_cols_to_clean = [
        "MAKE", "MODEL", "CITY", "COUNTRY", "SUBLOC", "LEVEL",
        "LOCNAME", "SEATNO", "STATUS", "USER", "ASSIGNEE", "ASSIGNED_TO", "OWNER", "CAT",
        "EMPNAME", "EMPID", "SUBCAT", "ITEMTYPE"
    ]
    if email_col and email_col in df.columns:
        text_cols_to_clean.append(email_col)

    text_cols_to_clean = [c for c in text_cols_to_clean if c in df.columns]
    df = normalize_text_columns(df, text_cols_to_clean)

    return {
        "df": df,
        "user_col": user_col,
        "age_col": age_col,
        "purchase_col": purchase_col,
        "asset_id_col": asset_id_col,
        "serial_id_col": serial_id_col,
        "email_col": email_col,  # NEW
    }

# ---------- Filter utilities ----------
def options_for(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).str.strip().unique().tolist()
    return sorted([v for v in vals if v != ""])

def apply_list_filter(df_in: pd.DataFrame, col: str, selected: list) -> pd.DataFrame:
    """Apply a .isin filter. Avoid repeated astype calls; cast only if needed."""
    if not selected or col not in df_in.columns:
        return df_in
    series = df_in[col]
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
        series = series.astype(str)
    right = [str(x).strip() for x in selected]
    return df_in[series.isin(right)]

def keyword_search(df_in: pd.DataFrame, query: str, cols: list[str]) -> pd.DataFrame:
    """Case-insensitive contains search across selected columns. Cast non-strings on-the-fly only."""
    if not query or not str(query).strip():
        return df_in
    q = str(query).strip()
    cols_to_search = [c for c in cols if c in df_in.columns] or list(df_in.columns)
    mask = None
    for c in cols_to_search:
        series = df_in[c]
        if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
            series = series.astype(str)
        col_mask = series.str.contains(q, case=False, na=False)
        mask = col_mask if mask is None else (mask | col_mask)
    return df_in[mask] if mask is not None else df_in

def users_with_multiple_assets(df: pd.DataFrame, user_col: str | None, asset_id_col: str | None) -> pd.DataFrame:
    if (not user_col) or (not asset_id_col) or (user_col not in df.columns) or (asset_id_col not in df.columns):
        return pd.DataFrame(columns=[user_col or "USER", "ASSET_COUNT"])
    counts = (
        df.groupby(user_col, dropna=False)[asset_id_col]
        .nunique()
        .reset_index(name="ASSET_COUNT")
    )
    return counts[counts["ASSET_COUNT"] > 1].sort_values("ASSET_COUNT", ascending=False)

def filter_age_threshold(df: pd.DataFrame, age_col: str | None, threshold: int) -> pd.DataFrame:
    if not age_col or age_col not in df.columns:
        return df.head(0)
    return df[df[age_col].fillna(-1) >= threshold]

def export_df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def read_excel_sheets(file_like_or_path) -> list[str]:
    xls = pd.ExcelFile(file_like_or_path, engine="openpyxl")
    return xls.sheet_names

# ---------- Email helpers (NEW) ----------
def _first_non_null(series: pd.Series) -> str | None:
    if series is None:
        return None
    vals = pd.Series(series).dropna().astype(str).str.strip()
    return vals.iloc[0] if not vals.empty else None

def make_assets_text_table(df_user: pd.DataFrame, columns: list[str]) -> str:
    """
    Render a compact plain-text table for email body.
    (Plain text is more reliable than HTML in mailto: bodies and easy to paste into Outlook.)
    """
    cols = [c for c in columns if c in df_user.columns]
    if not cols:
        return "(no details available)"

    # Build rows (as strings)
    rows = [[("" if pd.isna(v) else str(v)) for v in r] for r in df_user[cols].itertuples(index=False, name=None)]

    # Compute widths, cap for readability
    max_width = 30
    col_widths = []
    for idx, c in enumerate(cols):
        content_lengths = [len(r[idx]) for r in rows] if rows else [0]
        w = min(max([len(c)] + content_lengths), max_width)
        col_widths.append(w)

    def fmt_row(values):
        trunc = [(v if len(v) <= w else (v[: w - 1] + "…")) for v, w in zip(values, col_widths)]
        return " | ".join(v.ljust(w) for v, w in zip(trunc, col_widths))

    header = fmt_row(cols)
    sep = "-+-".join("-" * w for w in col_widths)
    body = "\n".join(fmt_row(r) for r in rows)
    return f"{header}\n{sep}\n{body}"

def build_mailto_link(to_addr: str | None, subject: str, body: str) -> str:
    """
    Create a mailto: URL. Uses CRLF for line breaks as many clients prefer it.
    """
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

# ---------- Sidebar: Source & Load ----------
st.sidebar.header("📄 Data Source")

# 1) File uploader (recommended for internal users)
uploaded = st.sidebar.file_uploader(
    "Upload an Excel file (.xlsx)",
    type=["xlsx"],
    help="Pick an Excel file from your computer. If you also type a path below, the uploaded file takes priority."
)

# 2) Optional path input (for admins/power users)
excel_path = st.sidebar.text_input("Or provide Excel file path", value=DEFAULT_EXCEL_PATH)

# Detect available sheets
source = uploaded if uploaded is not None else excel_path
sheet_names = []
try:
    if source:
        sheet_names = read_excel_sheets(source)
except Exception as e:
    st.sidebar.warning(f"Could not read sheet names: {e}")

# Sheet selection
if sheet_names:
    sheet = st.sidebar.selectbox("Sheet", options=sheet_names, index=0)
else:
    sheet_in = st.sidebar.text_input("Sheet name or index", value=str(DEFAULT_SHEET_NAME))
    sheet = int(sheet_in) if sheet_in.isdigit() else sheet_in

# Load the data (uploaded has priority)
try:
    if uploaded is not None:
        data = load_data(uploaded, sheet)
    else:
        data = load_data(excel_path, sheet)

    df = data["df"]
    user_col = data["user_col"]
    age_col = data["age_col"]
    purchase_col = data["purchase_col"]
    asset_id_col = data["asset_id_col"]
    serial_id_col = data["serial_id_col"]
    email_col = data["email_col"]  # NEW
except Exception as e:
    st.sidebar.error(f"Failed to load Excel: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} rows")

with st.sidebar.expander("Detected columns (auto)", expanded=False):
    st.write({
        "User column": user_col,
        "Age column": age_col,
        "Purchase date column": purchase_col,
        "Asset ID column": asset_id_col,
        "Serial ID column": serial_id_col,
        "Email column": email_col,  # NEW
    })

# ---------- Column mapping overrides (best UX) ----------
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
    email_override = st.selectbox(  # NEW
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
            for candidate in ["ASSETNO", "SRNO", "SERIAL", "SERIAL_NO", "SERIALNUMBER", "ASSET_NO", "ASSET_NUMBER", "ASSET_ID"]:
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

# ---------- Tabs ----------
tab_table, tab_charts, tab_summary = st.tabs(["📋 Table", "📊 Charts", "ℹ️ Summary"])

# ---------- Sidebar: Filters & Actions ----------
st.sidebar.header("🎛️ Filters & Actions")

# 🔝 Asset type filter (SUBCAT) first
subcat_options = options_for(df, "SUBCAT")
subcat_sel = st.sidebar.multiselect("SUBCAT (Asset type)", subcat_options, help="Select one or more asset types.")

# NEW: add grouped view into the action list
action = st.sidebar.radio(
    "Choose action",
    ["All assets", "All users → all assets (grouped)", "Users with multiple assets", "Assets aged ≥ N years"],
    index=0
)

with st.sidebar.expander("Attribute filters", expanded=False):
    # Build staged slices to keep options relevant
    df_after_subcat = apply_list_filter(df, "SUBCAT", subcat_sel) if subcat_sel else df

    itemtype_sel = st.multiselect("ITEMTYPE", options_for(df_after_subcat, "ITEMTYPE"))
    df_after_itemtype = apply_list_filter(df_after_subcat, "ITEMTYPE", itemtype_sel) if itemtype_sel else df_after_subcat

    country_sel = st.multiselect("COUNTRY", options_for(df_after_itemtype, "COUNTRY"))
    df_after_country = apply_list_filter(df_after_itemtype, "COUNTRY", country_sel) if country_sel else df_after_itemtype

    city_sel    = st.multiselect("CITY",    options_for(df_after_country, "CITY"))
    make_sel    = st.multiselect("MAKE",    options_for(df_after_country, "MAKE"))
    model_sel   = st.multiselect("MODEL",   options_for(df_after_country, "MODEL"))
    subloc_sel  = st.multiselect("SUBLOC (Building)", options_for(df_after_country, "SUBLOC"))
    level_sel   = st.multiselect("LEVEL",   options_for(df_after_country, "LEVEL"))
    locname_sel = st.multiselect("LOCNAME (Office)",  options_for(df_after_country, "LOCNAME"))
    seatno_sel  = st.multiselect("SEATNO",  options_for(df_after_country, "SEATNO"))

# Build safer defaults for search
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
filtered = df
filtered = apply_list_filter(filtered, "SUBCAT",   subcat_sel)
filtered = apply_list_filter(filtered, "ITEMTYPE", itemtype_sel)
filtered = apply_list_filter(filtered, "COUNTRY",  country_sel)
filtered = apply_list_filter(filtered, "CITY",     city_sel)
filtered = apply_list_filter(filtered, "MAKE",     make_sel)
filtered = apply_list_filter(filtered, "MODEL",    model_sel)
filtered = apply_list_filter(filtered, "SUBLOC",   subloc_sel)
filtered = apply_list_filter(filtered, "LEVEL",    level_sel)
filtered = apply_list_filter(filtered, "LOCNAME",  locname_sel)
filtered = apply_list_filter(filtered, "SEATNO",   seatno_sel)

filtered = keyword_search(filtered, query, search_cols)

st.caption(f"Filtered rows (ALL assets): {len(filtered):,} / {len(df):,}")

# ---------- Main actions ----------
if action == "All assets":
    with tab_table:
        st.subheader("All Assets (after filters & search)")
    result = filtered

elif action == "All users → all assets (grouped)":
    with tab_table:
        st.subheader("All users → all assets (grouped, only users with ≥2 assets)")
        # Validate required columns
        if not user_col or user_col not in filtered.columns:
            st.warning("User column not detected. Use the 'Column mapping' override to select EMPNAME/USER/etc.")
            result = filtered.head(0)
        elif not asset_id_col or asset_id_col not in filtered.columns:
            st.warning("Asset ID column not detected. Use the 'Column mapping' override to select ASSETNO/SRNO/SERIAL.")
            result = filtered.head(0)
        else:
            # Summary (user → unique asset count)
            summary = (
                filtered.groupby(user_col, dropna=False)[asset_id_col]
                .nunique()
                .reset_index(name="ASSET_COUNT")
                .sort_values(["ASSET_COUNT", user_col], ascending=[False, True])
            )

            # ONLY users with multiple assets (≥2)
            view_summary = summary.query("ASSET_COUNT >= 2").copy()

            st.write("Summary (user → asset count)")
            st.dataframe(view_summary, use_container_width=True, height=300)

            # Pre-sort once for reuse
            df_sorted = filtered.sort_values([user_col, asset_id_col], na_position="last")

            # -------- Details by user (ONLY ≥2 assets) --------
            st.markdown("#### Details by user (≥2 assets)")
            email_table_pref_cols = [
                asset_id_col, serial_id_col, "MAKE", "MODEL", "SUBCAT", "STATUS", "CITY", "COUNTRY", "LOCNAME"
            ]
            email_table_pref_cols = [c for c in email_table_pref_cols if c]  # drop None

            for i, (_, row) in enumerate(view_summary.iterrows()):
                u = row[user_col]
                label = u if pd.notna(u) else "(missing user)"
                user_assets = df_sorted[df_sorted[user_col] == u]

                with st.expander(f"{label} — {len(user_assets)} asset(s)"):
                    # Show preferred columns if available
                    cols_to_show = [c for c in PREFERRED_DISPLAY_COLUMNS if c in user_assets.columns] or list(user_assets.columns)
                    st.dataframe(user_assets[cols_to_show], use_container_width=True, height=300)

                    # -------- Email generation UI (per user, copy-ready for Outlook) --------
                    st.markdown("**✉️ Generate ready-to-send email**")

                    # Recipient (from email column if present)
                    recipient = None
                    if email_col and (email_col in user_assets.columns):
                        recipient = _first_non_null(user_assets[email_col])

                    # Subject + Body defaults (editable)
                    default_subject = "Asset review — items assigned to you"
                    default_greeting = f"Hi {label},"
                    body_table = make_assets_text_table(user_assets, email_table_pref_cols or cols_to_show)
                    default_body = (
                        f"{default_greeting}\n\n"
                        "Please return any assets that you are not using from the following assets under your name:\n\n"
                        f"{body_table}\n\n"
                        "If any device is in active use, no action is needed.\n"
                        "Otherwise, please return unused items to IT. Thank you!\n\n"
                        "— IT Asset Management"
                    )

                    subject_val = st.text_input(
                        "Email subject",
                        value=default_subject,
                        key=f"subject_group_{i}"
                    )
                    body_val = st.text_area(
                        "Email body (editable; plain text works best in Outlook copy/paste)",
                        value=default_body,
                        height=220,
                        key=f"body_group_{i}"
                    )

                    # Convenience: mailto link (optional)
                    mailto_url = build_mailto_link(recipient, subject_val, body_val)

                    cols_email = st.columns([1, 2])
                    with cols_email[0]:
                        st.link_button("Open in Mail", mailto_url, use_container_width=True)
                    with cols_email[1]:
                        if recipient:
                            st.caption(f"To: {recipient}")
                        else:
                            st.info(
                                "No email found for this person. Map the Email column in **Column mapping** or enter the address manually after clicking **Open in Mail**."
                            )

                    # Copy-ready preview (includes Subject + Body) — easy to paste into Outlook
                    preview_text = f"To: {recipient or '<add recipient>'}\nSubject: {subject_val}\n\n{body_val}"
                    st.markdown("**Copy-ready preview (click the copy icon):**")
                    st.code(preview_text, language="text")

            # Set result to a flat table so export works for this scope (only multi-asset users)
            result = df_sorted[df_sorted[user_col].isin(view_summary[user_col])]

elif action == "Users with multiple assets":
    # ---------- HEADER (inside the Table tab) ----------
    with tab_table:
        st.subheader("All assets for users who own ≥ 2 assets (flat table)")

    # ---------- VALIDATION ----------
    if not user_col or user_col not in filtered.columns:
        with tab_table:
            st.warning("User column not detected. Use the 'Column mapping' override to select EMPNAME/USER/etc.")
        result = filtered.head(0)

    elif not asset_id_col or asset_id_col not in filtered.columns:
        with tab_table:
            st.warning("Asset ID column not detected. Use the 'Column mapping' override to select ASSETNO/SRNO/SERIAL.")
        result = filtered.head(0)

    else:
        # ---------- COMPUTE THE FLAT TABLE (ALL ROWS FOR USERS WITH ≥2 ASSETS) ----------
        counts = (
            filtered.groupby(user_col, dropna=False)[asset_id_col]
            .nunique()
            .reset_index(name="ASSET_COUNT")
        )
        multi_users = counts.loc[counts["ASSET_COUNT"] >= 2, user_col].dropna()

        result = filtered[filtered[user_col].isin(multi_users)].copy()

        # Stable, readable ordering
        sort_cols = [c for c in [user_col, asset_id_col] if c in result.columns]
        if sort_cols:
            result = result.sort_values(sort_cols, na_position="last", kind="mergesort").reset_index(drop=True)

        # ---------- SHOW CAPTION + DEDICATED DOWNLOAD BUTTON ----------
        with tab_table:
            st.caption(f"Users with ≥2 assets: {multi_users.nunique()} • Rows shown: {len(result):,}")

            cols_to_export = [c for c in PREFERRED_DISPLAY_COLUMNS if c in result.columns] or list(result.columns)
            try:
                xbytes_multi = export_df_to_excel_bytes(result[cols_to_export])
            except Exception as ex:
                st.error(f"Export failed: {ex}")
                xbytes_multi = None

            if xbytes_multi is not None:
                timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
                export_name = f"assets_users_with_multiple_assets_{timestamp}.xlsx"
                st.download_button(
                    label="⬇️ Download as spreadsheet (Excel)",
                    data=xbytes_multi,
                    file_name=export_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_multi_assets_excel_top",
                    use_container_width=True
                )

elif action == "Assets aged ≥ N years":
    threshold = st.sidebar.number_input(
        "Age threshold (years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )
    with tab_table:
        st.subheader(f"Assets aged ≥ {threshold} years")
    result = filter_age_threshold(filtered, age_col, threshold)
    if result.empty and (not age_col or age_col not in df.columns):
        st.warning("No AGE column detected. Add an AGE column or a purchase date so the app can compute AGE.")
else:
    result = filtered

# ---------- TABLE TAB: display & export ----------
with tab_table:
    cols_to_show = [c for c in PREFERRED_DISPLAY_COLUMNS if c in result.columns] or list(result.columns)

    st.markdown("### Results")
    st.write(f"Rows: {len(result):,}")
    st.dataframe(result[cols_to_show], use_container_width=True, height=560)

    # Timestamped export filename
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    safe_action = action.replace(" ", "_").replace("≥", "ge").replace("→", "to")
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
    st.subheader("Key Visuals")

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
    st.subheader("Data Quality & Summary")

    missing_name = int(filtered["EMPNAME"].isna().sum()) if "EMPNAME" in filtered.columns else 0
    missing_id   = int(filtered["EMPID"].isna().sum())   if "EMPID" in filtered.columns else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows (filtered)", f"{len(filtered):,}")
    with c2:
        st.metric("Missing EMPNAME", f"{missing_name:,}")
    with c3:
        st.metric("Missing EMPID", f"{missing_id:,}")

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
