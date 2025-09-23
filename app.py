# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 12:16:23 2025
FKTS Sales Monitor (Streamlit) — Python >= 3.9
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import string
import requests
import io
from requests.exceptions import HTTPError, Timeout, RequestException
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Must be the first Streamlit command:
st.set_page_config(page_title="FKTS Sales Monitor", layout="wide")
pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"
# color sequence used across charts
COLOR_SEQ = px.colors.qualitative.Dark24  # 24 distinct colors
px.defaults.color_discrete_sequence = COLOR_SEQ

# ---------- CONFIG ----------
# --- Google Drive public files (Anyone with link: Viewer) ---
FILES = {
    "Dataset A": "1uVpM0NVoChXgCTRZqHsy2GS3hNDWs3oR",
    "Dataset B": "1Q3dUhLpsJ9fjeAzwzSYrCJ6ZNPYvH2rl",
}

COLS = {
    "transaction_date": ["Transaction Date", "date"],
    "company":          ["Company Name", "client/company", "company", "client"],
    "receipt_type":     ["Receipt Type", "type", "invoice type"],
    "receipt_no":       ["Receipt No.", "receipt no", "invoice no.", "invoice no", "receipt number"],
    "amount":           ["Receipt Amount", "amount", "total"],
}
REQUIRED_MIN = {"transaction_date","company","amount"}
TOP_N_DEFAULT = 10
PIE_LABEL_MAX = 10

# Optional: map known variants -> one canonical label (edit as you learn them)
ALIASES = {
    # "FRIGIDZONE MARIKINA": "FRIGIDZONE",
    # "FRIGIDZONE KAMIAS":   "FRIGIDZONE",
}

# ---------- Header detection helpers ----------
def _norm(s):
    if pd.isna(s): return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _find_header_and_rename(df_raw):
    if df_raw is None or df_raw.empty: return None, {}
    df = df_raw.dropna(how="all").dropna(axis=1, how="all")
    if df.empty: return None, {}
    norm_aliases = {k: [_norm(a) for a in v] for k, v in COLS.items()}

    max_scan = min(60, len(df))
    for i in range(max_scan):
        header_vals = list(df.iloc[i].values)
        norm_header = [_norm(v) for v in header_vals]
        idx_map = {}
        for key, aliases in norm_aliases.items():
            hit = None
            for j, v in enumerate(norm_header):
                if v in aliases:
                    hit = j; break
            if hit is not None:
                idx_map[key] = hit
        if REQUIRED_MIN.issubset(idx_map):
            # rename-by-position
            rename_by_pos = {idx: key for key, idx in idx_map.items()}
            data = df.iloc[i+1:].copy()
            new_cols = []
            for pos, col in enumerate(df.columns):
                if pos in rename_by_pos:
                    new_cols.append(rename_by_pos[pos])
                else:
                    nc = _norm(col) or f"col_{pos}"
                    new_cols.append(nc)
            data.columns = new_cols
            # ensure canonical columns exist and order them first
            keep = list(COLS.keys())
            for k in keep:
                if k not in data.columns:
                    data[k] = pd.NA
            ordered = keep + [c for c in data.columns if c not in keep]
            data = data[ordered]
            return data, {k: header_vals[idx_map[k]] for k in idx_map}
    return None, {}

def _parse_date(x):
    if pd.isna(x): return pd.NaT
    if isinstance(x, (int, float)) and not pd.isna(x):
        try:
            return pd.to_datetime("1899-12-30") + pd.to_timedelta(float(x), unit="D")
        except Exception:
            pass
    return pd.to_datetime(x, errors="coerce")

def _clean_amount(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d\.\-]", "", s.replace("(", "").replace(")", ""))
    if s in ("", "-", "."): return np.nan
    try:
        v = float(s);  return -v if neg else v
    except: return np.nan

def _tidy_company(s):
    if pd.isna(s): return np.nan
    s = re.sub(r"\s+", " ", str(s).strip())
    return s if s else np.nan

def _company_key(s: str):
    """Uppercase, remove punctuation, collapse spaces; apply ALIASES."""
    if pd.isna(s): return np.nan
    s = str(s).strip().upper()
    # replace punctuation with space, collapse spaces
    s = s.translate(str.maketrans({c: " " for c in string.punctuation}))
    s = re.sub(r"\s+", " ", s).strip()
    # optional light cleanup of branch-like tails (tweak as needed)
    s = re.sub(r"\b(BRANCH|INC|CORP|CORPORATION|CO|LTD|PHILIPPINES)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s) if s else np.nan

# Peso formatter for Matplotlib axes
def _peso_fmt(x, pos=None):
    return f"₱{x:,.0f}"

def plot_yearly_tendency_stream(monthly_ts):
    """
    monthly_ts: DataFrame with ['month_dt','monthly_total'] sorted.
    Returns (fig, yearly_df, cagr, slope_per_year).
    """
    if monthly_ts is None or monthly_ts.empty:
        return None, pd.DataFrame(), np.nan, np.nan

    tmp = monthly_ts.copy()
    tmp["year"] = tmp["month_dt"].dt.year

    # yearly totals + completeness
    year_tot = (tmp.groupby("year", as_index=False)["monthly_total"]
                  .sum().rename(columns={"monthly_total": "year_total"}))
    counts = tmp.groupby("year")["month_dt"].nunique().rename("n_months")
    year_tot = year_tot.merge(counts, on="year", how="left").sort_values("year")
    if year_tot.empty:
        return None, pd.DataFrame(), np.nan, np.nan

    # CAGR
    first_val = float(year_tot["year_total"].iloc[0])
    last_val  = float(year_tot["year_total"].iloc[-1])
    n_years   = int(year_tot["year"].iloc[-1] - year_tot["year"].iloc[0])
    cagr = (last_val / first_val) ** (1.0 / n_years) - 1.0 if (n_years > 0 and first_val > 0) else np.nan

    # Trend (OLS on yearly totals)
    years  = year_tot["year"].to_numpy(dtype=float)
    totals = year_tot["year_total"].to_numpy(dtype=float)
    slope = np.nan
    xfit = yfit = None
    if len(years) >= 2:
        m, b = np.polyfit(years, totals, 1)
        slope = m  # ₱/year
        xfit = np.linspace(years.min()-0.2, years.max()+0.2, 100)
        yfit = m * xfit + b

    # Plot (Matplotlib)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(years, totals, width=0.6, color="#1f77b4")
    ax.yaxis.set_major_formatter(FuncFormatter(_peso_fmt))
    ax.set_xlabel("Year"); ax.set_ylabel("Total Sales (₱)")

    title = "Yearly Total Sales"
    if not np.isnan(cagr):
        title += f" • Compound Annual Growth Rate: {cagr*100:,.1f}%"
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # annotate bars
    for b in bars:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val, f"₱{val:,.0f}",
                va="bottom", ha="center", fontsize=9)

    # trend line + slope label
    if xfit is not None:
        ax.plot(xfit, yfit, linewidth=2.0, color="#f39c12")
        ax.text(0.02, 0.95, f"Trend slope: ₱{slope:,.0f}/yr",
                transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#555555")

    plt.tight_layout()
    return fig, year_tot.reset_index(drop=True), cagr, slope

# ---------- Data loading (Google Drive: always load BOTH) ----------
def _gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"

@st.cache_data(ttl=3600, show_spinner=False)
def _download_bytes(file_id: str) -> bytes:
    url = _gdrive_url(file_id)
    last_err = None
    for _ in range(3):  # retry a few times
        try:
            r = requests.get(url, timeout=45, headers={"User-Agent": "fkts-streamlit"})
            r.raise_for_status()
            return r.content
        except (HTTPError, Timeout, RequestException) as e:
            last_err = e
    raise RuntimeError(f"Drive fetch failed ({file_id}): {last_err}")

def _read_any_tablelike(content: bytes):
    """
    Try CSV first (fast), then Excel if needed.
    Return list of (sheet_name, df_without_header).
    """
    out, bio = [], io.BytesIO(content)
    # CSV first
    try:
        bio.seek(0)
        df_csv = pd.read_csv(bio, dtype=object, header=None)
        if not df_csv.empty:
            out.append(("csv", df_csv))
            return out
    except Exception:
        pass
    # Excel fallback (multi-sheet)
    try:
        bio.seek(0)
        book = pd.read_excel(bio, sheet_name=None, engine="openpyxl", dtype=object, header=None)
        for sname, df in (book or {}).items():
            if df is not None and not df.empty:
                out.append((str(sname), df))
    except Exception:
        pass
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def load_both_drive_files(file_ids: dict, data_version: str = "v1") -> pd.DataFrame:
    """
    Load ALL files in file_ids, normalize columns/values, and concatenate.
    data_version is a cache-buster you can bump when the Drive files change.
    """
    frames = []
    for label, fid in file_ids.items():
        try:
            content = _download_bytes(fid)
        except Exception as e:
            st.warning(f"Skipping {label}: {e}")
            continue

        for sname, raw in _read_any_tablelike(content):
            data, _hdr = _find_header_and_rename(raw)  # <- your existing header detector
            if data is None or data.empty:
                continue
            data["__sheet__"]  = sname
            data["__source__"] = label
            frames.append(data)

    if not frames:
        return pd.DataFrame(columns=list(COLS.keys()))

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Clean & normalize (your existing helpers)
    df["transaction_date"] = df["transaction_date"].apply(_parse_date)
    df["amount"]           = df["amount"].apply(_clean_amount).astype("float64")
    df["company"]          = df["company"].apply(_tidy_company)
    df = df[df["company"].notna()]

    # Canonical key + de-dup
    df["company_key"] = df["company"].apply(_company_key)
    DEDUP_KEYS = ["transaction_date", "receipt_no", "company_key", "amount"]
    if set(DEDUP_KEYS).issubset(df.columns):
        df = df.drop_duplicates(subset=DEDUP_KEYS, keep="first")

    # Derived fields
    df["year"]      = df["transaction_date"].dt.year
    df["month_key"] = df["transaction_date"].dt.strftime("%Y-%m")

    # Sort for deterministic plots
    df = df.sort_values(["transaction_date", "company_key"], kind="mergesort").reset_index(drop=True)
    return df

# Load
df = load_both_drive_files(FILES)

# ---------- UI ----------
st.title("Sales Monitor")

if df.empty:
    st.error("No data loaded. Check Google Drive sharing (Anyone with link: Viewer) and file format.")
    st.stop()

# Sidebar filters
years = sorted(df["year"].dropna().unique().astype(int))
rtype_opts = sorted([x for x in df.get("receipt_type", pd.Series()).dropna().unique()])

with st.sidebar:
    st.header("Filters")
    # Default to all years instead of last year only
    year_sel = st.multiselect("Years", years, default=years if years else [])
    rtype_sel = st.multiselect("Receipt Type", rtype_opts, default=rtype_opts)
    top_n = st.slider("Top-N companies", 5, 30, TOP_N_DEFAULT)

# Apply filters
d = df.copy()
if year_sel:
    d = d[d["year"].isin(year_sel)]
if rtype_sel and "receipt_type" in d:
    d = d[d["receipt_type"].isin(rtype_sel)]

# ---------- KPIs ----------
colk1, colk2 = st.columns(2)

# totals
total_sum = float(d["amount"].sum()) if "amount" in d else 0.0

# date range (robust to NaT/missing)
dates = pd.to_datetime(d["transaction_date"], errors="coerce") if "transaction_date" in d else pd.Series([], dtype="datetime64[ns]")
date_min = dates.min()
date_max = dates.max()
date_range = f"{date_min:%Y-%m} to {date_max:%Y-%m}" if pd.notna(date_min) and pd.notna(date_max) else "n/a"

# uniques (optional)
unique_companies = d["company_key"].nunique() if "company_key" in d else 0

colk1.metric("Total", f"₱{total_sum:,.0f}")
colk2.metric("Date Range", date_range)

st.divider()

# ---------- Row 1: Active Companies per Year & Yearly Total Sales side by side ----------
st.subheader("Active Companies & Yearly Total Sales")

# prefer canonical key; fall back to raw name if not present
key_col = "company_key" if "company_key" in d.columns else "company"

ac = (d.dropna(subset=["year", key_col])
        .drop_duplicates(subset=["year", key_col])  # one row per (year, company)
        .groupby("year")[key_col].size()
        .reset_index(name="active_companies")
        .sort_values("year"))

# Build monthly_ts for yearly totals (once)
monthly_ts = d.groupby("month_key", dropna=False)["amount"].sum().rename("monthly_total").reset_index()
monthly_ts["month_dt"] = pd.to_datetime(monthly_ts["month_key"], format="%Y-%m", errors="coerce")
monthly_ts = monthly_ts.dropna().sort_values("month_dt").reset_index(drop=True)

# Precompute yearly totals
tmp = monthly_ts.copy()
tmp["year"] = tmp["month_dt"].dt.year
year_tot = tmp.groupby("year")["monthly_total"].sum().rename("year_total").reset_index()

# CAGR and trend line
if not year_tot.empty:
    first_val = float(year_tot["year_total"].iloc[0])
    last_val  = float(year_tot["year_total"].iloc[-1])
    n_years   = int(year_tot["year"].iloc[-1] - year_tot["year"].iloc[0])
    cagr = (last_val / first_val) ** (1.0 / n_years) - 1.0 if (n_years > 0 and first_val > 0) else None
else:
    cagr = None

if len(year_tot) >= 2:
    m, b = np.polyfit(year_tot["year"], year_tot["year_total"], 1)
    xfit = np.linspace(year_tot["year"].min()-0.2, year_tot["year"].max()+0.2, 100)
    yfit = m*xfit + b
else:
    m = None
    xfit = yfit = None

# Two columns side by side
c1, c2 = st.columns(2)

# --- Left chart: Active Companies per Year ---
if ac.empty:
    c1.info("No data available for Active Companies.")
else:
    fig_ac = px.bar(
        ac, x="year", y="active_companies",
        title="Active Companies per Year (unique within each year)",
        text="active_companies"
    )
    fig_ac.update_traces(textposition="outside")
    ymax = ac["active_companies"].max()
    fig_ac.update_yaxes(range=[0, ymax * 1.15])
    c1.plotly_chart(fig_ac, use_container_width=True)

# --- Right chart: Yearly Total Sales with trend ---
if year_tot.empty:
    c2.info("Not enough data to compute yearly totals for the current filters.")
else:
    # Plotly version with trend line
    fig_year = px.bar(
        year_tot, x="year", y="year_total",
        text="year_total",
        title=("Yearly Total Sales" +
               (f" • CAGR {cagr*100:,.1f}%" if cagr else ""))
    )
    fig_year.update_traces(texttemplate="₱%{text:,.0f}", textposition="outside")
    fig_year.update_yaxes(tickprefix="₱", separatethousands=True)

    if xfit is not None:
        fig_year.add_scatter(
            x=xfit, y=yfit, mode="lines", name=f"Trend (₱{m:,.0f}/yr)",
            line=dict(color="#f39c12", width=3)
        )
    c2.plotly_chart(fig_year, use_container_width=True)

st.divider()

# ---------- Row 2: Run-rate gauge (full width) ----------
st.subheader("Run-rate Gauge")

if d["transaction_date"].notna().any():
    last_dt = d["transaction_date"].max()
    yr, mo = int(last_dt.year), int(last_dt.month)
    cur_ytd  = d[(d["transaction_date"].dt.year == yr) & (d["transaction_date"].dt.month <= mo)]["amount"].sum()
    prev_ytd = d[(d["transaction_date"].dt.year == yr - 1) & (d["transaction_date"].dt.month <= mo)]["amount"].sum()
    proj = cur_ytd * (12.0 / mo) if mo > 0 else np.nan
    gauge = pd.DataFrame({"metric": ["Prev YTD", "Current YTD", "Projection"],
                          "value": [prev_ytd, cur_ytd, proj]})
    
    color_map = {
        "Prev YTD": "#c6dbef",        
        "Current YTD": "#6baed6",     
        "Projection": "#2171b5"      
    }
    
    fig_g = px.bar(
        gauge,
        x="value", y="metric", orientation="h",
        color="metric",
        title="Run-rate Gauge",
        color_discrete_map=color_map
    )
    # More height + smaller bargap = thicker bars
    fig_g.update_layout(
        showlegend=False,
        height=400,          # taller figure => thicker bars
        bargap=0.02,         # less space between bars => thicker bars
        bargroupgap=0.0,
        template="plotly_white"
    )
    fig_g.update_xaxes(tickprefix="₱", separatethousands=True)
    st.plotly_chart(fig_g, use_container_width=True)

st.divider()

# ---------- Row 3: Monthly Total Sales (full width) ----------
st.subheader("Monthly Total Sales")

# Monthly totals
ts = d.groupby("month_key", dropna=False)["amount"].sum().rename("monthly_total").reset_index()
ts["month_dt"] = pd.to_datetime(ts["month_key"], format="%Y-%m", errors="coerce")
ts = ts.dropna().sort_values("month_dt").reset_index(drop=True)
# 3-mo centered rolling mean (robust to short series)
ts["trend_3mo"] = ts["monthly_total"].rolling(window=3, center=True, min_periods=1).mean()

fig_ts = px.line(
    ts, x="month_dt", y="monthly_total", markers=True,
    title="Monthly Total Sales"
)
fig_ts.update_traces(name="Monthly", hovertemplate="%{x|%b %Y}<br>₱%{y:,.0f}<extra></extra>")
fig_ts.update_yaxes(tickprefix="₱", separatethousands=True)
# Add orange trend line
fig_ts.add_scatter(
    x=ts["month_dt"], y=ts["trend_3mo"], mode="lines",
    name="3-mo Trend", line=dict(width=3, color="#f39c12")
)
st.plotly_chart(fig_ts, use_container_width=True)

# ---------- Row 4: Top Companies & Share of Total Sales ----------
st.subheader("Top Companies & Share of Total Sales")

# Top-N companies (current filter)
ytd = d.groupby("company", dropna=False)["amount"].sum().reset_index()
ytd = ytd.sort_values("amount", ascending=False)

# side-by-side
c1, c2 = st.columns(2)

# Left: Top 10 Companies (bar)
fig_top = px.bar(
    ytd.head(top_n)[::-1], x="amount", y="company", orientation="h",
    title=f"Top {top_n} Companies"
)
fig_top.update_xaxes(tickprefix="₱", separatethousands=True)
c1.plotly_chart(fig_top, use_container_width=True)

# Right: Share pie (Top-10 + Others)
pie_topn = 10
pie = ytd.copy()
others = pie["amount"][pie_topn:].sum()
pie = pie.head(pie_topn)
if others > 0:
    pie = pd.concat([pie, pd.DataFrame({"company": ["Others"], "amount": [others]})], ignore_index=True)
pie["label"] = pie["company"].str.slice(0, PIE_LABEL_MAX) + np.where(
    pie["company"].str.len() > PIE_LABEL_MAX, "…", ""
)
fig_pie = px.pie(
    pie, names="label", values="amount",
    title=f"Share of Total Sales (Top-{pie_topn} + Others)",
    color_discrete_sequence=COLOR_SEQ
)
c2.plotly_chart(fig_pie, use_container_width=True)

# ---------- Row 5: Invoice-type mix (monthly/yearly + normalized) ----------
if "receipt_type" in d.columns and d["receipt_type"].notna().any():
    d_receipts = d[d["receipt_type"].notna()].copy()
    d_receipts["receipt_type"] = d_receipts["receipt_type"].astype(str).str.strip()

    # ---------- Monthly stacked ----------
    m = d_receipts.groupby(["month_key","receipt_type"], as_index=False)["amount"].sum()
    # pivot to wide form
    m_wide = m.pivot(index="month_key", columns="receipt_type", values="amount").fillna(0.0)
    m_wide = m_wide.sort_index()
    # now plot with go.Figure to get true stacked bars
    import plotly.graph_objects as go
    fig_mix_m = go.Figure()
    for col in m_wide.columns:
        fig_mix_m.add_bar(name=col, x=m_wide.index, y=m_wide[col])
    fig_mix_m.update_layout(
        barmode="stack",
        title="Invoice-Type Mix by Month",
        xaxis_title="Month",
        yaxis_title="Amount",
        yaxis_tickprefix="₱",
        yaxis_separatethousands=True,
        template="plotly_white"
    )
    st.plotly_chart(fig_mix_m, use_container_width=True)

    # ---------- Yearly stacked ----------
    y = d_receipts.groupby(["year","receipt_type"], as_index=False)["amount"].sum()
    y_wide = y.pivot(index="year", columns="receipt_type", values="amount").fillna(0.0)
    y_wide = y_wide.sort_index()
    fig_mix_y = go.Figure()
    for col in y_wide.columns:
        fig_mix_y.add_bar(name=col, x=y_wide.index.astype(str), y=y_wide[col])
    fig_mix_y.update_layout(
        barmode="stack",
        title="Invoice-Type Mix by Year",
        xaxis_title="Year",
        yaxis_title="Amount",
        yaxis_tickprefix="₱",
        yaxis_separatethousands=True,
        template="plotly_white"
    )

# ---------- Yearly normalized ----------
    y_share = y_wide.div(y_wide.sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0) * 100.0
    fig_norm = go.Figure()
    for col in y_share.columns:
        fig_norm.add_bar(
            name=col,
            x=y_share.index.astype(str),
            y=y_share[col],
            text=y_share[col],  # values for texttemplate
        )
    # Use HTML bold inside texttemplate
    fig_norm.update_traces(
        texttemplate="<b>%{text:.1f}%</b>",  # <-- bold text
        textposition="inside",
        insidetextanchor="middle"
    )
    fig_norm.update_layout(
        barmode="stack",
        title="Invoice-Type Mix by Year (Normalized)",
        xaxis_title="Year",
        yaxis_title="Percent",
        yaxis_range=[0,100],
        yaxis_ticksuffix="%",
        template="plotly_white"
    )
    
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_mix_y, use_container_width=True)
    c2.plotly_chart(fig_norm, use_container_width=True)

st.divider()