# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 12:16:23 2025
FKTS Sales Monitor (Streamlit) — Python >= 3.9
"""

import os, re, glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from statsmodels.tsa.seasonal import STL
import string, collections

# Must be the first Streamlit command:
st.set_page_config(page_title="FKTS Sales Monitor", layout="wide")

# ---------- CONFIG ----------
INPUT_FILES = [
    r"C:\Users\LB945465\Desktop\FKTS\Daily Sales\2025 - FKTS Payment Monitoring updated.xlsm",
    r"C:\Users\LB945465\Desktop\FKTS\Daily Sales\1.FKTS Inc. -  Payment Monitoring.xlsm",
]
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

# color sequence used across charts
COLOR_SEQ = px.colors.qualitative.Dark24  # 24 distinct colors

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

# ---------- Data loading ----------
@st.cache_data
def load_workbooks(files):
    frames = []
    for pat in files:
        for fpath in sorted(glob.glob(pat)):
            try:
                book = pd.read_excel(fpath, sheet_name=None, engine="openpyxl", dtype=object, header=None)
            except Exception as e:
                st.warning(f"Skip file {os.path.basename(fpath)}: {e}")
                continue
            for sname, df in (book or {}).items():
                if df is None or df.empty:
                    continue
                data, _ = _find_header_and_rename(df)
                if data is None:
                    continue
                data["__sheet__"]  = str(sname)
                data["__source__"] = os.path.basename(fpath)
                frames.append(data)
    if not frames:
        return pd.DataFrame(columns=list(COLS.keys()))
    df = pd.concat(frames, ignore_index=True, sort=False)

    # clean
    df["transaction_date"] = df["transaction_date"].apply(_parse_date)
    df["amount"]           = df["amount"].apply(_clean_amount)
    df["company"]          = df["company"].apply(_tidy_company)
    df = df[df["company"].notna()]
    
    # canonical key for uniqueness
    df["company_key"] = df["company"].apply(_company_key)
    
    # optional: deduplicate receipts across sources (keeps first occurrence)
    DEDUP_KEYS = ["transaction_date", "receipt_no", "company_key", "amount"]
    if set(DEDUP_KEYS).issubset(df.columns):
        before = len(df)
        df = df.drop_duplicates(subset=DEDUP_KEYS, keep="first")
        # st.write(f"De-dup removed {before - len(df)} rows")  # optional debug

    # derived
    df["year"]  = df["transaction_date"].dt.year
    df["month"] = df["transaction_date"].dt.month
    df["month_key"] = df["transaction_date"].dt.strftime("%Y-%m")

    return df

# Load after page_config is set
df = load_workbooks(INPUT_FILES)

# ---------- UI ----------
st.title("FKTS Sales Monitor")

if df.empty:
    st.error("No data loaded. Check your INPUT_FILES paths.")
    st.stop()

# Sidebar filters
years = sorted(df["year"].dropna().unique().astype(int))
rtype_opts = sorted([x for x in df.get("receipt_type", pd.Series()).dropna().unique()])
with st.sidebar:
    st.header("Filters")
    year_sel = st.multiselect("Years", years, default=years[-1:] if years else [])
    rtype_sel = st.multiselect("Receipt Type", rtype_opts, default=rtype_opts)
    top_n = st.slider("Top-N companies", 5, 30, TOP_N_DEFAULT)

# Apply filters
d = df.copy()
if year_sel:
    d = d[d["year"].isin(year_sel)]
if rtype_sel and "receipt_type" in d:
    d = d[d["receipt_type"].isin(rtype_sel)]

# ---------- KPIs ----------
colk1, colk2, colk3 = st.columns(3)
total_sum = d["amount"].sum()
month_latest = d["transaction_date"].max()
unique_companies = d["company_key"].nunique()
colk1.metric("Total (filter)", f"₱{total_sum:,.0f}")
colk2.metric("Last date", month_latest.strftime("%Y-%m-%d") if pd.notna(month_latest) else "n/a")
#colk3.metric("Companies (unique)", f"{unique_companies:,}")
# (Optional) show per-year actives to sanity-check
per_year = d.dropna(subset=["year"]).groupby("year")["company_key"].nunique()
st.caption("Active companies per year: " + ", ".join(f"{int(y)}: {n}" for y, n in per_year.items()))

st.divider()

# ---------- Active companies per year (bar) ----------
st.subheader("Active Companies per Year")

# prefer canonical key; fall back to raw name if not present
key_col = "company_key" if "company_key" in d.columns else "company"

ac = (d.dropna(subset=["year", key_col])
        .drop_duplicates(subset=["year", key_col])       # one row per (year, company)
        .groupby("year")[key_col].size()
        .reset_index(name="active_companies")
        .sort_values("year"))

if ac.empty:
    st.info("No data available for the selected filters.")
else:
    fig_ac = px.bar(
        ac, x="year", y="active_companies",
        title="Active Companies per Year (unique within each year)",
        text="active_companies"
    )
    fig_ac.update_traces(textposition="outside")
    # add a little headroom so labels don't clip
    ymax = ac["active_companies"].max()
    fig_ac.update_yaxes(range=[0, ymax * 1.15])
    st.plotly_chart(fig_ac, use_container_width=True)

# ---------- Row 1: Monthly totals & Top-N ----------
c1, c2 = st.columns(2)

# Monthly totals
ts = d.groupby("month_key", dropna=False)["amount"].sum().rename("monthly_total").reset_index()
ts["month_dt"] = pd.to_datetime(ts["month_key"], format="%Y-%m", errors="coerce")
ts = ts.dropna().sort_values("month_dt")
fig_ts = px.line(ts, x="month_dt", y="monthly_total", markers=True, title="Monthly Total Sales")
fig_ts.update_yaxes(tickprefix="₱", separatethousands=True)
c1.plotly_chart(fig_ts, use_container_width=True)

# Top-N companies (current filter)
ytd = d.groupby("company", dropna=False)["amount"].sum().reset_index()
ytd = ytd.sort_values("amount", ascending=False)
fig_top = px.bar(ytd.head(top_n)[::-1], x="amount", y="company", orientation="h",
                 title=f"Top {top_n} Companies (current filter)")
fig_top.update_xaxes(tickprefix="₱", separatethousands=True)
c2.plotly_chart(fig_top, use_container_width=True)

# ---------- Row 2: Share pie & Run-rate ----------
c3, c4 = st.columns(2)

# Share pie (Top-10 + Others) with truncated labels
pie_topn = 10
pie = ytd.copy()
others = pie["amount"][pie_topn:].sum()
pie = pie.head(pie_topn)
if others > 0:
    pie = pd.concat([pie, pd.DataFrame({"company":["Others"], "amount":[others]})], ignore_index=True)
pie["label"] = pie["company"].str.slice(0, PIE_LABEL_MAX) + np.where(pie["company"].str.len()>PIE_LABEL_MAX, "…", "")
fig_pie = px.pie(
    pie, names="label", values="amount",
    title=f"Share of Total Sales (Top-{pie_topn} + Others)",
    color_discrete_sequence=COLOR_SEQ
)
c3.plotly_chart(fig_pie, use_container_width=True)

# Run-rate gauge: prev YTD vs current YTD vs projection
if d["transaction_date"].notna().any():
    last_dt = d["transaction_date"].max()
    yr, mo = int(last_dt.year), int(last_dt.month)
    cur_ytd  = d[(d["transaction_date"].dt.year==yr) & (d["transaction_date"].dt.month<=mo)]["amount"].sum()
    prev_ytd = d[(d["transaction_date"].dt.year==yr-1) & (d["transaction_date"].dt.month<=mo)]["amount"].sum()
    proj = cur_ytd * (12.0/mo) if mo>0 else np.nan
    gauge = pd.DataFrame({"metric":["Prev YTD","Current YTD","Projection"], "value":[prev_ytd,cur_ytd,proj]})
    fig_g = px.bar(gauge, x="value", y="metric", orientation="h",
                   title=f"Run-rate gauge • YTD to {yr}-{mo:02d}")
    fig_g.update_xaxes(tickprefix="₱", separatethousands=True)
    c4.plotly_chart(fig_g, use_container_width=True)

st.divider()

# ---------- Row 3: Invoice-type mix (monthly/yearly + normalized) ----------
if "receipt_type" in d.columns and d["receipt_type"].notna().any():
    # Monthly stacked
    m = d.groupby(["month_key","receipt_type"])["amount"].sum().reset_index()
    m["month_dt"] = pd.to_datetime(m["month_key"])
    fig_mix_m = px.bar(
        m.sort_values("month_dt"), x="month_dt", y="amount", color="receipt_type",
        barmode="stack", title="Invoice-Type Mix by Month",
        color_discrete_sequence=COLOR_SEQ
    )
    fig_mix_m.update_yaxes(tickprefix="₱", separatethousands=True)
    st.plotly_chart(fig_mix_m, use_container_width=True)

    # Yearly stacked (absolute)
    y = d.groupby(["year","receipt_type"])["amount"].sum().reset_index()
    fig_mix_y = px.bar(
        y, x="year", y="amount", color="receipt_type", barmode="stack",
        title="Invoice-Type Mix by Year",
        color_discrete_sequence=COLOR_SEQ
    )
    fig_mix_y.update_yaxes(tickprefix="₱", separatethousands=True)
    st.plotly_chart(fig_mix_y, use_container_width=True)

    # Yearly normalized (100%) with % annotations
    y_norm = y.pivot(index="year", columns="receipt_type", values="amount").fillna(0.0)
    y_share = (y_norm.div(y_norm.sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0) * 100.0)
    y_share_m = y_share.reset_index().melt(id_vars="year", var_name="receipt_type", value_name="pct")
    fig_norm = px.bar(
        y_share_m, x="year", y="pct", color="receipt_type", barmode="stack",
        title="Invoice-Type Mix by Year (Normalized)",
        text="pct", color_discrete_sequence=COLOR_SEQ
    )
    fig_norm.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    fig_norm.update_yaxes(range=[0,100], ticksuffix="%")
    st.plotly_chart(fig_norm, use_container_width=True)

st.divider()

# ---------- STL decomposition (computed on the fly for current filter) ----------
with st.expander("STL Decomposition (monthly totals)", expanded=False):
    ts2 = d.groupby("month_key")["amount"].sum().reset_index()
    ts2["month_dt"] = pd.to_datetime(ts2["month_key"], format="%Y-%m", errors="coerce")
    ts2 = ts2.dropna().sort_values("month_dt")
    if not ts2.empty:
        s = ts2.set_index("month_dt")["amount"].copy()
        s.index = s.index.to_period("M").to_timestamp("M")
        full_idx = pd.period_range(s.index.min().to_period("M"), s.index.max().to_period("M"), freq="M").to_timestamp("M")
        s = s.reindex(full_idx).fillna(0.0)
        res = STL(s, period=12, robust=True).fit()
        comp = pd.DataFrame({
            "date": s.index,
            "Observed": s.values,
            "Trend": res.trend,
            "Seasonal": res.seasonal,
            "Residual": res.resid
        })
        st.line_chart(comp.set_index("date")[["Observed"]])
        st.line_chart(comp.set_index("date")[["Trend"]])
        #st.line_chart(comp.set_index("date")[["Seasonal"]])
        #st.line_chart(comp.set_index("date")[["Residual"]])
    else:
        st.info("Not enough monthly data after filters.")
