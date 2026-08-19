"""
Multi Data Analyst — v9.0
===========================
5-Agent AI · Domain-Aware · Quality-Gate · Gallery
"""

import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import traceback

from dotenv import load_dotenv
import os

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Query Lab Groq configuration
_QL_GROQ_KEY = GROQ_API_KEY
_QL_GROQ_MODEL = GROQ_MODEL
QL_AVAILABLE = bool(_QL_GROQ_KEY)

print("Loaded key:", GROQ_API_KEY[:8])

def _ql_query(prompt: str) -> str:
    """Send a Query Lab prompt to Groq and return the text response."""
    if not QL_AVAILABLE:
        return "⚠ Groq unavailable. Add GROQ_API_KEY to your .env file."
    try:
        from groq import Groq as _Groq
        client = _Groq(api_key=_QL_GROQ_KEY)
        resp = client.chat.completions.create(
            model=_QL_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠ Groq error: {e}"

def _build_schema_context(df) -> str:
    """Build a compact schema description from a DataFrame for Gemini prompts."""
    if df is None:
        return "No dataset loaded yet."
    lines = [f"Table: uploaded_data  ({len(df)} rows, {len(df.columns)} columns)"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(3).tolist()
        lines.append(f"  - {col} ({dtype}): sample {sample}")
    return "\n".join(lines)

def optimize_sql(query: str, schema: str) -> str:
    prompt = f"""You are a senior SQL performance engineer.
Given this database schema:
{schema}

Optimize the following SQL query. Return:
1. The optimized SQL (in a ```sql code block)
2. A brief explanation of each optimization made
3. Expected performance improvements

Original query:
{query}"""
    return _ql_query(prompt)

def optimize_dax(query: str, schema: str) -> str:
    prompt = f"""You are a senior Power BI / DAX expert.
Given this data schema:
{schema}

Optimize the following DAX measure/query. Return:
1. The optimized DAX (in a ```dax code block)
2. A brief explanation of each optimization
3. Notes on context transition or CALCULATE improvements

Original DAX:
{query}"""
    return _ql_query(prompt)

def auto_generate_sql(df) -> str:
    schema = _build_schema_context(df)
    prompt = f"""You are a SQL expert. Given this dataset schema:
{schema}

Generate 5 useful SQL queries a data analyst would run on this data.
For each query provide:
- A short label (e.g. "Top 10 by Revenue")
- The SQL (in a ```sql code block)
Keep queries practical, relevant to the column names, and progressively complex.
"""
    return _ql_query(prompt)

def auto_generate_dax(df) -> str:
    schema = _build_schema_context(df)
    prompt = f"""You are a Power BI DAX expert. Given this dataset schema:
{schema}

Generate 5 useful DAX measures a Power BI developer would create for this data.
For each measure provide:
- A short label (e.g. "Total Revenue")
- The DAX (in a ```dax code block)
Keep measures practical and relevant to the column names.
"""
    return _ql_query(prompt)

# ─────────────────────────────────────────────────────────────────────────────

from utils.data_processor import DataProcessor
from utils.visualizer import DataVisualizer

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="analyst-LLM — AI Data Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── RESET & BASE ─────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}

/* ── DESIGN TOKENS ────────────────────────────────── */
:root{
  --bg:      #0d0f14;
  --bg2:     #12151d;
  --bg3:     #181c28;
  --card:    #1a1f2e;
  --card2:   #1e2435;
  --acc:     #7c6aff;
  --acc2:    #5b4ef5;
  --acc-dim: rgba(124,106,255,.12);
  --acc-glow:rgba(124,106,255,.28);
  --cyan:    #38bdf8;
  --green:   #34d399;
  --amber:   #fbbf24;
  --red:     #f87171;
  --b1: rgba(255,255,255,.06);
  --b2: rgba(255,255,255,.10);
  --b3: rgba(255,255,255,.18);
  --t1: #f1f3ff;
  --t2: #9399b8;
  --t3: #545b78;
  --r:  10px;
  --r2: 14px;
  --r3: 20px;
  --dur: .18s;
  --ease: cubic-bezier(.16,1,.3,1);
  --sh:  0 0 0 1px var(--b1), 0 4px 24px rgba(0,0,0,.6);
  --sh2: 0 0 0 1px rgba(124,106,255,.35), 0 4px 24px rgba(124,106,255,.2);
}

/* ── STREAMLIT SCAFFOLDING ────────────────────────── */
#MainMenu,footer,.stDeployButton,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
/* Keep header visible (contains sidebar toggle) but make it transparent */
[data-testid="stHeader"]{background:transparent!important;border-bottom:none!important;}
[data-testid="stHeader"] [data-testid="stMainMenuButton"]{display:none!important;}
.stApp{font-family:'Inter',-apple-system,sans-serif;background:#0d0f14!important;color:#f1f3ff;}
.main .block-container{padding:2rem 2.4rem 3rem!important;max-width:1140px!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--b2);border-radius:99px;}
::-webkit-scrollbar-thumb:hover{background:var(--acc);}

/* ── SIDEBAR SHELL — always visible, never collapses ─ */
section[data-testid="stSidebar"]{
  background:#12151d!important;
  border-right:1px solid rgba(255,255,255,.06)!important;
  width:220px!important;
  min-width:220px!important;
  display:flex!important;
  transform:none!important;
  visibility:visible!important;
  opacity:1!important;
  position:relative!important;
  left:0!important;
}
/* Force open even when Streamlit sets aria-expanded=false after a rerun */
section[data-testid="stSidebar"][aria-expanded="false"]{
  display:flex!important;
  transform:none!important;
  visibility:visible!important;
  width:220px!important;
  min-width:220px!important;
}
section[data-testid="stSidebar"] > div:first-child{padding:0!important;}
section[data-testid="stSidebar"] .block-container{padding:0!important;}
/* Hide the collapse arrow button so users can't close the sidebar */
[data-testid="stSidebarCollapseButton"]{display:none!important;}
/* Hide the floating re-open chevron (sidebar is never closed) */
[data-testid="collapsedControl"]{display:none!important;}

/* ── SIDEBAR BRAND ────────────────────────────────── */
.sb-brand{padding:1.3rem 1.1rem 1rem;border-bottom:1px solid var(--b1);}
.sb-logo{display:flex;align-items:center;gap:9px;}
.sb-logo-icon{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,var(--acc),var(--cyan));display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;}
.sb-logo-name{font-size:.95rem;font-weight:800;color:var(--t1);letter-spacing:-.4px;}
.sb-logo-tag{font-family:'JetBrains Mono',monospace;font-size:.45rem;color:var(--t3);letter-spacing:2px;text-transform:uppercase;margin-top:2px;}

/* ── SIDEBAR NAV (radio) ──────────────────────────── */
section[data-testid="stSidebar"] .stRadio > label{display:none!important;}
section[data-testid="stSidebar"] .stRadio > div{
  display:flex!important;
  flex-direction:column!important;
  gap:2px!important;
  padding:6px 8px!important;
}
section[data-testid="stSidebar"] .stRadio > div > label{
  display:flex!important;
  align-items:center!important;
  padding:.52rem .9rem!important;
  border-radius:var(--r)!important;
  cursor:pointer!important;
  border:1px solid transparent!important;
  transition:all var(--dur) var(--ease)!important;
  margin:0!important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover{
  background:var(--acc-dim)!important;
  border-color:var(--b2)!important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"]{
  background:var(--acc-dim)!important;
  border-color:rgba(124,106,255,.35)!important;
}
section[data-testid="stSidebar"] .stRadio input[type="radio"]{display:none!important;}
section[data-testid="stSidebar"] .stRadio > div > label p{
  font-family:'Inter',sans-serif!important;
  font-size:.82rem!important;
  font-weight:500!important;
  color:var(--t2)!important;
  margin:0!important;
  line-height:1!important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] p{
  color:var(--t1)!important;
  font-weight:600!important;
}

/* ── SIDEBAR MISC ─────────────────────────────────── */
.sb-sep{height:1px;background:var(--b1);margin:4px 10px;}
.sb-section-lbl{
  font-family:'JetBrains Mono',monospace;
  font-size:.46rem;font-weight:600;color:var(--t3);
  text-transform:uppercase;letter-spacing:2.5px;
  padding:.9rem 1.1rem .3rem;display:block;
}
.sb-status{
  display:flex;align-items:center;gap:7px;
  font-size:.73rem;color:var(--t2);
  padding:.3rem 1.1rem .2rem;
}
.sb-dot{width:6px;height:6px;border-radius:50%;background:var(--green);flex-shrink:0;}
.sb-dot.off{background:var(--t3);}
.sb-wf{padding:2px 0 6px;}
.sb-wf-row{display:flex;align-items:center;gap:8px;padding:.28rem 1rem;}
.sb-wf-num{font-family:'JetBrains Mono',monospace;font-size:.44rem;color:var(--t3);width:16px;flex-shrink:0;}
.sb-wf-dot{width:5px;height:5px;border-radius:50%;background:var(--b2);border:1px solid var(--b3);flex-shrink:0;}
.sb-wf-row.done .sb-wf-dot{background:var(--green);border-color:var(--green);}
.sb-wf-lbl{font-size:.73rem;color:var(--t3);}
.sb-wf-row.done .sb-wf-lbl{color:var(--t2);}

/* ── BUTTONS ──────────────────────────────────────── */
.stButton > button{
  font-family:'Inter',sans-serif!important;
  font-size:.82rem!important;font-weight:600!important;
  background:var(--card)!important;color:var(--t1)!important;
  border:1px solid var(--b2)!important;border-radius:var(--r)!important;
  padding:.52rem 1.1rem!important;cursor:pointer!important;
  transition:all var(--dur) var(--ease)!important;
  box-shadow:none!important;
}
.stButton > button:hover{
  background:var(--acc-dim)!important;
  border-color:rgba(124,106,255,.4)!important;
  color:var(--t1)!important;
  transform:translateY(-1px)!important;
  box-shadow:var(--sh2)!important;
}
.stButton > button:active{transform:translateY(0)!important;}
.stButton > button[kind="primary"]{
  background:var(--acc)!important;border-color:transparent!important;
  color:#fff!important;box-shadow:0 2px 16px var(--acc-glow)!important;
}
.stButton > button[kind="primary"]:hover{background:var(--acc2)!important;transform:translateY(-1px)!important;}

/* ── TABS ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{
  background:transparent!important;
  border-bottom:1px solid rgba(255,255,255,.06)!important;
  gap:0!important;padding:0!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;border:none!important;border-radius:0!important;
  font-family:'Inter',sans-serif!important;font-size:.8rem!important;
  font-weight:600!important;color:#545b78!important;
  padding:.65rem 1rem!important;
  border-bottom:2px solid transparent!important;
  transition:color .15s ease,border-color .15s ease!important;
  outline:none!important;box-shadow:none!important;
}
.stTabs [data-baseweb="tab"]:hover{color:#f1f3ff!important;}
.stTabs [aria-selected="true"]{
  color:#7c6aff!important;
  border-bottom:2px solid #7c6aff!important;
  background:transparent!important;
}
/* Override BaseWeb internal highlight */
.stTabs [data-baseweb="tab"] [data-highlighted="true"]{background:transparent!important;}
.stTabs [data-baseweb="tab-highlight"]{background:#7c6aff!important;height:2px!important;}
.stTabs [data-baseweb="tab-border"]{background:rgba(255,255,255,.06)!important;}
.stTabs [data-baseweb="tab-panel"]{padding:1.4rem 0 0!important;}

/* ── INPUTS ───────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea{
  background:var(--card)!important;color:var(--t1)!important;
  border:1px solid var(--b2)!important;border-radius:var(--r)!important;
  font-family:'Inter',sans-serif!important;font-size:.84rem!important;
  transition:border-color var(--dur) ease!important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus{
  border-color:var(--acc)!important;
  box-shadow:0 0 0 3px var(--acc-dim)!important;
}
.stTextArea > div > div > textarea{font-family:'JetBrains Mono',monospace!important;font-size:.79rem!important;}
.stSelectbox > div > div{
  background:var(--card)!important;color:var(--t1)!important;
  border:1px solid var(--b2)!important;border-radius:var(--r)!important;
}
.stSelectbox label,.stTextInput label,.stTextArea label{
  font-family:'Inter',sans-serif!important;font-size:.76rem!important;
  font-weight:600!important;color:var(--t2)!important;letter-spacing:.1px!important;
}
[data-testid="stFileUploader"]{
  background:var(--card)!important;border:2px dashed var(--b2)!important;
  border-radius:var(--r2)!important;padding:2rem!important;
  transition:border-color var(--dur) ease!important;
}
[data-testid="stFileUploader"]:hover{border-color:var(--acc)!important;}
.stProgress > div > div{background:linear-gradient(90deg,var(--acc),var(--cyan))!important;border-radius:99px!important;}
.stProgress > div{background:var(--b1)!important;border-radius:99px!important;}
.stSlider [role="slider"]{background:var(--acc)!important;border-color:var(--acc)!important;}

/* ── DATAFRAME ────────────────────────────────────── */
[data-testid="stDataFrame"]{border-radius:var(--r);overflow:hidden;border:1px solid var(--b1);}
.stDataFrame th{
  background:var(--bg3)!important;color:var(--t2)!important;
  font-family:'JetBrains Mono',monospace!important;font-size:.67rem!important;
  font-weight:600!important;letter-spacing:.5px!important;text-transform:uppercase!important;
}
.stDataFrame td{
  font-family:'Inter',sans-serif!important;font-size:.8rem!important;
  color:var(--t1)!important;background:var(--card)!important;
}
.streamlit-expanderHeader{
  font-family:'Inter',sans-serif!important;font-size:.83rem!important;
  font-weight:600!important;color:var(--t1)!important;
  background:var(--card)!important;border:1px solid var(--b1)!important;
  border-radius:var(--r)!important;
}
hr{border:none!important;border-top:1px solid var(--b1)!important;margin:1.4rem 0!important;}

/* ══════════════════════════════════════════════════
   PAGE COMPONENTS
   ══════════════════════════════════════════════════ */

/* ── PAGE HERO BANNER ─────────────────────────────── */
.pg-hero{
  display:flex;align-items:flex-start;gap:1.4rem;
  background:linear-gradient(135deg,rgba(124,106,255,.1) 0%,rgba(56,189,248,.05) 100%);
  border:1px solid rgba(124,106,255,.2);
  border-left:3px solid var(--acc);border-radius:var(--r2);
  padding:1.6rem 1.8rem;margin-bottom:1.8rem;
}
.pg-hero-icon{font-size:1.8rem;flex-shrink:0;margin-top:.1rem;}
.pg-hero-eyebrow{
  font-family:'JetBrains Mono',monospace;font-size:.52rem;font-weight:600;
  color:var(--acc);text-transform:uppercase;letter-spacing:2px;
  display:block;margin-bottom:.4rem;
}
.pg-hero-title{
  font-size:1.5rem;font-weight:800;color:var(--t1);
  letter-spacing:-.6px;margin-bottom:.3rem;line-height:1.2;
}
.pg-hero-title .hi{
  background:linear-gradient(120deg,var(--acc),var(--cyan));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.pg-hero-desc{font-size:.84rem;color:var(--t2);line-height:1.65;margin:0;}

/* ── HOME HERO ────────────────────────────────────── */
.home-hero{padding:2.8rem 0 1.5rem;animation:fadeUp .4s var(--ease) both;}
.home-eyebrow{
  display:inline-flex;align-items:center;gap:6px;
  background:var(--acc-dim);border:1px solid rgba(124,106,255,.25);
  border-radius:99px;padding:4px 14px;
  font-family:'JetBrains Mono',monospace;font-size:.56rem;
  font-weight:600;color:var(--acc);letter-spacing:.5px;
  margin-bottom:1rem;text-transform:uppercase;
}
.home-eyebrow::before{content:'●';font-size:.35rem;}
.home-title{font-size:2.6rem;font-weight:900;color:var(--t1);letter-spacing:-1.5px;line-height:1.1;margin:0 0 .9rem;}
.home-title .hi{background:linear-gradient(120deg,var(--acc),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.home-desc{font-size:1rem;color:var(--t2);line-height:1.7;max-width:540px;margin:0;}

/* ── SECTION DIVIDER ──────────────────────────────── */
.sec-rule{display:flex;align-items:center;gap:10px;margin:2rem 0 1.2rem;}
.sec-rule-label{font-family:'JetBrains Mono',monospace;font-size:.5rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:2.5px;white-space:nowrap;}
.sec-rule::after{content:'';flex:1;height:1px;background:var(--b1);}

/* ── METRIC CARDS ─────────────────────────────────── */
.metric-card{
  background:var(--card);border:1px solid var(--b1);
  border-radius:var(--r2);padding:1.2rem 1.4rem;
  transition:all var(--dur) var(--ease);text-align:left;
}
.metric-card:hover{border-color:rgba(124,106,255,.3);box-shadow:var(--sh2);transform:translateY(-2px);}
.metric-val{font-size:1.85rem;font-weight:800;color:var(--t1);letter-spacing:-1px;line-height:1;display:block;margin-bottom:.3rem;}
.metric-lbl{font-family:'JetBrains Mono',monospace;font-size:.53rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:2px;}
.metric-sub{font-size:.72rem;color:var(--t3);margin-top:.2rem;display:block;}

/* ── STEP CARDS ───────────────────────────────────── */
.step-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1.5rem;}
.step-card{background:var(--card);border:1px solid var(--b1);border-radius:var(--r2);padding:1.2rem;transition:all var(--dur) var(--ease);}
.step-card:hover{border-color:rgba(124,106,255,.3);box-shadow:var(--sh2);transform:translateY(-2px);}
.step-num-big{font-size:2rem;font-weight:900;color:var(--b2);display:block;margin-bottom:.3rem;letter-spacing:-1px;line-height:1;}
.step-card:hover .step-num-big{color:var(--acc);}
.step-icon-lg{font-size:1.25rem;display:block;margin-bottom:.45rem;}
.step-card-title{font-size:.87rem;font-weight:700;color:var(--t1);margin:0 0 .3rem;}
.step-card-desc{font-size:.75rem;color:var(--t2);margin:0;line-height:1.55;}

/* ── PIPELINE ─────────────────────────────────────── */
.pipeline-row{display:flex;align-items:center;gap:4px;overflow-x:auto;padding:4px 0;}
.pipeline-step{
  flex:1;min-width:105px;background:var(--card);border:1px solid var(--b1);
  border-radius:var(--r);padding:.85rem .7rem;text-align:center;
  transition:all var(--dur) var(--ease);
}
.pipeline-step:hover{border-color:rgba(124,106,255,.3);box-shadow:var(--sh2);}
.pip-num{font-family:'JetBrains Mono',monospace;font-size:.48rem;font-weight:600;color:var(--acc);letter-spacing:1.5px;display:block;margin-bottom:.3rem;}
.pip-icon{font-size:1.05rem;display:block;margin-bottom:.25rem;}
.pip-name{font-size:.74rem;font-weight:700;color:var(--t1);margin:0 0 .18rem;}
.pip-tag{font-family:'JetBrains Mono',monospace;font-size:.52rem;color:var(--t3);letter-spacing:.5px;}
.pipeline-arrow{color:var(--t3);font-size:.9rem;flex-shrink:0;}

/* ── QUOTE ────────────────────────────────────────── */
.data-quote{
  border-left:3px solid var(--acc);padding:.9rem 1.4rem;
  margin:1.5rem 0;background:var(--card);border-radius:0 var(--r) var(--r) 0;
}
.data-quote blockquote{font-size:.98rem;font-weight:600;color:var(--t1);margin:0 0 .35rem;line-height:1.55;font-style:normal;}
.data-quote cite{font-family:'JetBrains Mono',monospace;font-size:.54rem;font-weight:500;color:var(--t3);letter-spacing:1.5px;text-transform:uppercase;}

/* ── GLASS PANEL / CARDS ──────────────────────────── */
.glass-panel{background:var(--card);border:1px solid var(--b1);border-radius:var(--r2);padding:1.3rem;}
.info-card{background:var(--card);border:1px solid var(--b1);border-radius:var(--r);padding:.9rem 1.1rem;margin-bottom:.7rem;}
.info-card-title{font-size:.83rem;font-weight:700;color:var(--t1);margin-bottom:.25rem;}
.info-card-body{font-size:.78rem;color:var(--t2);line-height:1.6;}

/* ── PAGE NAV (next/back) ─────────────────────────── */
.page-nav{height:1px;background:var(--b1);margin:2.5rem 0 1rem;}

/* ── PROFILE STATS ────────────────────────────────── */
.profile-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:9px;margin-bottom:1.5rem;}
.pstat{background:var(--card);border:1px solid var(--b1);border-radius:var(--r);padding:.95rem 1rem;transition:border-color var(--dur) ease;}
.pstat:hover{border-color:rgba(124,106,255,.3);}
.pstat-key{font-family:'JetBrains Mono',monospace;font-size:.57rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:1.5px;display:block;margin-bottom:.3rem;}
.pstat-val{font-size:1.05rem;font-weight:700;color:var(--t1);letter-spacing:-.4px;display:block;}
.pstat-sub{font-size:.7rem;color:var(--t3);margin-top:.15rem;display:block;}
.missing-high{color:var(--red)!important;}.missing-mid{color:var(--amber)!important;}.missing-low{color:var(--green)!important;}

/* ── STATUS BADGES ────────────────────────────────── */
.status-badge{
  display:inline-flex;align-items:center;gap:6px;
  font-size:.73rem;font-weight:500;color:var(--t2);
  padding:.3rem .85rem;background:var(--bg3);
  border-radius:99px;border:1px solid var(--b1);
  width:fit-content;
}
.s-dot{width:6px;height:6px;border-radius:50%;background:var(--green);flex-shrink:0;}
.s-dot.off{background:var(--t3);}
.s-dot.warn{background:var(--amber);}

/* ── AI RESULTS ───────────────────────────────────── */
.result-header{
  background:var(--card);border:1px solid var(--b1);border-radius:var(--r2);
  padding:1.2rem 1.4rem;margin-bottom:1.2rem;
  display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:1rem;
}
.result-title{font-size:.98rem;font-weight:700;color:var(--t1);margin:0 0 .18rem;letter-spacing:-.2px;}
.result-sub{font-size:.74rem;color:var(--t2);margin:0;}
.qg-metric{background:var(--bg3);border:1px solid var(--b1);border-radius:var(--r);padding:.45rem .75rem;text-align:center;min-width:60px;}
.qg-metric-val{display:block;font-size:1.1rem;font-weight:800;color:var(--t1);}
.qg-metric-lbl{display:block;font-family:'JetBrains Mono',monospace;font-size:.48rem;color:var(--t3);text-transform:uppercase;letter-spacing:1.5px;}
.verdict-badge{display:inline-flex;align-items:center;gap:6px;font-size:.86rem;font-weight:700;padding:.45rem 1rem;border-radius:99px;border:1px solid;}
.verdict-pass{background:rgba(52,211,153,.1);border-color:rgba(52,211,153,.3);color:var(--green);}
.verdict-conditional{background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.3);color:var(--amber);}
.verdict-fail{background:rgba(248,113,113,.1);border-color:rgba(248,113,113,.3);color:var(--red);}
.qg-block{background:var(--card);border:1px solid var(--b1);border-radius:var(--r);padding:1rem 1.2rem;margin:.5rem 0;}
.dl-card{background:var(--card);border:1px solid var(--b1);border-radius:var(--r);padding:.95rem 1.1rem;margin-bottom:.8rem;}
.dl-icon{font-size:1.3rem;display:block;margin-bottom:.4rem;}
.dl-title{font-size:.86rem;font-weight:700;color:var(--t1);margin:0 0 .18rem;}
.dl-desc{font-size:.74rem;color:var(--t2);margin:0;}

/* ── QUERY LAB ────────────────────────────────────── */
.ql-badge{display:inline-flex;align-items:center;gap:5px;border-radius:99px;padding:3px 11px;font-family:'JetBrains Mono',monospace;font-size:.55rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-right:6px;margin-bottom:.5rem;}
.ql-badge.sql{background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.25);color:var(--cyan);}
.ql-badge.dax{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.25);color:var(--amber);}
.ql-badge.gemini{background:var(--acc-dim);border:1px solid rgba(124,106,255,.25);color:var(--acc);}
.ql-section{background:var(--card);border:1px solid var(--b1);border-radius:var(--r2);padding:1.5rem;margin-bottom:1.1rem;}
.ql-section-title{font-size:.97rem;font-weight:700;color:var(--t1);margin:0 0 .18rem;}
.ql-section-sub{font-size:.79rem;color:var(--t2);margin:0 0 1.1rem;}
.ql-result{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:1.1rem 1.3rem;margin-top:.9rem;}
.ql-result-label{font-family:'JetBrains Mono',monospace;font-size:.5rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:2px;display:block;margin-bottom:.5rem;}
.ql-tip{background:rgba(124,106,255,.06);border:1px solid rgba(124,106,255,.15);border-radius:var(--r);padding:.7rem .95rem;margin-bottom:.9rem;font-size:.79rem;color:var(--t2);}
.ql-tip strong{color:var(--acc);}

/* ── GALLERY EMPTY ────────────────────────────────── */
.gal-empty{text-align:center;padding:3.5rem 2rem;}
.gal-empty-icon{font-size:1.9rem;display:block;margin-bottom:.7rem;}
.gal-empty-txt{font-size:.98rem;font-weight:600;color:var(--t2);margin:0 0 .35rem;}
.gal-empty-sub{font-size:.79rem;color:var(--t3);margin:0;}

/* ── TOAST / ALERT ────────────────────────────────── */
.toast-success{
  display:flex;align-items:center;gap:10px;
  background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);
  border-radius:var(--r);padding:.75rem 1.1rem;margin:.5rem 0;
}
.toast-icon{font-size:.95rem;flex-shrink:0;color:var(--green);}
.toast-text{font-size:.83rem;font-weight:500;color:var(--green);}
.alert-box{
  background:var(--card);border:1px solid var(--b2);border-radius:var(--r2);
  padding:2.5rem;text-align:center;
}
.alert-icon{font-size:1.6rem;margin-bottom:.7rem;}
.alert-title{font-size:1rem;font-weight:700;color:var(--t1);margin:0 0 .3rem;}
.alert-desc{font-size:.82rem;color:var(--t2);margin:0;}
.error-card{background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.25);border-radius:var(--r);padding:1rem 1.2rem;}
.error-title{font-size:.9rem;font-weight:700;color:var(--red);margin:0 0 .25rem;}
.error-text{font-size:.8rem;color:var(--t2);margin:0;}

/* ── FLOW GUIDE ───────────────────────────────────── */
.flow-guide{max-width:680px;margin-bottom:2rem;}
.flow-item{display:grid;grid-template-columns:3rem 1fr;gap:0 1.2rem;padding:1.4rem 0;border-bottom:1px solid var(--b1);animation:fadeUp .35s var(--ease) both;}
.flow-item:last-child{border-bottom:none;}
.flow-num{font-size:2.2rem;font-weight:900;color:var(--b1);line-height:1;letter-spacing:-2px;padding-top:.05rem;transition:color var(--dur) ease;}
.flow-item:hover .flow-num{color:var(--acc);}
.flow-tag{font-family:'JetBrains Mono',monospace;font-size:.52rem;font-weight:600;color:var(--acc);text-transform:uppercase;letter-spacing:2px;display:inline-block;margin-bottom:.25rem;}
.flow-title{font-size:.95rem;font-weight:700;color:var(--t1);letter-spacing:-.3px;margin:0 0 .25rem;}
.flow-desc{font-size:.82rem;color:var(--t2);line-height:1.65;margin:0;}

/* ── TOP ACCENT BAR ───────────────────────────────── */
.iq-topbar{
  position:fixed;top:0;left:0;right:0;height:2px;z-index:9999;
  background:linear-gradient(90deg,var(--acc),var(--cyan),var(--acc));
  background-size:200% 100%;animation:shimmer 4s linear infinite;
}

/* ── AMBIENT GLOW ─────────────────────────────────── */
.iq-bg{position:fixed;inset:0;pointer-events:none;z-index:-1;overflow:hidden;}
.iq-glow1{position:absolute;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(124,106,255,.05) 0%,transparent 70%);top:-100px;left:-80px;animation:driftA 22s ease-in-out infinite;}
.iq-glow2{position:absolute;width:420px;height:420px;border-radius:50%;background:radial-gradient(circle,rgba(56,189,248,.03) 0%,transparent 70%);bottom:-100px;right:-80px;animation:driftB 28s ease-in-out infinite;}

/* ── ANIMATIONS ───────────────────────────────────── */
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.05)}}
@keyframes driftA{0%,100%{transform:translate(0,0)}50%{transform:translate(35px,25px)}}
@keyframes driftB{0%,100%{transform:translate(0,0)}50%{transform:translate(-25px,-35px)}}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
@keyframes spinRing{to{transform:rotate(360deg)}}
@keyframes dotBounce{0%,80%,100%{transform:translateY(0);opacity:.4}40%{transform:translateY(-7px);opacity:1}}
@keyframes pageFlash{0%{opacity:.55}100%{opacity:0;pointer-events:none}}
@keyframes barGlow{0%,100%{box-shadow:0 0 0px var(--acc)}50%{box-shadow:0 0 10px var(--acc)}}
@keyframes slideInRight{from{opacity:0;transform:translateX(18px)}to{opacity:1;transform:translateX(0)}}
@keyframes ripple{0%{transform:scale(0);opacity:.5}100%{transform:scale(3.5);opacity:0}}
@keyframes skeletonWave{0%{background-position:-400px 0}100%{background-position:400px 0}}

/* ── PAGE TRANSITION FLASH ────────────────────────── */
.page-flash{position:fixed;inset:0;z-index:9999;background:linear-gradient(135deg,rgba(124,106,255,.12),rgba(56,189,248,.08));pointer-events:none;animation:pageFlash .45s cubic-bezier(.4,0,.2,1) both;}
.page-body{animation:slideInRight .28s var(--ease) both;}

/* ── DOT BOUNCE LOADER ────────────────────────────── */
.iq-dots{display:inline-flex;align-items:center;gap:5px;padding:0 2px;}
.iq-dots span{width:6px;height:6px;border-radius:50%;background:var(--acc);animation:dotBounce 1.2s ease-in-out infinite;}
.iq-dots span:nth-child(2){animation-delay:.2s;}
.iq-dots span:nth-child(3){animation-delay:.4s;}
.iq-processing{display:flex;align-items:center;gap:10px;background:var(--card);border:1px solid var(--b1);border-radius:var(--r);padding:.7rem 1.1rem;margin:.6rem 0;font-size:.82rem;color:var(--t2);}
.iq-processing strong{color:var(--t1);font-size:.84rem;}

/* ── SPIN RING (inline use) ───────────────────────── */
.spin-ring{display:inline-block;width:14px;height:14px;border-radius:50%;border:2px solid rgba(124,106,255,.25);border-top-color:var(--acc);animation:spinRing .7s linear infinite;flex-shrink:0;}

/* ── SKELETON SHIMMER ─────────────────────────────── */
.skel{border-radius:6px;background:linear-gradient(90deg,var(--bg3) 25%,var(--bg2) 50%,var(--bg3) 75%);background-size:400px 100%;animation:skeletonWave 1.4s ease-in-out infinite;}
.skel-line{height:12px;margin-bottom:8px;}.skel-line.w80{width:80%;}.skel-line.w60{width:60%;}.skel-line.w40{width:40%;}
.skel-block{height:80px;width:100%;margin-bottom:10px;}

/* ── NAV BUTTON ANIMATIONS ────────────────────────── */
[data-testid='stButton'] button{position:relative;overflow:hidden;transition:all .15s var(--ease) !important;}
[data-testid='stButton'] button::after{content:'';position:absolute;inset:50% 50%;width:0;height:0;border-radius:50%;background:rgba(124,106,255,.2);transform:translate(-50%,-50%) scale(0);transition:none;pointer-events:none;}
[data-testid='stButton'] button:active{transform:scale(.96) !important;}
[data-testid='stButton'] button:active::after{width:200%;height:200%;opacity:0;transition:all .4s ease-out !important;}

/* ── PROGRESS BAR GLOW ────────────────────────────── */
[data-testid='stProgress'] > div > div{animation:barGlow 1.8s ease-in-out infinite !important;transition:width .3s ease !important;}

/* ── STREAMLIT SPINNER CUSTOM ─────────────────────── */
[data-testid='stSpinner'] > div{gap:10px !important;}
[data-testid='stSpinner'] svg{stroke:var(--acc) !important;}

/* ── LEGACY COMPAT (keep old class names working) ─── */
.hi{background:linear-gradient(120deg,var(--acc),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.gradient-text{background:linear-gradient(120deg,var(--acc),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.m-hero{background:var(--card);border:1px solid var(--b1);border-left:3px solid var(--acc);border-radius:var(--r);padding:1.6rem 1.9rem;margin-bottom:1.6rem;}
.m-hero-eyebrow{font-family:'JetBrains Mono',monospace;font-size:.56rem;font-weight:600;color:var(--acc);text-transform:uppercase;letter-spacing:2px;display:block;margin-bottom:.5rem;}
.m-hero-title{font-size:1.4rem;font-weight:800;color:var(--t1);letter-spacing:-.5px;margin:0 0 .3rem;}
.m-hero-sub{font-size:.82rem;color:var(--t2);margin:0;}
.m-brand{padding:1.3rem 1.1rem 1rem;border-bottom:1px solid var(--b1);}
.m-logo{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:7px;background:linear-gradient(135deg,var(--acc),var(--cyan));font-size:13px;margin-bottom:.3rem;}
.m-brand-name{font-size:.93rem;font-weight:800;color:var(--t1);display:block;}
.m-brand-sub{font-family:'JetBrains Mono',monospace;font-size:.44rem;color:var(--t3);letter-spacing:2px;text-transform:uppercase;display:block;margin-top:2px;}
.m-status{display:flex;align-items:center;gap:7px;font-size:.73rem;color:var(--t2);padding:.3rem .9rem;background:var(--bg3);border-radius:99px;border:1px solid var(--b1);width:fit-content;}
.m-dot{width:6px;height:6px;border-radius:50%;background:var(--green);}.m-dot.off{background:var(--t3);}
.nav-section-label{font-family:'JetBrains Mono',monospace;font-size:.46rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:2.5px;padding:.9rem 1.1rem .3rem;margin:0;display:block;}
.pg-hdr{padding:1.6rem 0 1rem;border-bottom:1px solid var(--b1);margin-bottom:1.6rem;}
.pg-hdr-eyebrow{display:inline-flex;align-items:center;gap:5px;background:var(--acc-dim);border:1px solid rgba(124,106,255,.25);border-radius:99px;padding:3px 12px;font-family:'JetBrains Mono',monospace;font-size:.54rem;font-weight:600;color:var(--acc);letter-spacing:.5px;margin-bottom:.6rem;}
.pg-hdr-title{font-size:1.55rem;font-weight:800;color:var(--t1);letter-spacing:-.7px;margin:0 0 .3rem;}
.pg-hdr-desc{font-size:.83rem;color:var(--t2);margin:0;}
.hero-section{padding:3rem 0 2rem;}
.hero-title{font-size:2.5rem;font-weight:900;color:var(--t1);letter-spacing:-1.5px;line-height:1.1;margin:0 0 .9rem;}
.hero-desc{font-size:1rem;color:var(--t2);line-height:1.7;max-width:540px;margin:0;}
.hero-wrap{padding:2.8rem 0 1.5rem;}
.hero-eyebrow{display:inline-flex;align-items:center;gap:6px;background:var(--acc-dim);border:1px solid rgba(124,106,255,.25);border-radius:99px;padding:4px 14px;font-family:'JetBrains Mono',monospace;font-size:.55rem;font-weight:600;color:var(--acc);letter-spacing:.5px;margin-bottom:1rem;}
.hero-eyebrow::before{content:'●';font-size:.35rem;}
.cta-note{font-family:'Inter',sans-serif;font-size:.77rem;color:var(--t3);line-height:1.5;margin:0;}
.cta-note code{background:var(--bg3);border:1px solid var(--b1);border-radius:4px;padding:1px 5px;font-size:.7rem;}
.s-rule{display:flex;align-items:center;gap:10px;margin:2rem 0 1.2rem;}
.s-rule-label{font-family:'JetBrains Mono',monospace;font-size:.5rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:2.5px;white-space:nowrap;}
.s-rule::after{content:'';flex:1;height:1px;background:var(--b1);}
.steps-label{font-family:'JetBrains Mono',monospace;font-size:.5rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:2.5px;display:block;margin-bottom:.9rem;}
.steps-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1.5rem;}
.wf-steps{padding:.4rem 0 .3rem;}.wf-step{display:flex;align-items:center;gap:8px;padding:.26rem 1rem;}
.wf-step-num{font-family:'JetBrains Mono',monospace;font-size:.44rem;font-weight:600;color:var(--t3);letter-spacing:1px;flex-shrink:0;width:16px;}
.wf-step-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0;background:var(--b2);border:1px solid var(--b3);}
.wf-step.done .wf-step-dot{background:var(--green);border-color:var(--green);}
.wf-step-label{font-size:.73rem;color:var(--t3);}
.wf-step.done .wf-step-label{color:var(--t2);}
.page-nav-rule{height:1px;background:var(--b1);margin:2rem 0 1rem;}
.insight-header{display:flex;align-items:flex-start;justify-content:space-between;background:var(--card);border:1px solid var(--b1);border-radius:var(--r);padding:1.2rem 1.4rem;margin-bottom:1.2rem;gap:1rem;flex-wrap:wrap;}
.insight-title{font-size:.98rem;font-weight:700;color:var(--t1);margin:0 0 .18rem;}
.insight-sub{font-size:.74rem;color:var(--t2);margin:0;}

/* ── MISSING VARS ─────────────────────────────────── */
:root{
  --coral:    #f87171;
  --coral-lt: #fb9494;
  --coral-dim:rgba(248,113,113,.1);
  --border2:  rgba(255,255,255,.1);
}

/* ── SECTION HEADER ───────────────────────────────── */
.section-header{margin:1.1rem 0 .7rem;}
.section-header h3{font-size:.93rem;font-weight:700;color:var(--t1);letter-spacing:-.2px;margin:0;}

/* ── PAGE HERO (analytics / viz pages) ───────────── */
.hero-section{padding:1.4rem 0 .5rem;margin-bottom:1.4rem;}
.hero-icon{font-size:1.5rem;margin-bottom:.35rem;display:block;}
.hero-section .hero-title{font-size:1.5rem;font-weight:800;color:var(--t1);letter-spacing:-.6px;margin:0 0 .3rem;line-height:1.2;}
.hero-section .hero-desc{font-size:.84rem;color:var(--t2);margin:0;}

/* ── GALLERY CARD ─────────────────────────────────── */
.gal-card{background:var(--card);border:1px solid var(--b1);border-radius:var(--r);padding:.7rem .9rem;margin-bottom:.5rem;}
.gal-card-header{display:flex;align-items:center;justify-content:space-between;gap:.5rem;}
.gal-card-title{font-size:.82rem;font-weight:600;color:var(--t1);}
.gal-card-type{font-family:'JetBrains Mono',monospace;font-size:.5rem;font-weight:600;color:var(--acc);background:var(--acc-dim);border:1px solid rgba(124,106,255,.2);border-radius:99px;padding:2px 8px;text-transform:uppercase;letter-spacing:1px;}

/* ── DOMAIN BAR (AI Engine) ───────────────────────── */
.domain-bar{display:flex;align-items:center;gap:.65rem;flex-wrap:wrap;background:var(--card);border:1px solid var(--b1);border-radius:var(--r);padding:.65rem 1rem;margin:.8rem 0 1.2rem;}
.domain-label{font-family:'JetBrains Mono',monospace;font-size:.48rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:2px;}
.domain-pill{background:var(--acc-dim);border:1px solid rgba(124,106,255,.25);border-radius:99px;padding:3px 12px;font-size:.79rem;font-weight:600;color:var(--acc);}

/* ── AI PIPELINE STEP (inline labels in page_ai) ─── */
.step-num{font-family:'JetBrains Mono',monospace;font-size:.48rem;font-weight:600;color:var(--acc);letter-spacing:1.5px;display:block;margin-bottom:.3rem;}
.step-icon{font-size:1.05rem;display:block;margin-bottom:.25rem;}
.step-name{font-size:.74rem;font-weight:700;color:var(--t1);margin:0 0 .18rem;}
.step-tag{font-family:'JetBrains Mono',monospace;font-size:.52rem;color:var(--t3);letter-spacing:.5px;}

/* ── QUALITY GATE CARD ────────────────────────────── */
.qg-card{background:var(--card);border:1px solid var(--b1);border-radius:var(--r2);padding:1.2rem 1.4rem;margin-bottom:1.2rem;}
.qg-header{display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:1rem;}

/* ── AGENT MEMORY CARDS ───────────────────────────── */
.memory-card{
  background:var(--card);border:1px solid var(--b1);
  border-left:3px solid var(--mc-color,var(--acc));
  border-radius:var(--r);padding:.9rem 1.1rem;margin-bottom:.65rem;
  animation:fadeUp .3s var(--ease) both;
}
.memory-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:.4rem;}
.memory-sender{font-size:.8rem;font-weight:600;color:var(--t1);}
.memory-arrow{color:var(--t3);font-size:.8rem;}
.memory-receiver{font-size:.8rem;font-weight:600;}
.memory-time{font-family:'JetBrains Mono',monospace;font-size:.52rem;color:var(--t3);}
.memory-label{font-size:.77rem;color:var(--t2);margin-bottom:.4rem;}
.memory-preview{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--t3);background:var(--bg3);border-radius:6px;padding:.5rem .7rem;white-space:pre-wrap;word-break:break-word;max-height:100px;overflow:hidden;}
</style>
<style>
/* force Streamlit spinner color globally */
@keyframes _spin{to{transform:rotate(360deg)}}
.stSpinner > div > div{border-top-color:var(--acc,#7c6aff) !important;}
</style>
<div class="iq-bg"><div class="iq-glow1"></div><div class="iq-glow2"></div></div>
<div class="iq-topbar"></div>
""", unsafe_allow_html=True)


# ============================================================================
# ============================================================================
# 🎲  DEMO DATASETS
# ============================================================================
DEMO_DATASETS = {
    "hr": {
        "icon": "👥",
        "name": "HR Analytics",
        "desc": "200 employees · salary, dept, rating, attrition",
        "domain": "HR",
        "rows": 200,
        "tag": "People & Workforce",
    },
    "ecom": {
        "icon": "🛒",
        "name": "E-commerce Sales",
        "desc": "200 orders · revenue, category, region, returns",
        "domain": "Marketing",
        "rows": 200,
        "tag": "Sales & Revenue",
    },
    "finance": {
        "icon": "📈",
        "name": "Finance Report",
        "desc": "96 months · revenue, costs, profit, EBITDA",
        "domain": "Finance",
        "rows": 96,
        "tag": "Financial Metrics",
    },
}

def _load_demo(key: str):
    """Generate and store a built-in demo dataset then navigate to Profile."""
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    if key == "hr":
        depts  = ["Sales", "Marketing", "IT", "HR", "Finance", "Operations"]
        levels = ["Junior", "Mid", "Senior", "Lead", "Manager"]
        n = 200
        df = pd.DataFrame({
            "EmployeeID":        range(1001, 1001 + n),
            "Department":        rng.choices(depts, k=n),
            "Level":             rng.choices(levels, k=n),
            "Age":               np_rng.integers(22, 62, n),
            "Salary":            np_rng.integers(38000, 160000, n),
            "YearsAtCompany":    np_rng.integers(1, 22, n),
            "PerformanceRating": np_rng.choice([1, 2, 3, 4, 5], n,
                                     p=[0.05, 0.10, 0.30, 0.35, 0.20]),
            "Overtime":          rng.choices(["Yes", "No"], weights=[35, 65], k=n),
            "Attrition":         rng.choices(["Yes", "No"], weights=[20, 80], k=n),
        })
        file_name = "demo_hr_analytics.csv"

    elif key == "ecom":
        cats     = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports", "Toys"]
        regions  = ["North", "South", "East", "West", "Central"]
        statuses = ["Delivered", "Shipped", "Returned", "Cancelled"]
        n = 200
        revenue = np_rng.integers(25, 2500, n)
        df = pd.DataFrame({
            "OrderID":        [f"ORD-{i:05d}" for i in range(1, n + 1)],
            "Date":           pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "Category":       rng.choices(cats, k=n),
            "Region":         rng.choices(regions, k=n),
            "Revenue":        revenue,
            "Cost":           (revenue * np_rng.uniform(0.35, 0.65, n)).astype(int),
            "Quantity":       np_rng.integers(1, 20, n),
            "Status":         rng.choices(statuses, weights=[60, 20, 12, 8], k=n),
            "CustomerRating": np_rng.choice([1, 2, 3, 4, 5], n,
                                  p=[0.05, 0.08, 0.22, 0.40, 0.25]),
        })
        file_name = "demo_ecommerce_sales.csv"

    else:  # finance
        n = 96
        months  = pd.date_range("2016-01-01", periods=n, freq="MS").strftime("%Y-%m")
        base    = np_rng.integers(180000, 220000, n)
        trend   = (np.arange(n) * 800).astype(int)
        revenue = base + trend + np_rng.integers(-15000, 15000, n)
        opex    = np_rng.integers(30000, 60000, n)
        cogs    = (revenue * np_rng.uniform(0.52, 0.62, n)).astype(int)
        df = pd.DataFrame({
            "Month":             months,
            "Year":              [(2016 + i // 12) for i in range(n)],
            "Quarter":           [f"Q{((i % 12) // 3) + 1}" for i in range(n)],
            "Revenue":           revenue,
            "CostOfGoods":       cogs,
            "GrossProfit":       revenue - cogs,
            "OperatingExpenses": opex,
            "EBITDA":            revenue - cogs - opex,
            "Headcount":         np_rng.integers(45, 95, n),
        })
        file_name = "demo_finance_report.csv"

    processor = DataProcessor()
    processor.df = df
    processor.file_path = file_name
    st.session_state.df              = df
    st.session_state.processor       = processor
    st.session_state.df_cleaned      = None
    st.session_state.profile         = None
    st.session_state.analysis_results = None
    st.session_state.domain          = DEMO_DATASETS[key]["domain"]
    st.session_state.onboarded       = True
    st.session_state.nav_page        = "profile"
    st.rerun()


def show_success(message):
    st.markdown(f"""
    <div class="toast-success">
        <div class="toast-icon">✓</div>
        <span class="toast-text">{message}</span>
    </div>
    """, unsafe_allow_html=True)


def show_warning_upload():
    st.markdown("""
    <div class="alert-box">
        <div class="alert-icon">📂</div>
        <h3 class="alert-title">No Data Loaded</h3>
        <p class="alert-desc">Upload a dataset to unlock this feature</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("↗ Go to Upload", use_container_width=True):
        st.session_state.nav_page = "upload"
        st.rerun()


def show_error(title, message, details=None):
    st.markdown(f"""
    <div class="error-card">
        <h3 class="error-title">⚠ {title}</h3>
        <p class="error-text">{message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if details:
        with st.expander("Details"):
            st.code(details)


def get_chart_layout():
    """Return consistent chart layout."""
    return dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        font_family='Onest',
        title_font_size=18,
        title_font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
        margin=dict(l=40, r=40, t=60, b=40)
    )


def generate_pdf_content(report_text):
    """Generate PDF-like content (simplified as text for download)."""
    content = f"""
================================================================================
                         AI DATA ANALYST PRO - REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
--------------------------------------------------------------------------------

{report_text}

================================================================================
                              END OF REPORT
================================================================================
    """
    return content


def generate_json_report(results, df_info=None):
    """Generate JSON report from analysis results."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "tool": "AI Data Analyst Pro v7",
        "analysis": results.get("analysis", ""),
        "report": results.get("report", ""),
        "visualization_code": results.get("visualization", ""),
    }
    
    if df_info:
        report["data_info"] = df_info
    
    return json.dumps(report, indent=2, default=str)


# ============================================================================
# SESSION STATE
# ============================================================================
def init_session():
    defaults = {
        'df': None,
        'df_cleaned': None,
        'processor': None,
        'profile': None,
        'analysis_results': None,
        'charts': [],
        'page': 'home',
        'nav_page': None,
        'domain': 'General',
        'memory_log': [],
        'saved_charts': [],
        'theme': 'dark',
        'onboarded': True,
        '_show_demo': False,
        'ql_sql_result': '',
        'ql_dax_result': '',
        'ql_auto_sql': '',
        'ql_auto_dax': '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    with st.sidebar:
        # ── Brand ───────────────────────────────────────────────
        st.markdown("""
        <div style="padding:1.4rem 1.1rem 1rem;border-bottom:1px solid rgba(255,255,255,.06);">
          <div style="display:flex;align-items:center;gap:9px;">
            <div style="width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,#7c6aff,#38bdf8);
                        display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;">⬡</div>
            <div>
              <div style="font-size:.92rem;font-weight:800;color:#f1f3ff;letter-spacing:-.3px;line-height:1.2;">analyst-LLM</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:.44rem;color:#545b78;
                          letter-spacing:2px;text-transform:uppercase;">5-agent · ai</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Workflow progress ───────────────────────────────────
        st.markdown('<span class="nav-section-label">Progress</span>', unsafe_allow_html=True)
        _has_data    = st.session_state.df is not None
        _has_profile = st.session_state.profile is not None
        _has_results = st.session_state.analysis_results is not None
        _wf_items = [
            ("01", "Upload",    _has_data),
            ("02", "Profile",   _has_data and _has_profile),
            ("03", "Analyse",   _has_data),
            ("04", "Visualise", _has_data),
            ("05", "AI Engine", _has_results),
            ("06", "Query Lab", False),
        ]
        _wf_html = '<div class="wf-steps">'
        for _n, _l, _d in _wf_items:
            _cls = "done" if _d else ""
            _wf_html += (f'<div class="wf-step {_cls}">'
                         f'<span class="wf-step-num">{_n}</span>'
                         f'<span class="wf-step-dot"></span>'
                         f'<span class="wf-step-label">{_l}</span>'
                         f'</div>')
        _wf_html += '</div>'
        st.markdown(_wf_html, unsafe_allow_html=True)

        # ── Navigation ─────────────────────────────────────────
        st.markdown('<span class="nav-section-label" style="margin-top:.5rem;">Navigate</span>',
                    unsafe_allow_html=True)

        pages = {
            "⬡  Overview":    "home",
            "↑  Upload":      "upload",
            "◎  Profile":     "profile",
            "◫  Analytics":   "analysis",
            "△  Visualize":   "viz",
            "◇  AI Engine":   "ai",
            "◈  Query Lab":   "query_lab",
        }

        if st.session_state.nav_page:
            target = st.session_state.nav_page
            st.session_state.nav_page = None
            st.session_state.page = target   # keep page state in sync
            for label, val in pages.items():
                if val == target:
                    return val

        # Restore selected radio to current page
        current_page_key = st.session_state.get("page", "home")
        current_label = next((lbl for lbl, val in pages.items() if val == current_page_key), list(pages.keys())[0])
        current_idx = list(pages.keys()).index(current_label) if current_label in pages else 0

        selected_label = st.radio("Navigate", list(pages.keys()),
                                  index=current_idx,
                                  label_visibility="collapsed")
        page = pages.get(selected_label, "home")
        st.session_state.page = page

        # ── Status ─────────────────────────────────────────────
        st.markdown('<div style="height:1px;background:rgba(255,255,255,.06);margin:6px 10px;"></div>',
                    unsafe_allow_html=True)
        st.markdown('<span class="nav-section-label">Status</span>', unsafe_allow_html=True)

        groq_key = os.getenv("GROQ_API_KEY", "")
        _ai_dot  = "m-dot" if groq_key else "m-dot off"
        _ai_txt  = "AI Engine ready" if groq_key else "Add GROQ_API_KEY"
        st.markdown(f'<div class="m-status"><div class="{_ai_dot}"></div><span>{_ai_txt}</span></div>',
                    unsafe_allow_html=True)

        if st.session_state.df is not None:
            r, c = st.session_state.df.shape
            st.markdown(
                f'<div class="m-status" style="margin-top:4px;">'
                f'<div class="m-dot"></div><span>{r:,} rows · {c} cols</span></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="m-status" style="margin-top:4px;">'
                '<div class="m-dot off"></div><span>No dataset loaded</span></div>',
                unsafe_allow_html=True)

        return page


# ============================================================================
# 🏠 HOME PAGE
# ============================================================================
def page_home():
    st.markdown("""
    <div class="home-hero">
      <span class="home-eyebrow">multi-agent ai platform</span>
      <h1 class="home-title">Turn raw data into<br><span class="hi">real insights</span></h1>
      <p class="home-desc">
        Five specialised AI agents work in sequence — cleaning, analysing, visualising,
        writing a report, and auditing the findings. No code. No configuration.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if st.button("↑  Upload Your Data", use_container_width=True, key="home_cta_top"):
            st.session_state.nav_page = "upload"
            st.rerun()
    with col_b:
        if st.button("✦  Try HR Demo", use_container_width=True, key="home_demo_top"):
            _load_demo("hr")
    with col_c:
        st.markdown('<p class="cta-note" style="margin-top:.45rem;">No API key? Profiling &amp; charts still work.</p>',
                    unsafe_allow_html=True)

    # ── How it works ────────────────────────────────────────────
    st.markdown('<div class="sec-rule"><span class="sec-rule-label">How it works</span></div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="step-cards">
      <div class="step-card">
        <span class="step-num-big">01</span>
        <span class="step-icon-lg">📂</span>
        <h4 class="step-card-title">Upload</h4>
        <p class="step-card-desc">Drop any CSV or Excel file. The pipeline ingests and validates it instantly.</p>
      </div>
      <div class="step-card">
        <span class="step-num-big">02</span>
        <span class="step-icon-lg">🎯</span>
        <h4 class="step-card-title">Set Domain</h4>
        <p class="step-card-desc">Choose Finance, HR, Marketing, Healthcare or General for domain-tuned prompts.</p>
      </div>
      <div class="step-card">
        <span class="step-num-big">03</span>
        <span class="step-icon-lg">⚡</span>
        <h4 class="step-card-title">Run Agents</h4>
        <p class="step-card-desc">Five AI agents run in sequence: Engineer → Analyst → Visualizer → Reporter → QA.</p>
      </div>
      <div class="step-card">
        <span class="step-num-big">04</span>
        <span class="step-icon-lg">✨</span>
        <h4 class="step-card-title">Explore & Export</h4>
        <p class="step-card-desc">Read insights, view charts, audit the quality gate, and download full reports.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quote ────────────────────────────────────────────────────
    st.markdown("""
    <div class="data-quote">
      <blockquote>"The goal is to turn data into information, and information into insight."</blockquote>
      <cite>— Carly Fiorina, Former CEO of HP</cite>
    </div>
    """, unsafe_allow_html=True)

    # ── Agent pipeline ───────────────────────────────────────────
    st.markdown('<div class="sec-rule"><span class="sec-rule-label">Agent pipeline</span></div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-row">
      <div class="pipeline-step">
        <span class="pip-num">AGENT 01</span><span class="pip-icon">⬡</span>
        <p class="pip-name">Engineer</p><span class="pip-tag">clean · validate</span>
      </div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step">
        <span class="pip-num">AGENT 02</span><span class="pip-icon">◎</span>
        <p class="pip-name">Analyst</p><span class="pip-tag">stats · patterns</span>
      </div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step">
        <span class="pip-num">AGENT 03</span><span class="pip-icon">△</span>
        <p class="pip-name">Visualizer</p><span class="pip-tag">charts · code</span>
      </div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step">
        <span class="pip-num">AGENT 04</span><span class="pip-icon">◫</span>
        <p class="pip-name">Reporter</p><span class="pip-tag">narrative · summary</span>
      </div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step">
        <span class="pip-num">AGENT 05</span><span class="pip-icon">🛡</span>
        <p class="pip-name">Quality Gate</p><span class="pip-tag">audit · verify</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    



def _page_flash():
    """Inject a CSS page-transition flash overlay + slide-in on every page render."""
    st.markdown("""
    <div class="page-flash"></div>
    <style>.block-container{animation:slideInRight .28s cubic-bezier(.25,.46,.45,.94) both;}</style>
    """, unsafe_allow_html=True)


def _dot_spinner(label: str = "Processing"):
    """Return HTML string for an inline dot-bounce spinner with label."""
    return f"""<div class="iq-processing">
      <span class="spin-ring"></span>
      <span><strong>{label}</strong>&nbsp;&nbsp;<span class="iq-dots"><span></span><span></span><span></span></span></span>
    </div>"""


def page_upload():
    _page_flash()
    st.markdown("""
    <div class="m-hero"><span class="m-hero-eyebrow">data ingestion</span><h1 class="m-hero-title">Upload <span class="hi">Your Data</span></h1><p class="m-hero-desc">Drop a CSV or Excel file — the pipeline ingests and profiles it automatically.</p></div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader("Upload", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")
        
        if uploaded:
            try:
                processor = DataProcessor()
                df = processor.load_from_uploaded_file(uploaded)
                st.session_state.df = df
                st.session_state.processor = processor
                
                show_success(f"🎉 Loaded {df.shape[0]:,} rows × {df.shape[1]} columns successfully!")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("🧹 Clean Data", use_container_width=True):
                        with st.spinner("Cleaning..."):
                            df_clean = processor.clean_data()
                            profile = processor.generate_profile_text()
                            st.session_state.df_cleaned = df_clean
                            st.session_state.profile = profile
                        show_success("✨ Data cleaned successfully!")
                
                with c2:
                    if st.button("📊 Quick Stats", use_container_width=True):
                        st.dataframe(df.describe(), use_container_width=True)
                
                with c3:
                    if st.button("📋 Column Info", use_container_width=True):
                        info = pd.DataFrame({
                            'Column': df.columns,
                            'Type': df.dtypes.astype(str),
                            'Non-Null': df.count(),
                            'Unique': [df[c].nunique() for c in df.columns]
                        })
                        st.dataframe(info, use_container_width=True)
                        
            except Exception as e:
                show_error("Upload Failed", str(e), traceback.format_exc())
    
    with col2:
        # ── Supported formats ───────────────────────────────────
        st.markdown("""
        <div class="glass-panel" style="margin-bottom:1.2rem;">
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.56rem;font-weight:500;
               color:var(--t3);text-transform:uppercase;letter-spacing:2.5px;margin:0 0 0.75rem;">Supported formats</p>
            <p style="color:var(--t2);font-size:0.85rem;margin:0.3rem 0;">📄 CSV &nbsp;(.csv)</p>
            <p style="color:var(--t2);font-size:0.85rem;margin:0.3rem 0;">📊 Excel &nbsp;(.xlsx, .xls)</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Demo datasets ───────────────────────────────────────
        st.markdown("""
        <p style="font-family:'JetBrains Mono',monospace;font-size:0.56rem;font-weight:500;
           color:var(--t3);text-transform:uppercase;letter-spacing:2.5px;margin:0 0 0.75rem;display:block;">
           ✦ Try a demo dataset
        </p>""", unsafe_allow_html=True)

        for key, meta in DEMO_DATASETS.items():
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--b2);border-radius:10px;
                        padding:0.95rem 1.1rem;margin-bottom:8px;
                        transition:border-color 0.18s;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.3rem;">
                    <span style="font-size:1.25rem;">{meta['icon']}</span>
                    <span style="font-size:0.88rem;font-weight:700;color:var(--t1);letter-spacing:-0.2px;">{meta['name']}</span>
                    <span style="margin-left:auto;font-family:'JetBrains Mono',monospace;
                                 font-size:0.52rem;color:var(--acc);background:var(--acc-dim);
                                 border:1px solid rgba(124,106,255,.2);border-radius:99px;
                                 padding:2px 8px;">{meta['domain']}</span>
                </div>
                <p style="font-size:0.76rem;color:var(--t2);margin:0 0 0.55rem;line-height:1.5;">{meta['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Load {meta['name']}", key=f"demo_{key}", use_container_width=True):
                _load_demo(key)

    # ── Page navigation ──────────────────────────────────────────
    if st.session_state.df is not None:
        st.markdown('<div class="page-nav-rule"></div>', unsafe_allow_html=True)
        _, col_next = st.columns([3, 1])
        with col_next:
            if st.button("Profile →", use_container_width=True, key="upload_next"):
                st.session_state.nav_page = "profile"
                st.rerun()


# ============================================================================
# 🔍 PROFILE PAGE - FIXED
# ============================================================================
def page_profile():
    _page_flash()
    if st.session_state.df is None:
        show_warning_upload()
        return
    
    st.markdown("""
    <div class="m-hero"><span class="m-hero-eyebrow">data profiler</span><h1 class="m-hero-title">Dataset <span class="hi">Profile</span></h1><p class="m-hero-desc">Structure, statistics, nulls, category distributions — everything in one view.</p></div>
    """, unsafe_allow_html=True)
    
    processor = st.session_state.processor
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    
    # Auto clean if needed
    if st.session_state.profile is None and processor:
        try:
            with st.spinner("Analyzing data..."):
                if st.session_state.df_cleaned is None:
                    processor.clean_data()
                    st.session_state.df_cleaned = processor.df_cleaned
                    df = processor.df_cleaned if processor.df_cleaned is not None else st.session_state.df
                st.session_state.profile = processor.generate_profile_text()
        except Exception as e:
            st.warning(f"Could not auto-analyze: {e}")
            df = st.session_state.df
    
    # Stats Cards
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    missing_count = int(df.isnull().sum().sum())
    
    cols = st.columns(5)
    stats = [
        ("≡", df.shape[0], "Rows"),
        ("◫", df.shape[1], "Columns"),
        ("#", len(num_cols), "Numeric"),
        ("A", len(cat_cols), "Text"),
        ("?", missing_count, "Missing")
    ]
    
    for col, (icon, val, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <p class="metric-value">{val:,}</p>
                <p class="metric-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tabs = st.tabs(["📋 Data Preview", "📊 Column Info", "❓ Missing Values", "📈 Statistics", "🔤 Categorical"])
    
    with tabs[0]:
        st.markdown('<div class="section-header"><h3>📋 Data Preview (First 50 rows)</h3></div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=400)
    
    with tabs[1]:
        st.markdown('<div class="section-header"><h3>📊 Column Information</h3></div>', unsafe_allow_html=True)
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Null %': (df.isnull().sum() / len(df) * 100).round(2).values,
            'Unique': [df[c].nunique() for c in df.columns],
            'Sample Value': [str(df[c].dropna().iloc[0]) if len(df[c].dropna()) > 0 else 'N/A' for c in df.columns]
        })
        st.dataframe(info_df, use_container_width=True, height=400)
    
    with tabs[2]:
        st.markdown('<div class="section-header"><h3>❓ Missing Values Analysis</h3></div>', unsafe_allow_html=True)
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        if missing.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': missing_pct.values
            }).sort_values('Missing Count', ascending=False)
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(missing_df, use_container_width=True)
            with c2:
                fig = px.bar(
                    missing_df, 
                    x='Column', 
                    y='Missing %',
                    color='Missing %',
                    color_continuous_scale='Reds',
                    title="Missing Values by Column"
                )
                fig.update_layout(**get_chart_layout(), height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            show_success("✨ No missing values found! Your data is complete.")
    
    with tabs[3]:
        st.markdown('<div class="section-header"><h3>📈 Numerical Statistics</h3></div>', unsafe_allow_html=True)
        
        if num_cols:
            stats_df = df[num_cols].describe().T
            stats_df['range'] = stats_df['max'] - stats_df['min']
            stats_df['cv'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)
            stats_df = stats_df.round(2)
            st.dataframe(stats_df, use_container_width=True)
            
            # Distribution plot
            st.markdown('<div class="section-header"><h3>📊 Distribution Overview</h3></div>', unsafe_allow_html=True)
            selected_col = st.selectbox("Select column to visualize:", num_cols, key="profile_num_col")
            
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=selected_col, nbins=30, color_discrete_sequence=['#7c3aed'],
                                  title=f"Distribution of {selected_col}")
                fig.update_layout(**get_chart_layout(), height=350)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(df, y=selected_col, color_discrete_sequence=['#06b6d4'],
                            title=f"Box Plot of {selected_col}")
                fig.update_layout(**get_chart_layout(), height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numerical columns found in the dataset.")
    
    with tabs[4]:
        st.markdown('<div class="section-header"><h3>🔤 Categorical Analysis</h3></div>', unsafe_allow_html=True)
        
        if cat_cols:
            selected_cat = st.selectbox("Select categorical column:", cat_cols, key="profile_cat_col")
            
            value_counts = df[selected_cat].value_counts().head(15)
            
            c1, c2 = st.columns([1, 1])
            with c1:
                vc_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(df) * 100).round(2)
                })
                st.dataframe(vc_df, use_container_width=True)
            with c2:
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {selected_cat}",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(**get_chart_layout(), height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found in the dataset.")
    
    # Profile Text
    with st.expander("📜 Full Profile Text (for AI)"):
        if st.session_state.profile:
            st.code(st.session_state.profile)
        else:
            st.info("Click 'Clean Data' on Upload page to generate profile.")

    # ── Page navigation ──────────────────────────────────────────
    st.markdown('<div class="page-nav-rule"></div>', unsafe_allow_html=True)
    col_prev, _, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Upload", use_container_width=True, key="profile_prev"):
            st.session_state.nav_page = "upload"
            st.rerun()
    with col_next:
        if st.button("Analytics →", use_container_width=True, key="profile_next"):
            st.session_state.nav_page = "analysis"
            st.rerun()


# ============================================================================
# 📊 ANALYTICS PAGE - FIXED
# ============================================================================
def page_analysis():
    _page_flash()
    if st.session_state.df is None:
        show_warning_upload()
        return
    
    st.markdown("""
    <div class="pg-hdr">
        <span class="pg-hdr-eyebrow">analytics hub</span>
        <h1 class="pg-hdr-title"><span class="hi">Analytics</span> Hub</h1>
        <p class="pg-hdr-desc">Discover patterns, correlations, and statistical insights in your data</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    processor = st.session_state.processor
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Analytics tabs
    tabs = st.tabs(["🔗 Correlation", "📊 Segmentation", "📈 Distribution", "📋 Summary Stats"])
    
    with tabs[0]:
        st.markdown('<div class="section-header"><h3>🔗 Correlation Matrix</h3></div>', unsafe_allow_html=True)
        
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            
            fig = px.imshow(
                corr, 
                text_auto='.2f', 
                color_continuous_scale='RdBu_r',
                aspect='auto',
                title="Correlation Heatmap"
            )
            fig.update_layout(**get_chart_layout(), height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Find strong correlations
            st.markdown('<div class="section-header"><h3>🎯 Strong Correlations</h3></div>', unsafe_allow_html=True)
            
            strong_corrs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.5:
                        strong_corrs.append({
                            'Variable 1': corr.columns[i],
                            'Variable 2': corr.columns[j],
                            'Correlation': round(val, 3),
                            'Strength': 'Strong' if abs(val) > 0.7 else 'Moderate'
                        })
            
            if strong_corrs:
                st.dataframe(pd.DataFrame(strong_corrs), use_container_width=True)
            else:
                st.info("No strong correlations (|r| > 0.5) found between numeric variables.")
        else:
            st.warning("⚠️ Need at least 2 numeric columns for correlation analysis.")
    
    with tabs[1]:
        st.markdown('<div class="section-header"><h3>📊 Segmentation Analysis</h3></div>', unsafe_allow_html=True)
        
        if cat_cols and num_cols:
            c1, c2 = st.columns(2)
            with c1:
                grp_col = st.selectbox("🏷️ Group by (categorical):", cat_cols, key="seg_grp")
            with c2:
                val_col = st.selectbox("🔢 Measure (numeric):", num_cols, key="seg_val")
            
            if grp_col and val_col:
                grouped = df.groupby(grp_col)[val_col].agg(['mean', 'median', 'std', 'min', 'max', 'count']).round(2)
                grouped = grouped.sort_values('mean', ascending=False)
                grouped.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Count']
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.dataframe(grouped, use_container_width=True)
                with c2:
                    fig = px.bar(
                        grouped.reset_index(), 
                        x=grp_col, 
                        y='Mean',
                        color='Mean',
                        color_continuous_scale='Viridis',
                        title=f"Average {val_col} by {grp_col}"
                    )
                    fig.update_layout(**get_chart_layout(), height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Box plot comparison
                st.markdown("<br>", unsafe_allow_html=True)
                fig = px.box(
                    df, 
                    x=grp_col, 
                    y=val_col,
                    color=grp_col,
                    title=f"Distribution of {val_col} by {grp_col}"
                )
                fig.update_layout(**get_chart_layout(), height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Need both categorical and numeric columns for segmentation analysis.")
    
    with tabs[2]:
        st.markdown('<div class="section-header"><h3>📈 Distribution Analysis</h3></div>', unsafe_allow_html=True)
        
        if num_cols:
            dist_col = st.selectbox("📊 Select column:", num_cols, key="dist_col")
            
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(
                    df, 
                    x=dist_col, 
                    nbins=30, 
                    color_discrete_sequence=['#7c3aed'],
                    title=f"Histogram of {dist_col}"
                )
                fig.update_layout(**get_chart_layout(), height=380)
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                fig = px.box(
                    df, 
                    y=dist_col, 
                    color_discrete_sequence=['#06b6d4'],
                    title=f"Box Plot of {dist_col}"
                )
                fig.update_layout(**get_chart_layout(), height=380)
                st.plotly_chart(fig, use_container_width=True)
            
            # Stats summary
            col_data = df[dist_col].dropna()
            stats_info = {
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'Std Dev': col_data.std(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Range': col_data.max() - col_data.min(),
                'Skewness': col_data.skew(),
                'Kurtosis': col_data.kurtosis()
            }
            
            st.markdown('<div class="section-header"><h3>📋 Distribution Stats</h3></div>', unsafe_allow_html=True)
            stats_cols = st.columns(4)
            for i, (k, v) in enumerate(stats_info.items()):
                with stats_cols[i % 4]:
                    st.metric(k, f"{v:.2f}")
        else:
            st.warning("⚠️ No numeric columns found for distribution analysis.")
    
    with tabs[3]:
        st.markdown('<div class="section-header"><h3>📋 Summary Statistics</h3></div>', unsafe_allow_html=True)
        
        st.markdown("**Numerical Columns:**")
        if num_cols:
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
        else:
            st.info("No numerical columns.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Categorical Columns:**")
        if cat_cols:
            cat_summary = []
            for col in cat_cols:
                try:
                    mode_vals = df[col].mode()
                    most_common = mode_vals.iloc[0] if len(mode_vals) > 0 else 'N/A'
                    
                    val_counts = df[col].value_counts()
                    most_common_count = val_counts.iloc[0] if len(val_counts) > 0 else 0
                except Exception:
                    most_common = 'N/A'
                    most_common_count = 0
                
                cat_summary.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Common': most_common,
                    'Most Common Count': most_common_count,
                    'Missing': df[col].isnull().sum()
                })
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)
        else:
            st.info("No categorical columns.")

    # ── Page navigation ──────────────────────────────────────────
    st.markdown('<div class="page-nav-rule"></div>', unsafe_allow_html=True)
    col_prev, _, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Profile", use_container_width=True, key="analysis_prev"):
            st.session_state.nav_page = "profile"
            st.rerun()
    with col_next:
        if st.button("Visualize →", use_container_width=True, key="analysis_next"):
            st.session_state.nav_page = "viz"
            st.rerun()


# ============================================================================
# 📈 VISUALIZATION PAGE - ULTRA MODERN REDESIGN
# ============================================================================
def page_viz():
    _page_flash()
    if st.session_state.df is None:
        show_warning_upload()
        return
    
    st.markdown("""
    <div class="pg-hdr">
        <span class="pg-hdr-eyebrow">visualization studio</span>
        <h1 class="pg-hdr-title"><span class="hi">Visualization</span> Studio</h1>
        <p class="pg-hdr-desc">Create stunning interactive charts with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Modern Tab Interface
    viz_tabs = st.tabs(["🎨 Chart Builder", "✨ Auto-Generate", "📊 Gallery"])
    
    # TAB 1: Chart Builder
    with viz_tabs[0]:
        col_ctrl, col_chart = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown('<div class="section-header"><h3>🎨 Chart Settings</h3></div>', unsafe_allow_html=True)
            
            chart_type = st.selectbox(
                "📊 Chart Type",
                ["📊 Histogram", "📈 Scatter Plot", "📉 Line Chart", "📊 Bar Chart", 
                 "🥧 Pie Chart", "📦 Box Plot", "🔥 Heatmap", "🎻 Violin Plot"],
                key="viz_chart_type"
            )
            
            st.markdown("---")
            
            # Dynamic controls based on chart type
            if "Histogram" in chart_type:
                if num_cols:
                    x_col = st.selectbox("📍 Column", num_cols, key="hist_col")
                    bins = st.slider("📏 Bins", 10, 100, 30, key="hist_bins")
                    color_scheme = st.selectbox("🎨 Color", ["Purple", "Cyan", "Pink", "Green", "Orange"], key="hist_color")
                else:
                    st.warning("⚠️ No numeric columns")
                    x_col, bins, color_scheme = None, 30, "Purple"
            
            elif "Scatter" in chart_type:
                if len(num_cols) >= 2:
                    x_col = st.selectbox("📍 X Axis", num_cols, key="scatter_x")
                    y_col = st.selectbox("📍 Y Axis", num_cols, index=min(1, len(num_cols)-1), key="scatter_y")
                    color_by = st.selectbox("🎨 Color by", ["None"] + cat_cols, key="scatter_color")
                    show_trend = st.checkbox("📈 Show Trendline", True, key="scatter_trend")
                    size_by = st.selectbox("📏 Size by", ["None"] + num_cols, key="scatter_size")
                else:
                    st.warning("⚠️ Need 2+ numeric columns")
                    x_col, y_col, color_by, show_trend, size_by = None, None, "None", False, "None"
            
            elif "Line" in chart_type:
                if num_cols:
                    x_col = st.selectbox("📍 X Axis", all_cols, key="line_x")
                    y_col = st.selectbox("📍 Y Axis", num_cols, key="line_y")
                    smooth = st.checkbox("✨ Smooth Line", False, key="line_smooth")
                else:
                    st.warning("⚠️ No numeric columns")
                    x_col, y_col, smooth = None, None, False
            
            elif "Bar" in chart_type:
                if cat_cols:
                    x_col = st.selectbox("📍 Category", cat_cols, key="bar_col")
                    top_n = st.slider("🔝 Show Top N", 5, 25, 10, key="bar_top")
                    if num_cols:
                        agg_col = st.selectbox("📊 Aggregate", ["Count"] + num_cols, key="bar_agg")
                    else:
                        agg_col = "Count"
                    horizontal = st.checkbox("↔️ Horizontal", False, key="bar_horiz")
                else:
                    st.warning("⚠️ No categorical columns")
                    x_col, top_n, agg_col, horizontal = None, 10, "Count", False
            
            elif "Pie" in chart_type:
                if cat_cols:
                    x_col = st.selectbox("📍 Category", cat_cols, key="pie_col")
                    top_n = st.slider("🔝 Show Top N", 3, 15, 6, key="pie_top")
                    donut = st.checkbox("🍩 Donut Style", False, key="pie_donut")
                else:
                    st.warning("⚠️ No categorical columns")
                    x_col, top_n, donut = None, 6, False
            
            elif "Box" in chart_type:
                if num_cols:
                    y_col = st.selectbox("📍 Value", num_cols, key="box_val")
                    group_by = st.selectbox("🏷️ Group by", ["None"] + cat_cols, key="box_grp")
                    show_points = st.checkbox("• Show Points", False, key="box_points")
                else:
                    st.warning("⚠️ No numeric columns")
                    y_col, group_by, show_points = None, "None", False
            
            elif "Heatmap" in chart_type:
                if len(num_cols) >= 2:
                    selected_cols = st.multiselect("📍 Select Columns", num_cols, default=num_cols[:min(8, len(num_cols))], key="heat_cols")
                    color_scale = st.selectbox("🎨 Color Scale", ["RdBu_r", "Viridis", "Plasma", "Cividis", "Turbo"], key="heat_color")
                else:
                    st.warning("⚠️ Need 2+ numeric columns")
                    selected_cols, color_scale = [], "RdBu_r"
            
            elif "Violin" in chart_type:
                if num_cols:
                    y_col = st.selectbox("📍 Value", num_cols, key="violin_val")
                    group_by = st.selectbox("🏷️ Group by", ["None"] + cat_cols, key="violin_grp")
                    show_box = st.checkbox("📦 Show Box Inside", True, key="violin_box")
                else:
                    st.warning("⚠️ No numeric columns")
                    y_col, group_by, show_box = None, "None", True
    
        
        with col_chart:
            st.markdown('<div class="section-header"><h3>🖼️ Chart Preview</h3></div>', unsafe_allow_html=True)
            
            layout = get_chart_layout()
            layout['height'] = 550
            
            color_map = {
                "Purple": "#7c3aed", "Cyan": "#06b6d4", "Pink": "#ec4899", 
                "Green": "#10b981", "Orange": "#f59e0b"
            }
            
            try:
                if "Histogram" in chart_type and num_cols and x_col:
                    fig = px.histogram(
                        df, x=x_col, nbins=bins,
                        color_discrete_sequence=[color_map.get(color_scheme, "#7c3aed")],
                        title=f"📊 Distribution of {x_col}"
                    )
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Quick stats
                    col_data = df[x_col].dropna()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mean", f"{col_data.mean():.2f}")
                    c2.metric("Median", f"{col_data.median():.2f}")
                    c3.metric("Std Dev", f"{col_data.std():.2f}")
                    c4.metric("Range", f"{col_data.max() - col_data.min():.2f}")
                
                elif "Scatter" in chart_type and len(num_cols) >= 2 and x_col and y_col:
                    fig = px.scatter(
                        df, x=x_col, y=y_col,
                        color=None if color_by == "None" else color_by,
                        size=None if size_by == "None" else size_by,
                        trendline="ols" if show_trend else None,
                        title=f"📈 {y_col} vs {x_col}",
                        opacity=0.7
                    )
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation coefficient
                    corr = df[[x_col, y_col]].corr().iloc[0, 1]
                    c1, c2 = st.columns(2)
                    c1.metric("Correlation", f"{corr:.3f}")
                    c2.metric("R² (if linear)", f"{corr**2:.3f}")
                
                elif "Line" in chart_type and num_cols and x_col and y_col:
                    df_sorted = df.sort_values(x_col)
                    fig = px.line(
                        df_sorted, x=x_col, y=y_col,
                        color_discrete_sequence=['#7c3aed'],
                        title=f"📉 {y_col} over {x_col}"
                    )
                    if smooth:
                        fig.update_traces(line_shape='spline')
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend info
                    c1, c2 = st.columns(2)
                    c1.metric("Min", f"{df_sorted[y_col].min():.2f}")
                    c2.metric("Max", f"{df_sorted[y_col].max():.2f}")
                
                elif "Bar" in chart_type and cat_cols and x_col:
                    if agg_col == "Count":
                        vc = df[x_col].value_counts().head(top_n)
                        if horizontal:
                            fig = px.bar(
                                y=vc.index, x=vc.values,
                                color=vc.values, color_continuous_scale='Viridis',
                                title=f"📊 Count of {x_col}",
                                orientation='h'
                            )
                        else:
                            fig = px.bar(
                                x=vc.index, y=vc.values,
                                color=vc.values, color_continuous_scale='Viridis',
                                title=f"📊 Count of {x_col}"
                            )
                    else:
                        agg_df = df.groupby(x_col)[agg_col].mean().sort_values(ascending=False).head(top_n)
                        if horizontal:
                            fig = px.bar(
                                y=agg_df.index, x=agg_df.values,
                                color=agg_df.values, color_continuous_scale='Viridis',
                                title=f"📊 Average {agg_col} by {x_col}",
                                orientation='h'
                            )
                        else:
                            fig = px.bar(
                                x=agg_df.index, y=agg_df.values,
                                color=agg_df.values, color_continuous_scale='Viridis',
                                title=f"📊 Average {agg_col} by {x_col}"
                            )
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif "Pie" in chart_type and cat_cols and x_col:
                    vc = df[x_col].value_counts().head(top_n)
                    fig = px.pie(
                        values=vc.values, names=vc.index,
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        title=f"🥧 Distribution of {x_col}",
                        hole=0.4 if donut else 0
                    )
                    fig.update_layout(**layout)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                elif "Box" in chart_type and num_cols and y_col:
                    fig = px.box(
                        df, y=y_col,
                        x=None if group_by == "None" else group_by,
                        color=None if group_by == "None" else group_by,
                        points='all' if show_points else 'outliers',
                        title=f"📦 Box Plot of {y_col}"
                    )
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif "Heatmap" in chart_type and selected_cols and len(selected_cols) >= 2:
                    corr = df[selected_cols].corr()
                    fig = px.imshow(
                        corr, text_auto='.2f',
                        color_continuous_scale=color_scale,
                        title="🔥 Correlation Heatmap",
                        aspect='auto'
                    )
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Strong correlations
                    st.markdown("**🔍 Strong Correlations:**")
                    strong = []
                    for i in range(len(corr.columns)):
                        for j in range(i+1, len(corr.columns)):
                            val = corr.iloc[i, j]
                            if abs(val) > 0.6:
                                strong.append(f"{corr.columns[i]} ↔ {corr.columns[j]}: {val:.3f}")
                    if strong:
                        for s in strong[:5]:
                            st.text(s)
                    else:
                        st.info("No strong correlations found (|r| > 0.6)")
                
                elif "Violin" in chart_type and num_cols and y_col:
                    fig = px.violin(
                        df, y=y_col,
                        x=None if group_by == "None" else group_by,
                        color=None if group_by == "None" else group_by,
                        box=show_box, points="outliers",
                        title=f"🎻 Violin Plot of {y_col}"
                    )
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("👆 Configure the chart options on the left to generate a visualization.")

            except Exception as e:
                show_error("Chart Error", str(e), traceback.format_exc())

        # ── SAVE TO GALLERY BUTTON ─────────────────────────────────────────
        if st.button("💾 Save Current Chart to Gallery", use_container_width=True):
            try:
                if 'fig' in locals() and fig is not None:
                    safe_type = chart_type.split()[-1] if chart_type else "Chart"
                    entry = {
                        'fig':   fig,
                        'title': f"{safe_type} — {datetime.now().strftime('%H:%M:%S')}",
                        'type':  safe_type,
                    }
                    if 'saved_charts' not in st.session_state:
                        st.session_state.saved_charts = []
                    st.session_state.saved_charts.append(entry)
                    show_success("📸 Chart saved to Gallery!")
                else:
                    st.warning("Build a chart first, then save it.")
            except Exception as e:
                st.warning(f"Could not save chart: {e}")
    
    # TAB 2: Auto-Generate
    with viz_tabs[1]:
        st.markdown('<div class="section-header"><h3>✨ Auto-Generate All Charts</h3></div>', unsafe_allow_html=True)
        st.markdown("🤖 Let AI automatically create the best visualizations for your dataset")
        st.markdown("---")
        
        if st.button("🚀 Generate All Charts Automatically", type="primary", use_container_width=True):
            try:
                with st.spinner("🎨 Creating beautiful visualizations..."):
                    viz = DataVisualizer(df)
                    figs = viz.auto_generate_charts()
                    
                    layout = get_chart_layout()
                    layout['height'] = 400
                    
                    for i in range(0, len(figs), 2):
                        cols = st.columns(2)
                        for j, col in enumerate(cols):
                            if i + j < len(figs):
                                figs[i + j].update_layout(**layout)
                                col.plotly_chart(figs[i + j], use_container_width=True)
                    
                    show_success(f"🎉 Generated {len(figs)} charts automatically!")
            except Exception as e:
                show_error("Auto-Generate Error", str(e), traceback.format_exc())
    
    # TAB 3: Gallery
    with viz_tabs[2]:
        st.markdown('<div class="section-header"><h3>📸 Chart Gallery</h3></div>', unsafe_allow_html=True)

        saved = st.session_state.get('saved_charts', [])

        if saved:
            c_clear, c_dl = st.columns([1, 1])
            with c_clear:
                if st.button("🗑️ Clear Gallery", use_container_width=True):
                    st.session_state.saved_charts = []
                    st.rerun()
            with c_dl:
                st.info(f"📸 {len(saved)} saved chart(s)")

            for i in range(0, len(saved), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(saved):
                        entry = saved[i + j]
                        with col:
                            st.markdown(f"""
                            <div class="gal-card">
                                <div class="gal-card-header">
                                    <span class="gal-card-title">{entry['title']}</span>
                                    <span class="gal-card-type">{entry['type']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            fig = entry['fig']
                            fig.update_layout(**get_chart_layout(), height=340)
                            st.plotly_chart(fig, use_container_width=True, key=f"gal_{i}_{j}")
                            if st.button(f"🗑 Remove", key=f"rem_{i}_{j}", use_container_width=True):
                                st.session_state.saved_charts.pop(i + j)
                                st.rerun()
        else:
            st.markdown("""
            <div class="gal-empty">
                <span class="gal-empty-icon">🖼️</span>
                <p class="gal-empty-txt">Gallery is empty</p>
                <p class="gal-empty-sub">Build a chart in the Chart Builder tab, then click  <strong>💾 Save to Gallery</strong></p>
            </div>
            """, unsafe_allow_html=True)

    # ── Page navigation ──────────────────────────────────────────
    st.markdown('<div class="page-nav-rule"></div>', unsafe_allow_html=True)
    col_prev, _, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Analytics", use_container_width=True, key="viz_prev"):
            st.session_state.nav_page = "analysis"
            st.rerun()
    with col_next:
        if st.button("AI Engine →", use_container_width=True, key="viz_next"):
            st.session_state.nav_page = "ai"
            st.rerun()


# ============================================================================
# AI ENGINE PAGE — v2 (5-Agent Pipeline, Domain Selector, Memory Log, Quality Gate)
# ============================================================================
def page_ai():
    _page_flash()
    if st.session_state.df is None:
        show_warning_upload()
        return

    st.markdown("""
    <div class="pg-hdr">
        <span class="pg-hdr-eyebrow">ai engine</span>
        <h1 class="pg-hdr-title"><span class="hi">5-Agent</span> Analysis Pipeline</h1>
        <p class="pg-hdr-desc">Domain-aware · Memory-linked · Self-checking quality gate</p>
    </div>
    """, unsafe_allow_html=True)

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        st.markdown("""
        <div class="alert-box">
            <div class="alert-icon">◇</div>
            <h3 class="alert-title">AI Engine Not Configured</h3>
            <p class="alert-desc">A Groq API key is required to activate the pipeline</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-panel" style="text-align: center;">
            <h4 style="font-family: 'Syne', sans-serif; color: var(--t1); margin: 0 0 1rem;">Quick Setup</h4>
            <ol style="color: var(--t2); text-align: left; padding-left: 2rem; line-height: 2.2; font-size: 0.9rem;">
                <li>Visit <a href="https://console.groq.com" target="_blank" style="color: var(--cyan);">console.groq.com</a></li>
                <li>Create a free account &amp; generate an API key</li>
                <li>Add <code style="background: var(--bg3); padding: 2px 8px; border-radius: 6px;">GROQ_API_KEY=gsk_...</code> to your <code style="background: var(--bg3); padding: 2px 8px; border-radius: 6px;">.env</code> file</li>
                <li>Restart the app</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── DOMAIN SELECTOR ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Domain</h3></div>', unsafe_allow_html=True)

    domain_cols = st.columns([1, 4])
    with domain_cols[0]:
        st.markdown('<p style="color: var(--t2); font-size: 0.88rem; padding-top: 0.5rem;">Select domain to tune agent vocabulary &amp; metric priorities:</p>',
                    unsafe_allow_html=True)
    with domain_cols[1]:
        from agents.prompts import DOMAIN_CONTEXTS
        domain_options = list(DOMAIN_CONTEXTS.keys())
        domain_icons = {d: DOMAIN_CONTEXTS[d]["icon"] for d in domain_options}
        domain_labels = [f"{domain_icons[d]}  {d}" for d in domain_options]

        selected_label = st.radio(
            "Domain",
            domain_labels,
            index=domain_options.index(st.session_state.domain),
            horizontal=True,
            label_visibility="collapsed",
            key="domain_radio"
        )
        selected_domain = domain_options[domain_labels.index(selected_label)]
        st.session_state.domain = selected_domain

    # Show active domain context pill
    ctx = DOMAIN_CONTEXTS[selected_domain]
    st.markdown(f"""
    <div class="domain-bar">
        <span class="domain-label">Active Domain</span>
        <span class="domain-pill">{ctx['icon']}  {selected_domain}</span>
        <span style="color: var(--t3); font-size: 0.78rem;">·</span>
        <span style="color: var(--t2); font-size: 0.78rem;">{ctx['framing']}</span>
        <span style="color: var(--t3); font-size: 0.78rem; margin-left: auto;">Priority metrics: <strong style="color: var(--t2);">{ctx['priority_metrics']}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    # ── 5-AGENT PIPELINE VISUAL ──────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Agent Pipeline</h3></div>', unsafe_allow_html=True)

    pipeline_steps = [
        ("⬡", "#1", "Engineer", "Clean & profile"),
        ("◎", "#2", "Analyst", "Stats & insights"),
        ("△", "#3", "Visualizer", "Chart code"),
        ("▫", "#4", "Reporter", "Executive report"),
        ("🛡", "#5", "Quality Gate", "Audit & verify"),
    ]

    step_html = '<div class="pipeline-row">'
    for i, (icon, num, name, tag) in enumerate(pipeline_steps):
        step_html += f"""
        <div class="pipeline-step">
            <div class="step-num">Agent {num}</div>
            <span class="step-icon">{icon}</span>
            <div class="step-name">{name}</div>
            <div class="step-tag">{tag}</div>
        </div>"""
        if i < len(pipeline_steps) - 1:
            step_html += '<div class="pipeline-arrow">→</div>'
    step_html += "</div>"
    st.markdown(step_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LAUNCH BUTTON ────────────────────────────────────────────────────────
    launch_clicked = st.button(
        f"◇  Launch {selected_domain} Analysis Pipeline",
        type="primary",
        use_container_width=True
    )

    if launch_clicked:
        try:
            from agents.crew_agents import DataAnalysisCrew

            processor = st.session_state.processor
            profile = st.session_state.profile

            if profile is None:
                with st.spinner("Preparing data profile…"):
                    if processor is None:
                        processor = DataProcessor()
                        processor.df = st.session_state.df
                        processor.file_path = "data.csv"
                        st.session_state.processor = processor
                    processor.clean_data()
                    profile = processor.generate_profile_text()
                    st.session_state.profile = profile

            # Animated pipeline progress
            progress = st.progress(0, "Initialising agent pipeline…")
            status_html = st.empty()

            file_path = processor.file_path if processor and processor.file_path else "data"
            crew = DataAnalysisCrew(file_path, domain=selected_domain)

            steps = [
                ("⬡  Agent #1 — Data Engineer profiling data…", 18),
                ("◎  Agent #2 — Data Analyst running statistics…", 38),
                ("△  Agent #3 — Visualizer generating charts…", 58),
                ("▫  Agent #4 — Report Writer drafting insights…", 78),
                ("🛡  Agent #5 — Quality Gate auditing report…", 92),
                ("◇  Finalising pipeline…", 100),
            ]

            for text, pct in steps:
                progress.progress(pct, text)
                status_html.markdown(_dot_spinner(text.strip()), unsafe_allow_html=True)
                time.sleep(0.35)

            results = crew.run_full_analysis(profile)
            progress.progress(100, "◇  Pipeline Complete")
            status_html.empty()

            # Check for errors
            has_error = False
            error_msg = ""
            for key, value in results.items():
                if isinstance(value, str) and "Error executing task" in value:
                    has_error = True
                    error_msg = value
                    break

            if has_error:
                if "invalid_api_key" in error_msg.lower() or "Invalid API Key" in error_msg:
                    st.markdown("""
                    <div class="alert-box">
                        <div class="alert-icon">⚠</div>
                        <h3 class="alert-title">Invalid API Key</h3>
                        <p class="alert-desc">Your Groq API key is invalid or expired. Update your .env and restart.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    show_error("Pipeline Error", error_msg)
                st.session_state.analysis_results = None
            else:
                st.session_state.analysis_results = results
                st.session_state.memory_log = results.get("memory_log", [])
                show_success(f"✓  5-agent {selected_domain} pipeline complete — results ready below")

        except ImportError as e:
            show_error("Import Error", str(e),
                       "Make sure agents/crew_agents.py exists and groq is installed.")
        except Exception as e:
            show_error("AI Analysis Error", str(e), traceback.format_exc())

    # ── RESULTS ──────────────────────────────────────────────────────────────
    results = st.session_state.analysis_results
    if not results or not isinstance(results, dict):
        return

    completed_domain = results.get("domain", selected_domain)

    st.markdown('<div class="s-rule"><span class="s-rule-label">Results · ' + completed_domain + ' domain</span></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "Analysis",
        "Report",
        "Quality Gate",
        "Agent Memory",
        "Viz Code",
        "Download"
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1 — AI INSIGHTS (structured display of analysis output)
    # ═══════════════════════════════════════════════════════════════
    with tabs[0]:
        analysis_text = results.get("analysis", "")
        if not analysis_text or "Error executing task" in analysis_text:
            st.info("No analysis data available. Run the pipeline first.")
        else:
            # Header banner
            domain_ctx_r = DOMAIN_CONTEXTS.get(completed_domain, DOMAIN_CONTEXTS["General"])
            st.markdown(f"""
            <div class="insight-header">
                <div>
                    <h3 class="insight-title">{domain_ctx_r['icon']}  {completed_domain} Analysis Insights</h3>
                    <p class="insight-sub">{domain_ctx_r['framing']} · Analysed by Agent #2</p>
                </div>
                <div style="display:flex; gap: 0.75rem; flex-wrap: wrap;">
                    <div class="qg-metric">
                        <span class="qg-metric-val">{len(analysis_text.split())}</span>
                        <span class="qg-metric-lbl">Words</span>
                    </div>
                    <div class="qg-metric">
                        <span class="qg-metric-val">{analysis_text.count(chr(10))}</span>
                        <span class="qg-metric-lbl">Lines</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(analysis_text)

    # ═══════════════════════════════════════════════════════════════
    # TAB 2 — REPORT
    # ═══════════════════════════════════════════════════════════════
    with tabs[1]:
        report_text = results.get("report", "")
        if not report_text or "Error executing task" in report_text:
            st.info("No report available.")
        else:
            st.markdown(f"""
            <div class="insight-header">
                <div>
                    <h3 class="insight-title">▫  Executive Report</h3>
                    <p class="insight-sub">Written by Agent #4 · {completed_domain} framing</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(report_text)

    # ═══════════════════════════════════════════════════════════════
    # TAB 3 — QUALITY GATE
    # ═══════════════════════════════════════════════════════════════
    with tabs[2]:
        qg_text = results.get("quality_gate", "")
        if not qg_text or "Error executing task" in qg_text:
            st.info("Quality Gate audit not available.")
        else:
            # Parse verdict from output
            verdict_color = "rgba(52, 211, 153, 0.1)"
            verdict_class = "verdict-pass"
            verdict_emoji = "✅"
            if "CONDITIONAL PASS" in qg_text.upper():
                verdict_class = "verdict-conditional"
                verdict_emoji = "⚠️"
            elif "FAIL" in qg_text.upper() and "CONDITIONAL" not in qg_text.upper():
                verdict_class = "verdict-fail"
                verdict_emoji = "❌"

            # Extract verdict line
            verdict_line = "PASS"
            for line in qg_text.split("\n"):
                if "Verdict:" in line or "**Verdict:**" in line:
                    raw = line.split("Verdict:")[-1].strip().strip("*").strip()
                    verdict_line = raw if raw else verdict_line
                    break

            # Extract confidence
            confidence = "—"
            for line in qg_text.split("\n"):
                if "Confidence Score:" in line or "**Confidence Score:**" in line:
                    raw = line.split("Confidence Score:")[-1].strip().strip("*").strip()
                    confidence = raw if raw else confidence
                    break

            st.markdown(f"""
            <div class="qg-card">
                <div class="qg-header">
                    <div>
                        <h3 style="font-family: 'Syne', sans-serif; color: var(--t1); margin: 0 0 0.5rem;">
                            🛡 Quality Gate Audit
                        </h3>
                        <p style="color: var(--t2); font-size: 0.85rem; margin: 0;">
                            Agent #5 reviewed every claim in the report against the actual data profile
                        </p>
                    </div>
                    <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                        <span class="verdict-badge {verdict_class}">{verdict_emoji} {verdict_line}</span>
                        <div class="qg-metric">
                            <span class="qg-metric-val">{confidence}</span>
                            <span class="qg-metric-lbl">Confidence</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(qg_text)

    # ═══════════════════════════════════════════════════════════════
    # TAB 4 — AGENT MEMORY LOG
    # ═══════════════════════════════════════════════════════════════
    with tabs[3]:
        memory_log = results.get("memory_log", st.session_state.memory_log or [])
        if not memory_log:
            st.info("No agent memory log available.")
        else:
            st.markdown(f"""
            <div class="insight-header">
                <div>
                    <h3 class="insight-title">◈  Agent-to-Agent Memory Log</h3>
                    <p class="insight-sub">Shows exactly what each agent passed to the next — full pipeline transparency</p>
                </div>
                <div class="qg-metric">
                    <span class="qg-metric-val">{len(memory_log)}</span>
                    <span class="qg-metric-lbl">Handoffs</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Colour palette per agent
            agent_colors = {
                "1": "#8b5cf6",   # violet  — Engineer
                "2": "#34d399",   # emerald — Analyst
                "3": "#fb7185",   # rose    — Visualizer
                "4": "#fbbf24",   # amber   — Reporter
                "5": "#22d3ee",   # cyan    — Quality Gate
            }

            for i, entry in enumerate(memory_log):
                sender = entry.get("sender", "")
                receiver = entry.get("receiver", "")
                label = entry.get("label", "")
                preview = entry.get("preview", "")
                ts = entry.get("timestamp", "")

                # Determine colour by which agent sent
                col_key = "1"
                for k in agent_colors:
                    if f"#{k}" in sender:
                        col_key = k
                        break
                col = agent_colors[col_key]

                st.markdown(f"""
                <div class="memory-card" style="--mc-color: {col}; animation-delay: {i * 0.07}s;">
                    <div class="memory-header">
                        <div>
                            <span class="memory-sender">{sender}</span>
                            <span class="memory-arrow"> → </span>
                            <span class="memory-receiver" style="color: {col};">{receiver}</span>
                        </div>
                        <span class="memory-time">{ts}</span>
                    </div>
                    <div class="memory-label">📦 Payload: {label}</div>
                    <div class="memory-preview">{preview.replace('<', '&lt;').replace('>', '&gt;')}</div>
                </div>
                """, unsafe_allow_html=True)

                # Expandable full content
                with st.expander(f"View full payload from {sender}", expanded=False):
                    st.markdown(entry.get("full", preview))

    # ═══════════════════════════════════════════════════════════════
    # TAB 5 — VIZ CODE
    # ═══════════════════════════════════════════════════════════════
    with tabs[4]:
        viz_code = results.get("visualization", "")
        if not viz_code or "Error executing task" in viz_code:
            st.info("No visualization code available.")
        else:
            st.markdown(f"""
            <div class="insight-header">
                <div>
                    <h3 class="insight-title">△  Visualization Code</h3>
                    <p class="insight-sub">Generated by Agent #3 · Domain: {completed_domain}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.code(viz_code, language="python")

    # ═══════════════════════════════════════════════════════════════
    # TAB 6 — DOWNLOAD
    # ═══════════════════════════════════════════════════════════════
    with tabs[5]:
        report_text = results.get("report", "")
        qg_text = results.get("quality_gate", "")
        analysis_text = results.get("analysis", "")
        viz_code = results.get("visualization", "")
        memory_log = results.get("memory_log", st.session_state.memory_log if hasattr(st.session_state, 'memory_log') else [])
        df_info = None
        if hasattr(st.session_state, 'df') and st.session_state.df is not None:
            df = st.session_state.df
            df_info = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict()
            }

        # Diagnostics block
        missing = []
        if not report_text or "Error executing task" in report_text:
            missing.append("Executive Report")
        if not qg_text or "Error executing task" in qg_text:
            missing.append("Quality Gate Output")
        if not analysis_text or "Error executing task" in analysis_text:
            missing.append("Analysis Output")
        if not viz_code or "Error executing task" in viz_code:
            missing.append("Visualization Code")
        if not df_info:
            missing.append("DataFrame (df)")
        if not memory_log:
            missing.append("Agent Memory Log")

        if missing:
            st.error(f"Downloads unavailable. Missing or invalid: {', '.join(missing)}.\nRun the full pipeline and check for errors in earlier tabs.")
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            df_info = None
            if st.session_state.df is not None:
                df = st.session_state.df
                df_info = {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "column_names": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict()
                }

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("""
                <div class="dl-card">
                    <span class="dl-icon">▫</span>
                    <p class="dl-title">Text Report</p>
                    <p class="dl-desc">Formatted for printing</p>
                </div>
                """, unsafe_allow_html=True)
                pdf_content = generate_pdf_content(report_text)
                st.download_button("↓ Download .txt Report", pdf_content,
                                   f"datamind_report_{ts}.txt", "text/plain",
                                   use_container_width=True)

            with c2:
                st.markdown("""
                <div class="dl-card">
                    <span class="dl-icon">◇</span>
                    <p class="dl-title">JSON Export</p>
                    <p class="dl-desc">Machine-readable all data</p>
                </div>
                """, unsafe_allow_html=True)
                if df_info:
                    json_payload = {
                        "generated_at": datetime.now().isoformat(),
                        "domain": completed_domain,
                        "tool": "DataMind AI v9",
                        "data_info": df_info,
                        "analysis": analysis_text,
                        "report": report_text,
                        "quality_gate": qg_text,
                        "visualization_code": viz_code,
                        "agent_memory_handoffs": len(memory_log),
                    }
                    st.download_button("↓ Download JSON", json.dumps(json_payload, indent=2),
                                       f"datamind_report_{ts}.json", "application/json",
                                       use_container_width=True)

            with c3:
                st.markdown("""
                <div class="dl-card">
                    <span class="dl-icon">🛡</span>
                    <p class="dl-title">Quality Gate Report</p>
                    <p class="dl-desc">Audit findings only</p>
                </div>
                """, unsafe_allow_html=True)
                if qg_text:
                    st.download_button("↓ Download Audit Report", qg_text,
                                       f"datamind_audit_{ts}.md", "text/markdown",
                                       use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("↓ Download Full Markdown Report", report_text,
                               f"datamind_report_{ts}.md", "text/markdown",
                               use_container_width=True)

    # ── Bottom navigation ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    nav_l, nav_r = st.columns(2)
    with nav_l:
        if st.button("← Visualize", use_container_width=True, key="ai_prev"):
            st.session_state.nav_page = "viz"
            st.rerun()
    with nav_r:
        if st.button("Query Lab →", use_container_width=True, key="ai_next"):
            st.session_state.nav_page = "query_lab"
            st.rerun()


# ============================================================================
# ◈  QUERY LAB — SQL & DAX Optimizer (Groq-powered)
# ============================================================================
def page_query_lab():
    _page_flash()
    st.markdown("""
    <div class="pg-hdr">
        <span class="pg-hdr-eyebrow">groq-powered</span>
        <h1 class="pg-hdr-title"><span class="hi">Query</span> Lab</h1>
        <p class="pg-hdr-desc">Optimize SQL & DAX queries · Auto-generate from your dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # Groq status pill
    ql_ok = QL_AVAILABLE
    st.markdown(
        f'''<div style="display:flex;gap:8px;margin-bottom:1.5rem;">
        <span class="ql-badge gemini">{"✓ Groq connected" if ql_ok else "✗ Add GROQ_API_KEY to .env"}</span>
        <span class="ql-badge sql">SQL Optimizer</span>
        <span class="ql-badge dax">DAX Optimizer</span>
        </div>''',
        unsafe_allow_html=True
    )

    df = st.session_state.df
    has_data = df is not None

    sql_tab, dax_tab, auto_tab = st.tabs(["SQL Optimizer", "DAX Optimizer", "Auto-Generate from Data"])

    # ── SQL OPTIMIZER ─────────────────────────────────────────────────────────
    with sql_tab:
        st.markdown("""
        <div class="ql-section">
          <span class="ql-badge sql">SQL</span>
          <p class="ql-section-title">SQL Query Optimizer</p>
          <p class="ql-section-sub">Paste any SQL query — Groq rewrites it for performance, explains every change, and estimates improvement.</p>
        </div>
        """, unsafe_allow_html=True)

        if has_data:
            st.markdown('<div class="ql-tip"><strong>Tip:</strong> Schema from your uploaded dataset is automatically injected into the prompt for context-aware optimization.</div>', unsafe_allow_html=True)

        sql_input = st.text_area(
            "Paste your SQL query",
            height=200,
            placeholder="SELECT * FROM sales WHERE region = 'North' ORDER BY revenue DESC",
            key="ql_sql_input"
        )

        col_run, col_clr = st.columns([2, 1])
        with col_run:
            run_sql = st.button("⚡ Optimize SQL", use_container_width=True, key="ql_run_sql")
        with col_clr:
            if st.button("Clear", use_container_width=True, key="ql_clr_sql"):
                st.session_state.ql_sql_result = ""
                st.rerun()

        if run_sql and sql_input.strip():
            _proc = st.empty()
            _proc.markdown(_dot_spinner("Gemini is optimizing your SQL"), unsafe_allow_html=True)
            with st.spinner("Analyzing query…"):
                schema = _build_schema_context(df) if has_data else "No dataset loaded — optimizing without schema context."
                result = optimize_sql(sql_input.strip(), schema)
                st.session_state.ql_sql_result = result
            _proc.empty()

        if st.session_state.get("ql_sql_result"):
            st.markdown('<div class="ql-result"><span class="ql-result-label">Optimized result</span></div>', unsafe_allow_html=True)
            st.markdown(st.session_state.ql_sql_result)

    # ── DAX OPTIMIZER ─────────────────────────────────────────────────────────
    with dax_tab:
        st.markdown("""
        <div class="ql-section">
          <span class="ql-badge dax">DAX</span>
          <p class="ql-section-title">DAX Measure Optimizer</p>
          <p class="ql-section-sub">Paste any DAX measure or query — Groq optimizes context transitions, CALCULATE usage, and iterator performance.</p>
        </div>
        """, unsafe_allow_html=True)

        dax_input = st.text_area(
            "Paste your DAX measure",
            height=200,
            placeholder="Total Revenue = CALCULATE(SUM(Sales[Revenue]), ALL(Sales[Region]))",
            key="ql_dax_input"
        )

        col_run2, col_clr2 = st.columns([2, 1])
        with col_run2:
            run_dax = st.button("⚡ Optimize DAX", use_container_width=True, key="ql_run_dax")
        with col_clr2:
            if st.button("Clear", use_container_width=True, key="ql_clr_dax"):
                st.session_state.ql_dax_result = ""
                st.rerun()

        if run_dax and dax_input.strip():
            _proc2 = st.empty()
            _proc2.markdown(_dot_spinner("Gemini is optimizing your DAX"), unsafe_allow_html=True)
            with st.spinner("Analyzing measure…"):
                schema = _build_schema_context(df) if has_data else "No dataset — optimizing without schema context."
                result = optimize_dax(dax_input.strip(), schema)
                st.session_state.ql_dax_result = result
            _proc2.empty()

        if st.session_state.get("ql_dax_result"):
            st.markdown('<div class="ql-result"><span class="ql-result-label">Optimized result</span></div>', unsafe_allow_html=True)
            st.markdown(st.session_state.ql_dax_result)

    # ── AUTO-GENERATE ─────────────────────────────────────────────────────────
    with auto_tab:
        st.markdown("""
        <div class="ql-section">
          <span class="ql-badge gemini">Auto-Generate</span>
          <p class="ql-section-title">Generate from Your Dataset</p>
          <p class="ql-section-sub">Gemini reads your column names and types, then writes practical SQL queries and DAX measures tailored to your data.</p>
        </div>
        """, unsafe_allow_html=True)

        if not has_data:
            st.info("Upload a dataset (or load a demo) first — then come back here to auto-generate queries from your actual columns.")
        else:
            st.markdown(f"""
            <div class="ql-tip">
              <strong>Dataset loaded:</strong> {df.shape[0]} rows × {df.shape[1]} columns —
              {", ".join(list(df.columns)[:6])}{"…" if len(df.columns) > 6 else ""}
            </div>
            """, unsafe_allow_html=True)

            col_gsql, col_gdax = st.columns(2)

            with col_gsql:
                if st.button("⚡ Generate SQL queries", use_container_width=True, key="ql_gen_sql"):
                    _ps = st.empty()
                    _ps.markdown(_dot_spinner("Generating SQL from your schema"), unsafe_allow_html=True)
                    with st.spinner("Building queries…"):
                        st.session_state.ql_auto_sql = auto_generate_sql(df)
                    _ps.empty()
                if st.session_state.get("ql_auto_sql"):
                    st.markdown('<div class="s-rule"><span class="s-rule-label">Generated SQL</span></div>', unsafe_allow_html=True)
                    st.markdown(st.session_state.ql_auto_sql)
                    if st.button("Copy to SQL Optimizer →", key="ql_move_sql", use_container_width=True):
                        st.info("Paste the query you want into the SQL Optimizer tab.")

            with col_gdax:
                if st.button("⚡ Generate DAX measures", use_container_width=True, key="ql_gen_dax"):
                    _pd = st.empty()
                    _pd.markdown(_dot_spinner("Generating DAX from your schema"), unsafe_allow_html=True)
                    with st.spinner("Building measures…"):
                        st.session_state.ql_auto_dax = auto_generate_dax(df)
                    _pd.empty()
                if st.session_state.get("ql_auto_dax"):
                    st.markdown('<div class="s-rule"><span class="s-rule-label">Generated DAX</span></div>', unsafe_allow_html=True)
                    st.markdown(st.session_state.ql_auto_dax)

    # ── Bottom navigation ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    nav_l2, nav_r2 = st.columns(2)
    with nav_l2:
        if st.button("← AI Engine", use_container_width=True, key="ql_prev"):
            st.session_state.nav_page = "ai"
            st.rerun()
    with nav_r2:
        st.markdown('<p style="color:var(--t3);font-size:0.8rem;text-align:center;padding-top:0.6rem;">End of pipeline</p>', unsafe_allow_html=True)


# ============================================================================
# 🚀 WELCOME / ONBOARDING PAGE
# ============================================================================
def page_welcome():
    st.markdown("""
    <div class="hero-wrap">
      <span class="hero-eyebrow">multi-agent ai platform</span>
      <h1 class="hero-title">Turn raw data into<br><span class="hi">real insights</span></h1>
      <p class="hero-desc">
        Multi Data Analyst runs five specialised AI agents on your dataset —
        automatically cleaning, analysing, visualising, writing a report, and
        auditing the findings. No code. No configuration.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick nav ────────────────────────────────────────────────
    col_wb, col_wd, col_wn = st.columns([1, 1, 1])
    with col_wb:
        if st.button("→  Start with your data", use_container_width=True, key="welcome_cta_top"):
            st.session_state.onboarded = True
            st.session_state.nav_page = "upload"
            st.rerun()
    with col_wd:
        if st.button("✦  Try HR demo", use_container_width=True, key="welcome_demo_top"):
            _load_demo("hr")
    with col_wn:
        st.markdown('<p class="cta-note" style="margin-top:0.3rem;">No API key? Profiling &amp; charts still work.</p>', unsafe_allow_html=True)

    st.markdown('<div class="s-rule"><span class="s-rule-label">How it works</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="flow-guide">
      <div class="flow-item">
        <span class="flow-num">01</span>
        <div>
          <span class="flow-tag">Upload</span>
          <h3 class="flow-title">Drop any CSV or Excel file</h3>
          <p class="flow-desc">Any size, any shape — the Data Engineer agent validates and profiles your data instantly. Or try a built-in demo dataset to explore right away.</p>
        </div>
      </div>
      <div class="flow-item">
        <span class="flow-num">02</span>
        <div>
          <span class="flow-tag">Configure</span>
          <h3 class="flow-title">Pick a domain</h3>
          <p class="flow-desc">Finance, HR, Marketing, Healthcare or General — each domain tunes the AI agents’ prompts to your field’s vocabulary and priority metrics.</p>
        </div>
      </div>
      <div class="flow-item">
        <span class="flow-num">03</span>
        <div>
          <span class="flow-tag">Run Agents</span>
          <h3 class="flow-title">Five agents fire in sequence</h3>
          <p class="flow-desc">Engineer cleans → Analyst digs → Visualizer plots → Reporter writes → Quality Gate audits every claim. Fully automated, memory-linked pipeline.</p>
        </div>
      </div>
      <div class="flow-item">
        <span class="flow-num">04</span>
        <div>
          <span class="flow-tag">Export</span>
          <h3 class="flow-title">Explore insights and download</h3>
          <p class="flow-desc">Interactive charts, full narrative report, quality audit score, and JSON export — all organised in clean result tabs.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="cta-note" style="text-align:center;margin-top:0.5rem;">Needs a Groq API key in <code>.env</code> for the AI engine. Profiling &amp; charts work without one.</p>', unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================
def main():
    init_session()

    page = render_sidebar()

    pages = {
        "home":       page_home,
        "upload":     page_upload,
        "profile":    page_profile,
        "analysis":   page_analysis,
        "viz":        page_viz,
        "ai":         page_ai,
        "query_lab":  page_query_lab,
    }

    pages.get(page, page_home)()


if __name__ == "__main__":
    main()

