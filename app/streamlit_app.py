"""
DDI-GNN — Drug-Drug Interaction Predictor
─────────────────────────────────────────────
Modernized, professional UI for GAT + RL calibration.
"""

import sys
import os
import io
import json
import subprocess
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.inference import DDIInference, pubchem_smiles
from utils.visualize import draw_molecule_attention, draw_attention_colorbar
from data.data_loader import validate_twosides_file, preview_twosides


# ── Initialization ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDI-GNN | MolecuSense",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    # 1. Inject Fonts
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    
    # 2. Inject CSS
    if os.path.exists(css_path):
        with open(css_path, encoding='utf-8') as f:
            css_content = f.read()
        # Use a single-line style tag injection to prevent markdown parsing
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    
    # 3. Render Background Atoms
    st.markdown("""
    <div class="floating-atoms">
        <div class="atom atom-1"></div><div class="atom atom-2"></div><div class="atom atom-3"></div>
        <div class="atom atom-4"></div><div class="atom atom-5"></div><div class="atom atom-6"></div>
        <div class="atom atom-7"></div><div class="atom atom-8"></div><div class="atom atom-9"></div>
        <div class="atom atom-10"></div><div class="atom atom-11"></div><div class="atom atom-12"></div>
        <div class="atom atom-13"></div><div class="atom atom-14"></div><div class="atom atom-15"></div>
        <div class="bond-line bond-1"></div><div class="bond-line bond-2"></div>
        <div class="bond-line bond-3"></div><div class="bond-line bond-4"></div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return DDIInference(checkpoint_dir=os.path.join(ROOT, "checkpoints"), device="cpu")


# ── Helpers ────────────────────────────────────────────────────────────────────
def has_checkpoint(): return os.path.exists(os.path.join(ROOT, "checkpoints", "best_model.pt"))
def has_rl(): return os.path.exists(os.path.join(ROOT, "checkpoints", "rl_policy.pt"))

def hero_section(title, subtitle, badge=None):
    badge_html = f'<div class="hero-badge">{badge}</div>' if badge else ''
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)

def risk_badge(result):
    level = result["risk"]["level"]
    prob, desc = result["probability"], result["risk"]["description"]
    css = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[level]
    emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[level]
    st.markdown(f"""
    <div class="{css}">
        <div style="font-size: 1.2rem; margin-bottom: 4px;">{emoji} <b>{level} RISK detected</b></div>
        <div style="font-size: 1.1rem; margin-bottom: 8px;">Interaction probability: <b>{prob:.1%}</b></div>
        <div style="font-weight:400; opacity: 0.9; line-height: 1.4;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Pages ──────────────────────────────────────────────────────────────────────
def page_setup():
    hero_section("Pipeline Configuration", "Prepare datasets and train neural structures", "Step 1: Setup")
    
    st.markdown("""
    <div class="step-box">
        Follow these steps to prepare the <b>TWOSIDES</b> dataset and train your Graph Attention Network (GNN).
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📂 Step 1 — Dataset Preparation", expanded=not os.path.exists(os.path.join(ROOT, "data", "TWOSIDES.csv.gz"))):
        st.markdown("""
        1. Download `3003377s-s6.csv` (~1GB) from [Tatonetti Lab](http://tatonettilab.org/resources/tatonetti-stm.html).
        2. Place it in the `data/` directory.
        3. Rename it to `TWOSIDES.csv.gz` (or use the original CSV).
        """)
        data_path = st.text_input("Data Path", value="data/TWOSIDES.csv.gz")
        if st.button("Validate Dataset"):
            with st.spinner("Validating..."):
                ok, msg = validate_twosides_file(data_path)
                if ok: st.success(msg); st.session_state["validated_path"] = data_path
                else: st.error(msg)

    with st.expander("🧠 Step 2 — GNN Training", expanded=has_checkpoint()):
        st.markdown("### Training Parameters")
        c1, c2 = st.columns(2)
        pairs = c1.number_input("Max Pairs", 1000, 100000, 5000)
        epochs = c2.number_input("Epochs", 1, 500, 50)
        
        if st.button("🚀 Start GNN Training", type="primary", use_container_width=True):
            cmd = [sys.executable, "train.py", "--max_pairs", str(pairs), "--epochs", str(epochs)]
            st.info(f"Running: {' '.join(cmd)}")
            subprocess.Popen(cmd, cwd=ROOT)
            st.warning("Training started in background. Refresh later.")

    if has_checkpoint():
        with st.expander("🧪 Step 3 — RL Calibration Agent", expanded=not has_rl()):
            hero_section("RL Fine-Tuning", "Optimize confidence scores using REINFORCE", "Bonus Phase")
            if st.button("🧠 Train RL Agent", type="primary", use_container_width=True):
                cmd = [sys.executable, "train_rl.py", "--episodes", "500"]
                subprocess.Popen(cmd, cwd=ROOT)
                st.info("RL training initiated.")

def page_single(model, input_method):
    hero_section("MolecuSense", "Predict complex drug-drug interactions", "Step 2: Analysis")

    st.markdown('<div class="result-header">Molecular Inputs</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="drug-card"><div class="drug-card-title drug-card-title-a"><div class="drug-icon drug-icon-a">A</div>Primary Agent</div>', unsafe_allow_html=True)
        if input_method.startswith("Drug name"):
            name_a, smiles_a = st.text_input("Name", value="Aspirin", key="na"), None
        else:
            name_a, smiles_a = "Drug A", st.text_input("SMILES", value="CC(=O)Oc1ccccc1C(=O)O", key="sa")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="drug-card"><div class="drug-card-title drug-card-title-b"><div class="drug-icon drug-icon-b">B</div>Secondary Agent</div>', unsafe_allow_html=True)
        if input_method.startswith("Drug name"):
            name_b, smiles_b = st.text_input("Name", value="Ibuprofen", key="nb"), None
        else:
            name_b, smiles_b = "Drug B", st.text_input("SMILES", value="CC(C)Cc1ccc(cc1)C(C)C(=O)O", key="sb")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔬 Execute Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing graph structures..."):
            res = model.predict(smiles_a=smiles_a, smiles_b=smiles_b, name_a=name_a, name_b=name_b, fetch_smiles="name" in input_method.lower())
            if res.get("error"):
                st.error(res["error"])
            else:
                st.markdown('<div class="result-header">Interference Analysis</div>', unsafe_allow_html=True)
                risk_badge(res)
                
                # RL Info
                rl = res.get("rl_info")
                if rl and rl.get("rl_active"):
                    st.markdown(f'<div class="rl-box"><span class="rl-badge">🧠 RL CALIBRATED</span> &nbsp; Prob: <b>{res["probability"]:.1%}</b> (Adj: {rl["adjustment"]:+.3f})</div>', unsafe_allow_html=True)

                # Visualization
                st.markdown('<div class="result-header">Attention Mapping</div>', unsafe_allow_html=True)
                try:
                    img_a = draw_molecule_attention(res["smiles_a"], res["attention_a"])
                    img_b = draw_molecule_attention(res["smiles_b"], res["attention_b"])
                    ci1, ci2 = st.columns(2)
                    ci1.image(img_a, caption=res["name_a"], width=380)
                    ci2.image(img_b, caption=res["name_b"], width=380)
                except Exception as e: st.warning(f"Image error: {e}")

def page_batch(model):
    hero_section("Throughput Screening", "Bulk analysis of chemical pairs", "Step 3: Batch")
    st.markdown("Upload a CSV with columns: `drug_a`, `drug_b`.")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if not uploaded:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    missing = [c for c in ["drug_a", "drug_b"] if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.write(f"Rows loaded: {len(df):,}")
    run = st.button("Run Batch Prediction", type="primary", use_container_width=True)

    if not run:
        return

    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, row in df.iterrows():
        name_a = str(row.get("drug_a", "")).strip()
        name_b = str(row.get("drug_b", "")).strip()
        res = model.predict(name_a=name_a, name_b=name_b, fetch_smiles=True)

        if res.get("error"):
            results.append({
                "drug_a": name_a,
                "drug_b": name_b,
                "probability": None,
                "risk": None,
                "error": res["error"],
            })
        else:
            results.append({
                "drug_a": name_a,
                "drug_b": name_b,
                "probability": res["probability"],
                "risk": res["risk"]["level"],
                "error": "",
            })

        if (i + 1) % 5 == 0 or i + 1 == len(df):
            progress.progress((i + 1) / len(df))
            status.write(f"Processed {i + 1} / {len(df)}")

    out_df = pd.DataFrame(results)
    st.dataframe(out_df, use_container_width=True)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results",
        data=csv_bytes,
        file_name="batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

def page_info(model):
    hero_section("Model Intelligence", "Architecture and Training Metrics", "System Data")
    
    if not model.meta:
        st.warning("No training metadata found.")
        return

    st.markdown('<div class="result-header">GNN Performance</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Val AUROC", f"{model.meta.get('best_val_auroc', 0):.4f}")
    c2.metric("Test AUROC", f"{model.meta.get('test_auroc', 0):.4f}")
    c3.metric("Test AUPRC", f"{model.meta.get('test_auprc', 0):.4f}")

    st.markdown('<div class="result-header">Architecture Specs</div>', unsafe_allow_html=True)
    st.json({
        "Encoder": "GATConv x 3",
        "Classifier": "MLP (512 -> 128 -> 1)",
        "Attention Heads": model.meta.get("args", {}).get("heads", 4),
        "Hidden Dim": model.meta.get("args", {}).get("hidden", 64),
    })

    curves = os.path.join(ROOT, "checkpoints", "training_curves.png")
    if os.path.exists(curves):
        st.image(curves, caption="GNN Training Curves", width=900)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    load_css()
    
    st.sidebar.markdown('<h1 style="margin-top:0">⚗️ MolecuSense</h1>', unsafe_allow_html=True)
    st.sidebar.caption("GAT + REINFORCE Calibration")
    
    ckpt, rl = has_checkpoint(), has_rl()
    if ckpt: st.sidebar.success("GNN Loaded ✓")
    if rl: st.sidebar.success("RL Loaded ✓")
    
    pages = ["Single Pair", "Batch Predict", "System Info"]
    mode = st.sidebar.selectbox("Navigate", pages, index=0)
    
    input_method = "Drug name"
    if mode == "Single Pair":
        input_method = st.sidebar.radio("Method", ["Drug name (PubChem)", "SMILES string"])
        
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="disclaimer">RESEARCH ONLY — NOT CLINICAL</div>', unsafe_allow_html=True)

    model = load_model()
    if mode == "Single Pair": page_single(model, input_method)
    elif mode == "Batch Predict": page_batch(model)
    else: page_info(model)

if __name__ == "__main__":
    main()
