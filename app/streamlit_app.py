"""
MolecuSense — Drug-Drug Interaction Predictor (Streamlit UI)
"""

import sys
import os
import json
import pandas as pd
import streamlit as st

LIVE_APP_URL = "https://molecusense-ddip.streamlit.app/"
MAX_BATCH_ROWS = 200  # each unknown name can cost a PubChem request

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.inference import DDIInference
from utils.drug_lookup import resolve, cached_names
from utils.visualize import draw_molecule_attention


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


def load_result(name):
    path = os.path.join(ROOT, "results", name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

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
    desc = result["risk"]["description"]
    css = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[level]
    emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[level]
    pct = result.get("percentile")
    headline = (f"Percentile: <b>{int(pct)}</b> of 100" if pct is not None
                else f"Probability: <b>{result['probability']:.1%}</b>")
    st.markdown(f"""
    <div class="{css}">
        <div style="font-size: 1.2rem; margin-bottom: 4px;">{emoji} <b>Model score: {level.title()}</b></div>
        <div style="font-size: 1.1rem; margin-bottom: 8px;">{headline}</div>
        <div style="font-weight:400; opacity: 0.9; line-height: 1.4;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)
    if pct is not None:
        st.caption(
            f"Model probability {result['probability']:.1%} (calibrated for a population where "
            "half of all pairs interact, as in training, so it overstates how often real drug "
            "pairs interact). Bands use the percentile: high = above 95% of drug pairs not known "
            "to interact, medium = above 80%. The score mostly reflects how often each drug "
            "appears in FDA adverse-event reports, not the specific pair (see System Info)."
        )


# ── Pages ──────────────────────────────────────────────────────────────────────
def page_single(model, input_method):
    hero_section("MolecuSense", "Research model scores for drug pairs", "Single pair")
    if input_method.startswith("Drug name"):
        st.caption(f"{len(cached_names()):,} drug names are available offline; "
                   "others are looked up on PubChem.")

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
                st.markdown('<div class="result-header">Model Score</div>', unsafe_allow_html=True)
                risk_badge(res)
                if res["notes"]:
                    st.warning("**Treat this score with extra caution.**\n\n"
                               + "\n".join(f"- {n}" for n in res["notes"]))

                st.markdown('<div class="result-header">How the Model Reads Each Molecule</div>',
                            unsafe_allow_html=True)
                st.caption(
                    "Atoms coloured by the model's attention. Each molecule is encoded on its own, "
                    "so a drug's map is the same whatever it's paired with: it shows what the "
                    "model picks out in that molecule, not why this pair got its score."
                )
                try:
                    img_a = draw_molecule_attention(res["smiles_a"], res["attention_a"])
                    img_b = draw_molecule_attention(res["smiles_b"], res["attention_b"])
                    ci1, ci2 = st.columns(2)
                    ci1.image(img_a, caption=res["name_a"], width=380)
                    ci2.image(img_b, caption=res["name_b"], width=380)
                except Exception as e: st.warning(f"Image error: {e}")

def page_batch(model):
    hero_section("Batch Scoring", "Score many drug pairs at once", "Batch")
    st.markdown(f"Upload a CSV with columns `drug_a` and `drug_b` (up to {MAX_BATCH_ROWS} rows).")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if not uploaded:
        return

    try:
        df = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    missing = [c for c in ["drug_a", "drug_b"] if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return
    if len(df) > MAX_BATCH_ROWS:
        st.warning(f"The file has {len(df):,} rows; only the first {MAX_BATCH_ROWS} are scored.")
        df = df.head(MAX_BATCH_ROWS)

    st.write(f"Rows to score: {len(df):,}")
    # Results live in session state so they survive reruns (e.g. clicking
    # Download), keyed by the uploaded file so a new upload starts fresh.
    key = (uploaded.name, uploaded.size)
    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
        st.session_state["batch"] = {"key": key, **_score_batch(model, df)}
    saved = st.session_state.get("batch")
    if not saved or saved["key"] != key:
        return

    st.caption(saved["lookup_summary"])
    out_df = saved["results"]
    st.dataframe(out_df, use_container_width=True)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results",
        data=csv_bytes,
        file_name="batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _score_batch(model, df) -> dict:
    # Look each distinct name up once (local list first, then PubChem).
    names = sorted({n.strip() for n in pd.concat([df["drug_a"], df["drug_b"]]) if n.strip()})
    lookups = {}
    with st.spinner(f"Looking up {len(names)} distinct drug names..."):
        for n in names:
            lookups[n] = resolve(n)
    sources = pd.Series([r["source"] or "not found" for r in lookups.values()]).value_counts()
    summary = "Name lookups: " + ", ".join(f"{v} {k}" for k, v in sources.items())

    results = []
    progress = st.progress(0)
    for i, (a, b) in enumerate(zip(df["drug_a"].str.strip(), df["drug_b"].str.strip())):
        row = {"drug_a": a, "drug_b": b, "model_score": None, "percentile": None,
               "probability": None, "notes": "", "error": ""}
        la, lb = lookups.get(a), lookups.get(b)
        if not a or not b:
            row["error"] = "Missing drug name"
        elif not la["smiles"] or not lb["smiles"]:
            bad = [(n, l) for n, l in ((a, la), (b, lb)) if not l["smiles"]]
            row["error"] = "; ".join(
                (f"{n!r}: {l['reason']}" if l.get("reason") else f"Unknown drug {n!r}")
                + (f" (did you mean {', '.join(l['suggestions'])}?)" if l["suggestions"] else "")
                for n, l in bad)
        else:
            res = model.predict(smiles_a=la["smiles"], smiles_b=lb["smiles"],
                                name_a=a, name_b=b, fetch_smiles=False)
            if res.get("error"):
                row["error"] = res["error"]
            else:
                pct = res["percentile"]
                row.update(model_score=res["risk"]["level"],
                           percentile=round(pct, 1) if pct is not None else None,
                           probability=round(res["probability"], 4),
                           notes=" ".join(res["notes"]))
        results.append(row)
        progress.progress((i + 1) / len(df))

    return {"results": pd.DataFrame(results), "lookup_summary": summary}

def page_info(model):
    hero_section("Model Intelligence", "Architecture and Training Metrics", "System Data")
    
    if not model.meta:
        st.warning("No training metadata found.")
        return

    st.markdown('<div class="result-header">GNN Performance (test AUROC)</div>', unsafe_allow_html=True)
    drug_split = load_result("eval_drug_split.json")
    c1, c2, c3 = st.columns(3)
    c1.metric("New pairs of known drugs", f"{model.meta.get('test_auroc', 0):.3f}")
    if drug_split:
        c2.metric("Drugs never seen in training", f"{drug_split['gnn']['test_auroc']:.3f}")
    c3.metric("Random guessing", "0.500")

    audit = load_result("pair_audit_checkpoints.json")
    if audit:
        st.info(
            f"**What the score mostly measures.** Across {audit['n_pairs']:,} pairs of "
            f"{audit['n_drugs']} drugs, one fixed score per drug explains "
            f"{audit['r2_per_drug_probability']:.0%} of the model's output. The model mainly "
            "rates how often each drug is reported in FDA adverse-event data, not the "
            "chemistry of the specific pair, and simple baselines that score each drug on "
            "its own do as well or better. When the training data was rebalanced so drug "
            "frequency gave nothing away, neither this model nor the baselines did better "
            "than chance. Treat results as a research score, not an interaction prediction."
        )

    cal = model.meta.get("calibration")
    if cal:
        st.markdown('<div class="result-header">Calibration (test set)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature", f"{model.meta['temperature']:.2f}")
        c2.metric("ECE", f"{cal['ece_after']:.3f}", f"{cal['ece_after'] - cal['ece_before']:+.3f}",
                  delta_color="inverse")
        c3.metric("Brier score", f"{cal['brier_after']:.3f}",
                  f"{cal['brier_after'] - cal['brier_before']:+.3f}", delta_color="inverse")

    st.markdown('<div class="result-header">Architecture Specs</div>', unsafe_allow_html=True)
    st.json({
        "Encoder": "GATConv x 3",
        "Pair combination": "[A + B, |A - B|] (same answer in either drug order)",
        "Classifier": f"MLP (512 -> 128 -> {model.meta.get('n_classes', 1)})",
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
    st.sidebar.caption("Graph attention network · temperature-calibrated")
    st.sidebar.link_button("🔗 Live app", LIVE_APP_URL, use_container_width=True)

    if has_checkpoint(): st.sidebar.success("Model loaded ✓")
    
    pages = ["Single Pair", "Batch Predict", "System Info"]
    mode = st.sidebar.selectbox("Navigate", pages, index=0)
    
    input_method = "Drug name"
    if mode == "Single Pair":
        input_method = st.sidebar.radio("Method", ["Drug name", "SMILES string"])
        
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="disclaimer">RESEARCH ONLY — NOT CLINICAL</div>', unsafe_allow_html=True)

    model = load_model()
    if mode == "Single Pair": page_single(model, input_method)
    elif mode == "Batch Predict": page_batch(model)
    else: page_info(model)

if __name__ == "__main__":
    main()
