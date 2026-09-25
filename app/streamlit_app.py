"""
MolecuSense — Streamlit app
"""

import os
import sys
import json

import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.inference import DDIInference
from utils.drug_lookup import resolve, cached_names
from utils.visualize import draw_molecule_attention

MAX_BATCH_ROWS = 200  # each unknown name can cost a PubChem request
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

st.set_page_config(page_title="MolecuSense", page_icon="⚗️", layout="wide")


# ── Setup ──────────────────────────────────────────────────────────────────────
def load_css():
    with open(os.path.join(os.path.dirname(__file__), "style.css"), encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _model_version() -> tuple:
    """
    Changes whenever the model files or the code that uses them change. Streamlit
    Cloud keeps cached objects across code updates, so without this key a push
    leaves the app calling new page code on an old DDIInference object.
    """
    paths = [os.path.join(CHECKPOINT_DIR, f) for f in sorted(os.listdir(CHECKPOINT_DIR))]
    for pkg in ("utils", "models"):
        folder = os.path.join(ROOT, pkg)
        paths += [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".py")]
    return tuple((p, os.path.getmtime(p)) for p in paths)


@st.cache_resource
def load_model(version: tuple) -> DDIInference:
    return DDIInference(checkpoint_dir=CHECKPOINT_DIR, device="cpu")


def load_result(name):
    path = os.path.join(ROOT, "results", name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><div class="hero-title">{title}</div>'
                f'<div class="hero-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


# ── Score a pair ───────────────────────────────────────────────────────────────
def score_card(res):
    level = res["risk"]["level"]
    pct = res["percentile"]
    if pct is None:
        detail = f"Probability {res['probability']:.1%} (uncalibrated model)"
    else:
        detail = f"Higher than {int(pct)}% of drug pairs not known to interact."
    emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[level]
    st.markdown(f'<div class="score score-{level.lower()}">'
                f'<div class="score-title">{emoji} Model score: {level.title()}</div>'
                f'<div>{detail}</div></div>', unsafe_allow_html=True)


def page_single(model):
    hero("MolecuSense", "Score how likely two drugs are to be reported as interacting")

    by_name = st.radio("Enter drugs by", ["Name", "SMILES"], horizontal=True) == "Name"
    if by_name:
        st.caption(f"{len(cached_names()):,} drug names work offline; others are looked up "
                   "on PubChem.")

    cols = st.columns(2)
    inputs = {}
    for col, label, name, smiles in ((cols[0], "A", "Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
                                     (cols[1], "B", "Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O")):
        with col.container(border=True):
            st.markdown(f"**Drug {label}**")
            if by_name:
                inputs[label] = (st.text_input("Name", value=name, key=f"n{label.lower()}"), None)
            else:
                inputs[label] = (f"Drug {label}",
                                 st.text_input("SMILES", value=smiles, key=f"s{label.lower()}"))

    if not st.button("Score pair", type="primary", width="stretch"):
        return

    with st.spinner("Scoring..."):
        res = model.predict(name_a=inputs["A"][0], smiles_a=inputs["A"][1],
                            name_b=inputs["B"][0], smiles_b=inputs["B"][1],
                            fetch_smiles=by_name)
    if res.get("error"):
        st.error(res["error"])
        return

    section("Result")
    score_card(res)
    if res.get("notes"):
        st.warning("**Treat this score with extra caution.**\n\n"
                   + "\n".join(f"- {n}" for n in res["notes"]))
    st.caption("The score mostly reflects how often each drug appears in FDA reports, not the "
               "chemistry of this specific pair (see *About the model*). Research use only.")

    with st.expander("Details"):
        st.markdown(
            f"- **Percentile:** {res['percentile']}\n"
            f"- **Probability:** {res['probability']:.1%}. This assumes half of all drug pairs "
            "interact (the training mix), so it overstates real-world rates.\n"
            f"- **{res['name_a']}:** `{res['smiles_a']}`\n"
            f"- **{res['name_b']}:** `{res['smiles_b']}`"
        )

    section("How the model reads each molecule")
    st.caption("Brighter atoms get more attention. Each molecule is read on its own, so a "
               "drug's map is the same whatever it's paired with.")
    try:
        cols = st.columns(2)
        for col, key in ((cols[0], "a"), (cols[1], "b")):
            col.image(draw_molecule_attention(res[f"smiles_{key}"], res[f"attention_{key}"]),
                      caption=res[f"name_{key}"], width="stretch")
    except Exception as e:
        st.warning(f"Couldn't draw the molecules: {e}")


# ── Batch scoring ──────────────────────────────────────────────────────────────
EXAMPLE_CSV = "drug_a,drug_b\nAspirin,Ibuprofen\nWarfarin,Aspirin\nMetformin,Caffeine\n"


def page_batch(model):
    hero("Batch scoring", f"Score up to {MAX_BATCH_ROWS} drug pairs from a CSV file")
    st.markdown("Upload a CSV with the columns `drug_a` and `drug_b`.")
    st.download_button("Download an example CSV", EXAMPLE_CSV, file_name="example_pairs.csv",
                       mime="text/csv")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if not uploaded:
        return

    try:
        df = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        return
    missing = [c for c in ("drug_a", "drug_b") if c not in df.columns]
    if missing:
        st.error(f"Missing column(s): {', '.join(missing)}")
        return
    if len(df) > MAX_BATCH_ROWS:
        st.warning(f"The file has {len(df):,} rows; only the first {MAX_BATCH_ROWS} are scored.")
        df = df.head(MAX_BATCH_ROWS)

    # Results live in session state so they survive reruns (e.g. clicking
    # Download), keyed by the uploaded file so a new upload starts fresh.
    key = (uploaded.name, uploaded.size)
    if st.button(f"Score {len(df):,} pairs", type="primary", width="stretch"):
        st.session_state["batch"] = {"key": key, **_score_batch(model, df)}
    saved = st.session_state.get("batch")
    if not saved or saved["key"] != key:
        return

    st.caption(saved["lookup_summary"])
    st.dataframe(saved["results"], width="stretch", hide_index=True)
    st.download_button("Download results", saved["results"].to_csv(index=False).encode("utf-8"),
                       file_name="molecusense_scores.csv", mime="text/csv", width="stretch")


def _score_batch(model, df) -> dict:
    # Look each distinct name up once (local list first, then PubChem).
    names = sorted({n.strip() for n in pd.concat([df["drug_a"], df["drug_b"]]) if n.strip()})
    with st.spinner(f"Looking up {len(names)} drug names..."):
        lookups = {n: resolve(n) for n in names}
    sources = pd.Series([r["source"] or "not found" for r in lookups.values()]).value_counts()
    summary = "Name lookups: " + ", ".join(f"{v} {k}" for k, v in sources.items())

    results = []
    progress = st.progress(0.0)
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
                row.update(model_score=res["risk"]["level"].title(),
                           percentile=round(pct, 1) if pct is not None else None,
                           probability=round(res["probability"], 4),
                           notes=" ".join(res.get("notes", [])))
        results.append(row)
        progress.progress((i + 1) / len(df))
    progress.empty()

    return {"results": pd.DataFrame(results), "lookup_summary": summary}


# ── About the model ────────────────────────────────────────────────────────────
def page_about(model):
    hero("About the model", "How well it works, and what the score really measures")
    meta = model.meta
    if not meta:
        st.warning("No training metadata found.")
        return

    section("Accuracy on held-out test pairs (AUROC)")
    drug_split = load_result("eval_drug_split.json")
    c1, c2, c3 = st.columns(3)
    c1.metric("Drugs seen in training", f"{meta.get('test_auroc', 0):.3f}")
    if drug_split:
        c2.metric("Drugs never seen in training", f"{drug_split['gnn']['test_auroc']:.3f}")
    c3.metric("Random guessing", "0.500")
    st.caption("AUROC: 0.5 is guessing, 1.0 is perfect.")

    audit = load_result("pair_audit_checkpoints.json")
    baselines = load_result("eval_checkpoints.json")
    if audit:
        popularity = ""
        if baselines:
            popularity = (f" Just counting how often each drug appears scores "
                          f"{baselines['baselines']['drug_popularity']['test_auroc']:.3f}.")
        st.info(
            f"**What the score really measures.** One fixed score per drug explains "
            f"{audit['r2_per_drug_probability']:.0%} of the model's output, so it mostly rates "
            f"how often each drug appears in FDA reports, not how the two drugs interact."
            f"{popularity} Treat results as a research score, not medical advice."
        )

    section("Model")
    args = meta.get("args", {})
    n_train, n_val, n_test = meta.get("split_sizes", [0, 0, 0])
    st.markdown(
        f"- **Network:** 3-layer graph attention network, {args.get('heads', 4)} attention heads\n"
        f"- **Data:** TWOSIDES, {n_train + n_val + n_test:,} drug pairs "
        f"({n_train:,} train / {n_val:,} validation / {n_test:,} test)\n"
        f"- **Training:** best of {meta.get('epochs_run', '?')} epochs "
        f"(epoch {meta.get('best_epoch', '?')}), chosen by validation AUROC"
    )

    cal = meta.get("calibration")
    if cal:
        section("Calibration (test set)")
        c1, c2 = st.columns(2)
        c1.metric("Calibration error (ECE)", f"{cal['ece_after']:.3f}",
                  f"{cal['ece_after'] - cal['ece_before']:+.3f}", delta_color="inverse")
        c2.metric("Brier score", f"{cal['brier_after']:.3f}",
                  f"{cal['brier_after'] - cal['brier_before']:+.3f}", delta_color="inverse")
        st.caption("Lower is better; the change is from temperature scaling.")

    curves = os.path.join(CHECKPOINT_DIR, "training_curves.png")
    if os.path.exists(curves):
        section("Training curves")
        st.image(curves, width="stretch")


# ── Main ───────────────────────────────────────────────────────────────────────
PAGES = {"Score a pair": page_single, "Batch scoring": page_batch, "About the model": page_about}


def main():
    load_css()
    st.sidebar.title("⚗️ MolecuSense")
    st.sidebar.caption("Drug-pair scores from a graph neural network trained on FDA reports.")
    page = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Research project. Not medical advice.")

    PAGES[page](load_model(_model_version()))


if __name__ == "__main__":
    main()
