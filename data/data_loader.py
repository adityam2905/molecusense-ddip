"""
data/data_loader.py  —  Phase 1: Real Data Pipeline
─────────────────────────────────────────────────────
Supports three data sources:
  1. DrugBank DDI CSV  (download from drugbank.com after free registration)
  2. TWOSIDES dataset  (Stanford SNAP — polypharmacy side effects)
  3. Built-in toy set  (instant pipeline verification, no download needed)

PubChem SMILES lookup is included as a fallback when SMILES are missing.

Usage
─────
  from data.data_loader import load_dataset

  # Toy data (works immediately)
  pairs = load_dataset(source="toy")

  # DrugBank (after download)
  pairs = load_dataset(source="drugbank", path="data/raw/drugbank_ddi.csv")

  # TWOSIDES
  pairs = load_dataset(source="twosides", path="data/raw/twosides.csv")
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm


# ── Toy dataset ────────────────────────────────────────────────────────────────
# Real SMILES, hand-labelled interactions for instant pipeline testing.

TOY_PAIRS = [
    # (smiles_a, name_a, smiles_b, name_b, label, interaction_type)
    ("CC(=O)Oc1ccccc1C(=O)O",       "Aspirin",
     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  "Ibuprofen",
     1, "GI bleeding risk"),
    ("CC(=O)Oc1ccccc1C(=O)O",       "Aspirin",
     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  "Caffeine",
     1, "Increased aspirin absorption"),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",  "Ibuprofen",
     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  "Caffeine",
     0, "None"),
    ("c1ccc(cc1)CC(C(=O)O)N",       "Phenylalanine",
     "CCOC(=O)c1ccc(cc1)N",         "Benzocaine",
     0, "None"),
    ("CC(=O)Nc1ccc(O)cc1",          "Paracetamol",
     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  "Ibuprofen",
     0, "None"),
    ("CC(=O)Nc1ccc(O)cc1",          "Paracetamol",
     "CC(=O)Oc1ccccc1C(=O)O",       "Aspirin",
     1, "Hepatotoxicity risk"),
    ("CN1CCC[C@H]1c2cccnc2",        "Nicotine",
     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  "Caffeine",
     1, "CNS stimulation"),
    ("CN1CCC[C@H]1c2cccnc2",        "Nicotine",
     "CC(=O)Nc1ccc(O)cc1",          "Paracetamol",
     0, "None"),
    ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "Glucose",
     "CC(=O)Nc1ccc(O)cc1",          "Paracetamol",
     0, "None"),
    ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "Testosterone",
     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  "Ibuprofen",
     1, "Hormone metabolism interference"),
    ("OC(=O)c1ccccc1O",             "Salicylic acid",
     "CC(=O)Oc1ccccc1C(=O)O",       "Aspirin",
     1, "Salicylate toxicity"),
    ("c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", "Pyrene",
     "CC(=O)Nc1ccc(O)cc1",          "Paracetamol",
     0, "None"),
]


# ── PubChem SMILES lookup ──────────────────────────────────────────────────────

def pubchem_smiles(drug_name: str, retries: int = 1) -> str | None:
    """Fetch canonical SMILES for a drug name from PubChem REST API."""
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{requests.utils.quote(drug_name)}/property/CanonicalSMILES/JSON")
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=1.5)

            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                # PubChem may return ConnectivitySMILES even when CanonicalSMILES is requested.
                return props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or props.get("IsomericSMILES")
            elif resp.status_code == 404:
                return None
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None



def batch_smiles_lookup(names: list[str], cache_path: str = "data/smiles_cache.csv") -> dict:
    """
    Look up SMILES for a list of drug names, using a local cache to avoid
    repeated API calls.
    """
    cache = {}
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        cache = dict(zip(df["name"], df["smiles"]))

    missing = [n for n in names if n not in cache or str(cache.get(n, "")).strip() == ""]
    if missing:
        print(f"Fetching SMILES for {len(missing)} drugs from PubChem...")
        for name in tqdm(missing):
            smi = pubchem_smiles(name)
            cache[name] = smi if smi else ""
            time.sleep(0.2)  # be polite to PubChem

        # Save updated cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pd.DataFrame(cache.items(), columns=["name", "smiles"]).to_csv(cache_path, index=False)

    return cache


# ── DrugBank loader ────────────────────────────────────────────────────────────

def load_drugbank(path: str) -> pd.DataFrame:
    """
    Load DrugBank DDI export.

    Expected columns (from drugbank full database XML → CSV conversion):
      Drug1_SMILES, Drug2_SMILES, Interaction_Description
      OR
      Drug1_Name, Drug2_Name, Interaction_Description  (SMILES fetched via PubChem)

    Returns DataFrame with: smiles_a, smiles_b, label, interaction_type, name_a, name_b
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect whether SMILES or names are present
    has_smiles = "drug1_smiles" in df.columns and "drug2_smiles" in df.columns

    if has_smiles:
        df = df.rename(columns={
            "drug1_smiles": "smiles_a",
            "drug2_smiles": "smiles_b",
        })
    else:
        # Fetch SMILES from PubChem
        all_names = list(set(df["drug1_name"].tolist() + df["drug2_name"].tolist()))
        smiles_map = batch_smiles_lookup(all_names)
        df["smiles_a"] = df["drug1_name"].map(smiles_map)
        df["smiles_b"] = df["drug2_name"].map(smiles_map)
        df = df.rename(columns={"drug1_name": "name_a", "drug2_name": "name_b"})

    # Binary label: all rows in DrugBank are interactions
    df["label"] = 1
    df["interaction_type"] = df.get("interaction_description", "Unknown")

    # Generate negative samples (random non-interacting pairs)
    df = _add_negatives(df)

    return df[["smiles_a", "smiles_b", "label", "interaction_type",
               "name_a", "name_b"]].dropna(subset=["smiles_a", "smiles_b"])


def _add_negatives(df: pd.DataFrame, ratio: float = 1.0) -> pd.DataFrame:
    """
    Generate random negative drug pairs at a given positive:negative ratio.
    Ensures no generated pair exists in the positive set.
    """
    pos_set = set(
        zip(df["smiles_a"].tolist(), df["smiles_b"].tolist())
    )
    all_smiles_a = df["smiles_a"].dropna().unique().tolist()
    all_smiles_b = df["smiles_b"].dropna().unique().tolist()

    n_neg = int(len(df) * ratio)
    negatives = []
    rng = np.random.default_rng(42)

    attempts = 0
    while len(negatives) < n_neg and attempts < n_neg * 10:
        a = rng.choice(all_smiles_a)
        b = rng.choice(all_smiles_b)
        if (a, b) not in pos_set and (b, a) not in pos_set:
            negatives.append({
                "smiles_a": a, "smiles_b": b,
                "label": 0, "interaction_type": "None",
                "name_a": "", "name_b": "",
            })
        attempts += 1

    neg_df = pd.DataFrame(negatives)
    return pd.concat([df, neg_df], ignore_index=True).sample(frac=1, random_state=42)


# ── TWOSIDES loader ────────────────────────────────────────────────────────────

def _detect_twosides_columns(header: list[str]) -> dict:
    """
    TWOSIDES is distributed in several formats with different column names.
    This detects which format the file uses and returns a mapping to our
    standard names: drug_a, drug_b, side_effect, prr.

    Known formats
    ─────────────
    Format A  (Tatonetti lab original):
        drug_1_concept_name, drug_2_concept_name, condition_concept_name, PRR

    Format B  (SNAP biodata download):
        Drug1, Drug2, Side_Effect_Name, PRR_mean

    Format C  (some mirrors, no PRR):
        stitch_id_1, stitch_id_2, side_effect_name
        (no PRR column — treat all as interacting)

    Format D  (simplified CSV many repos share):
        drug1, drug2, side_effect
    """
    h = [c.strip().lower() for c in header]

    # Drug A column
    drug_a = next(
        (c for c in header if c.strip().lower() in
         ["drug_1_concept_name", "drug1", "drug_1", "drug1_name",
          "stitch_id_1", "drug1name", "drug a", "druga"]),
        None
    )
    # Drug B column
    drug_b = next(
        (c for c in header if c.strip().lower() in
         ["drug_2_concept_name", "drug2", "drug_2", "drug2_name",
          "stitch_id_2", "drug2name", "drug b", "drugb"]),
        None
    )
    # Side effect / condition
    se = next(
        (c for c in header if any(k in c.strip().lower() for k in
         ["condition", "side_effect", "sideeffect", "effect", "event"])),
        None
    )
    # PRR column (optional)
    prr = next(
        (c for c in header if "prr" in c.strip().lower()),
        None
    )

    if drug_a is None or drug_b is None:
        raise ValueError(
            f"Could not identify drug name columns in TWOSIDES file.\n"
            f"Found columns: {header}\n"
            f"Expected something like 'drug_1_concept_name' / 'drug_2_concept_name' "
            f"or 'Drug1' / 'Drug2'."
        )

    return {"drug_a": drug_a, "drug_b": drug_b, "side_effect": se, "prr": prr}


def load_twosides(path: str, max_pairs: int = 10000, prr_threshold: float = 2.0) -> pd.DataFrame:
    """
    Load the TWOSIDES polypharmacy side-effect dataset.

    Handles all known column-name variants automatically (see _detect_twosides_columns).

    File formats accepted
    ─────────────────────
    • TWOSIDES.csv.gz   (gzip-compressed, ~1 GB uncompressed)
    • TWOSIDES.csv      (plain CSV)
    • 3003377s-s6.csv   (original supplement file name)

    Processing steps
    ────────────────
    1. Peek at header to detect column layout
    2. Read in 500k-row chunks to keep memory low (full file is ~43M rows)
    3. Filter by PRR >= prr_threshold when PRR column is present
    4. Deduplicate drug pairs keeping highest-PRR side effect per pair
    5. Sample max_pairs positive pairs
    6. Generate equal-size random negative pairs
    7. Fetch SMILES from PubChem for all unique drug names (cached locally)

    Parameters
    ----------
    path          : path to TWOSIDES CSV or CSV.GZ
    max_pairs     : max positive drug pairs to use (default 10,000)
    prr_threshold : minimum PRR for significance; ignored when no PRR column

    Returns
    -------
    pd.DataFrame with columns: smiles_a, smiles_b, label, interaction_type, name_a, name_b
    """
    print(f"Loading TWOSIDES from: {path}")

    # ── 1. Detect column layout ───────────────────────────────────────────────
    header_df = pd.read_csv(path, nrows=0)
    col_map   = _detect_twosides_columns(list(header_df.columns))
    print(f"  Detected columns -> drug_a='{col_map['drug_a']}', "
          f"drug_b='{col_map['drug_b']}', "
          f"side_effect='{col_map['side_effect']}', "
          f"prr='{col_map['prr']}'")

    read_cols = [c for c in [col_map["drug_a"], col_map["drug_b"],
                              col_map["side_effect"], col_map["prr"]]
                 if c is not None]

    # ── 2. Chunked read ───────────────────────────────────────────────────────
    chunks    = []
    total_raw = 0
    for chunk in pd.read_csv(path, chunksize=500_000, low_memory=False):
        total_raw += len(chunk)
        chunk = chunk[read_cols].copy()

        # Filter by PRR if column exists
        if col_map["prr"] and col_map["prr"] in chunk.columns:
            chunk[col_map["prr"]] = pd.to_numeric(chunk[col_map["prr"]], errors="coerce")
            chunk = chunk[chunk[col_map["prr"]] >= prr_threshold]

        if len(chunk):
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Rows scanned: {total_raw:,} | After PRR filter: {len(df):,}")

    # ── 3. Rename to standard columns ─────────────────────────────────────────
    rename = {col_map["drug_a"]: "name_a", col_map["drug_b"]: "name_b"}
    if col_map["side_effect"]:
        rename[col_map["side_effect"]] = "interaction_type"
    if col_map["prr"]:
        rename[col_map["prr"]] = "prr"
    df = df.rename(columns=rename)

    if "interaction_type" not in df.columns:
        df["interaction_type"] = "Drug interaction"
    if "prr" not in df.columns:
        df["prr"] = 1.0

    # ── 4. Normalise drug names ───────────────────────────────────────────────
    df["name_a"] = df["name_a"].astype(str).str.strip().str.title()
    df["name_b"] = df["name_b"].astype(str).str.strip().str.title()

    # Remove rows where drug names are numeric IDs (STITCH IDs — can't look up)
    id_pattern = r"^\s*-?\d+\s*$"
    df = df[~df["name_a"].str.match(id_pattern) & ~df["name_b"].str.match(id_pattern)]

    # ── 5. Deduplicate pairs (keep strongest PRR side effect per pair) ────────
    df = df.sort_values("prr", ascending=False)
    
    # Vectorized deduplication (much faster than df.apply on 33M rows)
    a = df["name_a"].values
    b = df["name_b"].values
    mask = a > b
    df["_p1"] = np.where(mask, b, a)
    df["_p2"] = np.where(mask, a, b)
    df = df.drop_duplicates(subset=["_p1", "_p2"], keep="first").drop(columns=["_p1", "_p2", "prr"])

    print(f"  Unique drug pairs: {len(df):,}")

    # ── 6. Sample (Cache-aware) ───────────────────────────────────────────────
    # Prioritize drug pairs we ALREADY have SMILES for in the cache.
    # This prevents expensive API timeouts and 'no valid pairs' errors.
    cache_path = "data/smiles_cache.csv"
    cached_names = set()
    if os.path.exists(cache_path):
        try:
            cache_df = pd.read_csv(cache_path)
            cached_names = set(cache_df[cache_df["smiles"].notna()]["name"].tolist())
        except: pass

    def cache_score(row):
        score = 0
        if row["name_a"] in cached_names: score += 1
        if row["name_b"] in cached_names: score += 1
        return score

    if len(df) > max_pairs:
        # Sample candidates, then pick those with best cache coverage
        candidates = df.sample(n=min(len(df), max_pairs * 5), random_state=42).copy()
        candidates["_score"] = (candidates["name_a"].isin(cached_names).astype(int) + 
                               candidates["name_b"].isin(cached_names).astype(int))
        df = candidates.sort_values("_score", ascending=False).head(max_pairs)
        print(f"  Sampled {max_pairs:,} pairs (Priority: Cache hits)")

    df["label"] = 1
    df = df[["name_a", "name_b", "label", "interaction_type"]].reset_index(drop=True)

    # ── 7. PubChem SMILES lookup ──────────────────────────────────────────────
    all_names = list(set(df["name_a"].tolist() + df["name_b"].tolist()))
    print(f"  Unique drug names to look up: {len(all_names)}")
    smiles_map = batch_smiles_lookup(all_names)

    df["smiles_a"] = df["name_a"].map(smiles_map)
    df["smiles_b"] = df["name_b"].map(smiles_map)

    before = len(df)
    df = df.dropna(subset=["smiles_a", "smiles_b"])
    df = df[(df["smiles_a"] != "") & (df["smiles_b"] != "")]
    print(f"  Pairs with valid SMILES: {len(df):,} (dropped {before - len(df)} — no PubChem entry)")


    if len(df) == 0:
        raise RuntimeError(
            "No valid drug pairs after SMILES lookup. "
            "Check that drug names in the file are standard English names "
            "(not STITCH/CID IDs) and that you have internet access for PubChem."
        )

    # ── 8. Add negatives ──────────────────────────────────────────────────────
    df = _add_negatives(df)

    return df[["smiles_a", "smiles_b", "label", "interaction_type",
               "name_a", "name_b"]].reset_index(drop=True)


# ── Unified entry point ────────────────────────────────────────────────────────

def load_dataset(source: str = "toy", path: str = None, max_pairs: int = 10000) -> pd.DataFrame:
    """
    Load DDI pairs from the specified source.

    Parameters
    ----------
    source    : "toy" | "drugbank" | "twosides" | "csv"
    path      : required for drugbank / csv; auto-detected for twosides
    max_pairs : max positive pairs for twosides (default 10,000)

    Returns
    -------
    pd.DataFrame with columns:
        smiles_a, smiles_b, label, interaction_type, name_a, name_b
    """
    if source == "toy":
        rows = []
        for smi_a, name_a, smi_b, name_b, label, itype in TOY_PAIRS:
            rows.append({
                "smiles_a": smi_a, "name_a": name_a,
                "smiles_b": smi_b, "name_b": name_b,
                "label": label, "interaction_type": itype,
            })
        return pd.DataFrame(rows)

    if source == "drugbank":
        return load_drugbank(path)

    if source == "twosides":
        # Auto-detect common TWOSIDES path if not provided
        if path is None:
            for candidate in ["data/TWOSIDES.csv.gz", "data/twosides.csv.gz",
                              "data/twosides.csv"]:
                if os.path.exists(candidate):
                    path = candidate
                    break
        if path is None:
            raise FileNotFoundError(
                "TWOSIDES data file not found. Place TWOSIDES.csv.gz in data/ "
                "or specify --data path."
            )
        return load_twosides(path, max_pairs=max_pairs)

    if source == "csv":
        df = pd.read_csv(path)
        required = {"smiles_a", "smiles_b", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        if "interaction_type" not in df.columns:
            df["interaction_type"] = "Unknown"
        if "name_a" not in df.columns:
            df["name_a"] = ""
        if "name_b" not in df.columns:
            df["name_b"] = ""
        return df

    raise ValueError(f"Unknown source: {source!r}. Choose: toy, drugbank, twosides, csv")


# ── Class imbalance analysis ───────────────────────────────────────────────────

def dataset_stats(df: pd.DataFrame):
    """Print a quick summary of the dataset."""
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    print(f"\n{'─'*40}")
    print(f"  Total pairs   : {len(df):,}")
    print(f"  Interactions  : {n_pos:,}  ({100*n_pos/len(df):.1f}%)")
    print(f"  Non-interact  : {n_neg:,}  ({100*n_neg/len(df):.1f}%)")
    print(f"  Pos/neg ratio : {n_pos/max(n_neg,1):.2f}")
    print(f"  Interaction types: {df['interaction_type'].nunique()}")
    print(f"{'─'*40}\n")


def validate_twosides_file(path: str) -> dict:
    """
    Check a TWOSIDES file before loading — returns a status dict.
    Called by the Streamlit app to give early feedback to the user.

    Returns
    -------
    {
      "ok":       bool,
      "format":   str   (detected format name),
      "columns":  list  (actual columns found),
      "n_rows":   int   (estimated row count),
      "error":    str | None,
      "col_map":  dict  (drug_a, drug_b, side_effect, prr)
    }
    """
    result = {"ok": False, "format": None, "columns": [], "n_rows": 0,
              "error": None, "col_map": {}}
    try:
        header_df = pd.read_csv(path, nrows=0)
        result["columns"] = list(header_df.columns)
        col_map = _detect_twosides_columns(result["columns"])
        result["col_map"] = col_map

        # Estimate row count cheaply
        sample = pd.read_csv(path, nrows=5000)
        import os as _os
        file_size = _os.path.getsize(path)
        bytes_per_row = file_size / max(len(sample), 1)
        result["n_rows"] = int(file_size / max(bytes_per_row, 1))

        # Identify format
        cols_lower = [c.lower() for c in result["columns"]]
        if "drug_1_concept_name" in cols_lower:
            result["format"] = "Tatonetti lab original (drug_1_concept_name)"
        elif "drug1" in cols_lower:
            result["format"] = "SNAP biodata (Drug1 / Drug2)"
        elif "stitch_id_1" in cols_lower:
            result["format"] = "STITCH ID format (numeric IDs — names not lookupable)"
            result["error"] = (
                "This file uses STITCH numeric IDs instead of drug names. "
                "PubChem cannot look up SMILES for numeric IDs. "
                "Please use the Tatonetti lab or SNAP version with drug names."
            )
            return result
        else:
            result["format"] = "Unknown / custom"

        result["ok"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


def preview_twosides(path: str, n: int = 5) -> pd.DataFrame:
    """Return the first n rows of a TWOSIDES file for display."""
    return pd.read_csv(path, nrows=n)


if __name__ == "__main__":
    df = load_dataset("toy")
    dataset_stats(df)
    print(df.head())
