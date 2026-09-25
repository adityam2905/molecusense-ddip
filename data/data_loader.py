"""
data/data_loader.py  —  Loads drug pairs for training
─────────────────────────────────────────────────────
Sources:
  twosides  TWOSIDES (FDA adverse-event reports): real interacting pairs, plus
            generated non-interacting pairs (see --negatives in train.py)
  csv       Your own file with smiles_a, smiles_b, label columns
  toy       12 hand-written pairs, for checking the pipeline runs

Drug names are turned into SMILES with the local list in data/smiles_cache.csv,
falling back to PubChem.

Usage
─────
  from data.data_loader import load_dataset
  pairs = load_dataset(source="twosides", path="data/TWOSIDES.csv.gz")
"""

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.drug_lookup import pubchem_smiles


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


# ── PubChem SMILES lookup (shared with the app via utils/drug_lookup.py) ───────


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


def _generate_negative_name_pairs(
    names: list, exclude_pairs: set, n_neg: int, seed: int = 42, weights=None
) -> pd.DataFrame:
    """
    Sample random (name_a, name_b) pairs from `names`, rejecting any pair
    present in `exclude_pairs` (normalized (min, max) name tuples).

    `weights` (one per name) sets how often each drug is drawn. Passing each
    drug's count in the positive pairs makes every drug appear about as often
    in negatives as in positives, so "how often a drug shows up in FDA
    reports" stops being a shortcut for the label. None draws uniformly.

    Call it with the FULL set of known interacting pairs as `exclude_pairs`,
    even when the positive set
    being labeled has already been subsampled — so a pair that IS a real,
    documented interaction (just not one that was sampled into the positive
    set) never gets mislabeled as a negative.

    Also guards against generating the SAME negative pair more than once:
    a repeated pair would otherwise risk landing in both train and test after
    random_split, which is a direct train/test leak of the exact same graphs
    and label.
    """
    rng = np.random.default_rng(seed)
    names = list(names)
    p = None
    if weights is not None:
        p = np.asarray(weights, dtype=float)
        p = p / p.sum()
    negatives = []
    seen = set()
    attempts = 0
    max_attempts = max(n_neg * 20, 1000)

    while len(negatives) < n_neg and attempts < max_attempts:
        a, b = rng.choice(names, size=2, replace=False, p=p)
        key = (a, b) if a <= b else (b, a)
        if key not in exclude_pairs and key not in seen:
            seen.add(key)
            negatives.append({
                "name_a": a, "name_b": b,
                "label": 0, "interaction_type": "None",
            })
        attempts += 1

    if len(negatives) < n_neg:
        print(f"  [warn] Only generated {len(negatives)}/{n_neg} negative pairs "
              f"after {attempts} attempts — the known-interacting-pair space "
              f"may be densely covered by this drug pool.")

    return pd.DataFrame(negatives, columns=["name_a", "name_b", "label", "interaction_type"])


def _generate_balanced_negatives(pos_df: pd.DataFrame, exclude_pairs: set,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Non-interacting pairs in which every drug appears EXACTLY as many times as
    it does in the interacting pairs, so how often a drug shows up carries no
    information about the label.

    Each drug gets one "stub" per interacting pair it's in. Stubs are shuffled
    and paired off; pairs that are invalid (same drug, a known interaction, or
    a repeat) go back into the pool and are reshuffled. Stubs still unpaired
    after that are placed by swapping partners with an already-accepted pair
    (x, y) + (u, v) -> (x, u) + (y, v), which keeps every drug's count intact.
    """
    rng = np.random.default_rng(seed)
    stubs = sorted(pos_df["name_a"].tolist() + pos_df["name_b"].tolist())
    rng.shuffle(stubs)

    def key(a, b):
        return (a, b) if a <= b else (b, a)

    def ok(a, b, taken):
        return a != b and key(a, b) not in exclude_pairs and key(a, b) not in taken

    accepted, taken = [], set()
    pool = stubs
    for _ in range(200):
        rng.shuffle(pool)
        leftover = []
        for i in range(0, len(pool) - 1, 2):
            a, b = pool[i], pool[i + 1]
            if ok(a, b, taken):
                accepted.append((a, b))
                taken.add(key(a, b))
            else:
                leftover += [a, b]
        if len(pool) % 2:
            leftover.append(pool[-1])
        if len(leftover) >= len(pool) - 1:  # no progress this round
            pool = leftover
            break
        pool = leftover
        if len(pool) < 2:
            break

    # Swap step for stubs the random pairing couldn't place.
    for _ in range(len(pool) * 200):
        if len(pool) < 2:
            break
        x, y = pool[0], pool[1]
        j = int(rng.integers(len(accepted)))
        u, v = accepted[j]
        if rng.random() < 0.5:
            u, v = v, u
        taken.discard(key(u, v))
        # (no `taken | {...}` here: copying the set every attempt made this
        # step quadratic and took hours at 20k pairs)
        if ok(x, u, taken) and ok(y, v, taken) and key(x, u) != key(y, v):
            accepted[j] = (x, u)
            accepted.append((y, v))
            taken |= {key(x, u), key(y, v)}
            pool = pool[2:]
        else:
            taken.add(key(u, v))
            rng.shuffle(pool)

    neg_df = pd.DataFrame([{"name_a": a, "name_b": b, "label": 0, "interaction_type": "None"}
                           for a, b in accepted],
                          columns=["name_a", "name_b", "label", "interaction_type"])
    return neg_df, _trim_positives(pos_df, pool, rng)


def _trim_positives(pos_df: pd.DataFrame, leftover: list, rng) -> pd.DataFrame:
    """
    Stubs left unpaired belong to "hub" drugs reported with nearly every other
    drug, so there aren't enough non-interacting partners to balance them. Drop
    that many of their interacting pairs instead, preferring pairs where BOTH
    drugs have a leftover stub (which balances both at once).
    """
    from collections import Counter
    need = Counter(leftover)
    if not need:
        return pos_df
    order = rng.permutation(len(pos_df))
    a, b = pos_df["name_a"].to_numpy(), pos_df["name_b"].to_numpy()
    drop = set()
    for i in order:  # pass 1: both drugs over-represented
        if need[a[i]] > 0 and need[b[i]] > 0:
            drop.add(i); need[a[i]] -= 1; need[b[i]] -= 1
    for i in order:  # pass 2: at least one drug over-represented
        if i not in drop and (need[a[i]] > 0 or need[b[i]] > 0):
            drop.add(i)
            for d in (a[i], b[i]):
                need[d] = max(0, need[d] - 1)
    print(f"  Balanced negatives: dropped {len(drop)} interacting pairs of hub drugs "
          f"that have too few non-interacting partners")
    return pos_df.drop(pos_df.index[sorted(drop)])


def _dedupe_by_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Different drug names can map to the same molecule (synonyms, salt forms),
    and the model only ever sees the molecule. Drop:
      - rows whose SMILES RDKit can't parse
      - pairs of a molecule with itself
      - negatives that are structurally identical to a positive pair
      - repeats of the same molecule pair (would leak across train/test)
    Row order is otherwise preserved.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    canon = {}
    for s in set(df["smiles_a"]) | set(df["smiles_b"]):
        mol = Chem.MolFromSmiles(s)
        canon[s] = Chem.MolToSmiles(mol) if mol is not None else None
    ca, cb = df["smiles_a"].map(canon), df["smiles_b"].map(canon)

    invalid = ca.isna() | cb.isna()
    self_pair = ~invalid & (ca == cb)
    key = pd.Series([tuple(sorted((x, y))) if x and y else None for x, y in zip(ca, cb)],
                    index=df.index)
    positive_keys = set(key[(df["label"] == 1) & ~invalid])
    neg_is_positive = (df["label"] == 0) & key.isin(positive_keys)
    keep = ~(invalid | self_pair | neg_is_positive)
    duplicate = key[keep].duplicated(keep="first").reindex(df.index, fill_value=False)
    keep &= ~duplicate

    print(f"  Structure cleanup: dropped {int(invalid.sum())} invalid SMILES, "
          f"{int(self_pair.sum())} same-molecule pairs, {int(neg_is_positive.sum())} "
          f"negatives matching a positive, {int(duplicate.sum())} duplicate pairs")
    return df[keep].reset_index(drop=True)


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


DATASET_CACHE_DIR = "data/cache"
# Bump when the pair-building logic changes, so stale cached datasets are rebuilt.
DATASET_CACHE_VERSION = 2


def load_twosides(path: str, max_pairs: int = 10000, prr_threshold: float = 2.0,
                  negatives: str = "degree") -> pd.DataFrame:
    """
    Build (or load from data/cache/) the TWOSIDES pair dataset.

    Building scans the full ~43M-row file (~5 min), and the result also depends
    on the current SMILES cache, so the finished pair table is saved once and
    reused. That keeps repeated runs fast and guarantees they see identical data.
    """
    stat = os.stat(path)
    key = (f"v{DATASET_CACHE_VERSION}_{os.path.basename(path)}_{stat.st_size}_"
           f"{max_pairs}_{prr_threshold}_{negatives}")
    cache_file = os.path.join(DATASET_CACHE_DIR, f"twosides_{key}.csv")
    if os.path.exists(cache_file):
        print(f"Loading cached TWOSIDES pairs: {cache_file}")
        return pd.read_csv(cache_file, keep_default_na=False)

    df = _build_twosides(path, max_pairs, prr_threshold, negatives)
    os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
    df.to_csv(cache_file, index=False)
    print(f"  Saved pair table to {cache_file}")
    return df


def _build_twosides(path: str, max_pairs: int, prr_threshold: float,
                    negatives: str) -> pd.DataFrame:
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
    5. Sample max_pairs positive pairs (cache-aware)
    6. Generate equal-size random negative pairs, checked against the FULL
       deduplicated pair universe from step 4 (not just the max_pairs sample),
       so a negative can never be a real interaction that wasn't sampled.
       negatives="degree" (default) draws drugs in proportion to their
       positive count. This only partly removes the popularity shortcut:
       popular drugs' candidates are mostly rejected as known interactions.
       "balanced" makes every drug appear exactly as often in negatives as in
       positives (dropping positives of "hub" drugs that lack non-interacting
       partners); on TWOSIDES it removes almost all learnable signal, so it's
       kept as a diagnostic. "uniform" draws every drug equally (the original
       behaviour, which let drug popularity alone predict the label)
    7. Fetch SMILES from PubChem for all unique drug names (cached locally)
    8. Clean up by molecule: drop invalid SMILES, same-molecule pairs, and
       duplicate molecule pairs (different names can share one structure)

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

    # ── 2-5. Chunked read + incremental PRR filter + dedup ───────────────────
    # The real TWOSIDES file is ~43M rows and ~34M of them survive PRR>=2.0 —
    # concatenating all filtered rows into one frame before deduping (the old
    # approach) needs the whole thing in memory at once, which can exceed
    # available RAM on ordinary hardware. Instead, reduce each 500k-row chunk
    # down to one (max-PRR) row per drug pair immediately, then fold that into
    # a running "best row per pair" frame — peak memory stays bounded by one
    # chunk plus the unique-pair count seen so far, not the full file.
    id_pattern = r"^\s*-?\d+\s*$"
    rename = {col_map["drug_a"]: "name_a", col_map["drug_b"]: "name_b"}
    if col_map["side_effect"]:
        rename[col_map["side_effect"]] = "interaction_type"
    if col_map["prr"]:
        rename[col_map["prr"]] = "prr"

    running_best = None
    total_raw = 0
    total_after_prr = 0

    for chunk in pd.read_csv(path, chunksize=500_000, low_memory=False):
        total_raw += len(chunk)
        chunk = chunk[read_cols].copy()

        if col_map["prr"] and col_map["prr"] in chunk.columns:
            chunk[col_map["prr"]] = pd.to_numeric(chunk[col_map["prr"]], errors="coerce")
            chunk = chunk[chunk[col_map["prr"]] >= prr_threshold]

        if len(chunk) == 0:
            continue
        total_after_prr += len(chunk)

        chunk = chunk.rename(columns=rename)
        if "interaction_type" not in chunk.columns:
            chunk["interaction_type"] = "Drug interaction"
        if "prr" not in chunk.columns:
            chunk["prr"] = 1.0

        chunk["name_a"] = chunk["name_a"].astype(str).str.strip().str.title()
        chunk["name_b"] = chunk["name_b"].astype(str).str.strip().str.title()
        chunk = chunk[~chunk["name_a"].str.match(id_pattern) & ~chunk["name_b"].str.match(id_pattern)]
        if len(chunk) == 0:
            continue

        a = chunk["name_a"].values
        b = chunk["name_b"].values
        mask = a > b
        chunk["_p1"] = np.where(mask, b, a)
        chunk["_p2"] = np.where(mask, a, b)

        # Reduce this chunk to one row per pair (cheap: at most 500k rows)
        idx = chunk.groupby(["_p1", "_p2"])["prr"].idxmax()
        chunk_best = chunk.loc[idx]

        if running_best is None:
            running_best = chunk_best
        else:
            combined = pd.concat([running_best, chunk_best], ignore_index=True)
            idx2 = combined.groupby(["_p1", "_p2"])["prr"].idxmax()
            running_best = combined.loc[idx2]

    df = running_best if running_best is not None else pd.DataFrame(
        columns=["name_a", "name_b", "interaction_type", "prr", "_p1", "_p2"]
    )
    print(f"  Rows scanned: {total_raw:,} | After PRR filter: {total_after_prr:,}")

    # Full universe of known-interacting name pairs (normalized, order-independent).
    # We keep this from BEFORE subsampling so that negative sampling below can be
    # checked against everything TWOSIDES reports as interacting, not just the
    # max_pairs subset — otherwise a "negative" pair could just be a real
    # interaction that happened not to be sampled into the positive set.
    full_pos_pairs = set(zip(df["_p1"], df["_p2"]))
    df = df.drop(columns=["_p1", "_p2", "prr"])

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
        except Exception as e:
            print(f"  [warn] Could not read SMILES cache at {cache_path}: {e}")

    if len(df) > max_pairs:
        # Sample candidates, then pick those with best cache coverage
        candidates = df.sample(n=min(len(df), max_pairs * 5), random_state=42).copy()
        candidates["_score"] = (candidates["name_a"].isin(cached_names).astype(int) + 
                               candidates["name_b"].isin(cached_names).astype(int))
        df = candidates.sort_values("_score", ascending=False).head(max_pairs)
        print(f"  Sampled {max_pairs:,} pairs (Priority: Cache hits)")

    df["label"] = 1
    df = df[["name_a", "name_b", "label", "interaction_type"]].reset_index(drop=True)

    # ── 7. Negative sampling (name-space, checked against the FULL pair universe) ─
    # Drawn from the same drug names as the sampled positives, so negatives stay
    # plausible, but rejected against `full_pos_pairs` (every interacting pair
    # TWOSIDES reports) rather than just this subsample — this avoids mislabeling
    # a real, documented interaction as "no interaction".
    # sorted(), not list(set(...)): set iteration order for strings changes
    # between Python processes (hash randomization), which silently made the
    # seeded negative sampling produce different pairs on every run.
    pool_names = sorted(set(df["name_a"].tolist() + df["name_b"].tolist()))
    if negatives == "balanced":
        neg_df, df = _generate_balanced_negatives(df, full_pos_pairs, seed=42)
    elif negatives in ("degree", "uniform"):
        weights = None
        if negatives == "degree":
            counts = pd.concat([df["name_a"], df["name_b"]]).value_counts()
            weights = [counts[n] for n in pool_names]
        neg_df = _generate_negative_name_pairs(
            pool_names, full_pos_pairs, n_neg=len(df), seed=42, weights=weights
        )
    else:
        raise ValueError(f"Unknown negatives mode {negatives!r}; "
                         "use 'balanced', 'degree' or 'uniform'")
    print(f"  Generated {len(neg_df):,} {negatives} negative pairs "
          f"(checked against {len(full_pos_pairs):,} known interacting pairs)")

    df = pd.concat([df, neg_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # ── 8. PubChem SMILES lookup ──────────────────────────────────────────────
    all_names = list(set(df["name_a"].tolist() + df["name_b"].tolist()))
    print(f"  Unique drug names to look up: {len(all_names)}")
    smiles_map = batch_smiles_lookup(all_names)

    df["smiles_a"] = df["name_a"].map(smiles_map)
    df["smiles_b"] = df["name_b"].map(smiles_map)

    before = len(df)
    df = df.dropna(subset=["smiles_a", "smiles_b"])
    df = df[(df["smiles_a"] != "") & (df["smiles_b"] != "")]
    print(f"  Pairs with valid SMILES: {len(df):,} (dropped {before - len(df)} — no PubChem entry)")

    df = _dedupe_by_structure(df)

    if len(df) == 0:
        raise RuntimeError(
            "No valid drug pairs after SMILES lookup. "
            "Check that drug names in the file are standard English names "
            "(not STITCH/CID IDs) and that you have internet access for PubChem."
        )

    return df[["smiles_a", "smiles_b", "label", "interaction_type",
               "name_a", "name_b"]].reset_index(drop=True)


# ── Unified entry point ────────────────────────────────────────────────────────

def load_dataset(source: str = "toy", path: str = None, max_pairs: int = 10000,
                 negatives: str = "degree") -> pd.DataFrame:
    """
    Load DDI pairs from the specified source.

    Parameters
    ----------
    source    : "toy" | "twosides" | "csv"
    path      : required for csv; auto-detected for twosides
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
        return load_twosides(path, max_pairs=max_pairs, negatives=negatives)

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

    raise ValueError(f"Unknown source: {source!r}. Choose: toy, twosides, csv")


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


if __name__ == "__main__":
    df = load_dataset("toy")
    dataset_stats(df)
    print(df.head())
