"""
utils/drug_lookup.py  —  Drug name → SMILES
─────────────────────────────────────────────
Shared by training (data/data_loader.py) and the app (utils/inference.py), so
both resolve names the same way:

  1. data/smiles_cache.csv (~900 names already looked up; case-insensitive)
  2. PubChem, with successful lookups remembered for the rest of the session

Names that can't be found get spelling suggestions from the cache.
"""

import os
import time
import difflib
import functools

import pandas as pd
import requests

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "smiles_cache.csv")

# International names the cache stores under a different name.
SYNONYMS = {"paracetamol": "acetaminophen"}

_pubchem_hits: dict = {}


@functools.lru_cache(maxsize=1)
def _cache() -> dict:
    """lowercase name -> (display name, SMILES), only for names with a SMILES."""
    if not os.path.exists(CACHE_PATH):
        return {}
    df = pd.read_csv(CACHE_PATH, keep_default_na=False)
    return {str(n).strip().lower(): (str(n).strip(), s)
            for n, s in zip(df["name"], df["smiles"]) if str(s).strip()}


def cached_names() -> list[str]:
    """Drug names that resolve offline, from data/smiles_cache.csv."""
    return sorted(display for display, _ in _cache().values())


def pubchem_smiles(name: str, timeout: float = 8.0, retries: int = 2) -> str | None:
    """Look a name up on PubChem. Successful results are kept for the session."""
    key = name.strip().lower()
    if key in _pubchem_hits:
        return _pubchem_hits[key]
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{requests.utils.quote(name.strip())}/property/CanonicalSMILES/JSON")
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                props = resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                # PubChem may return ConnectivitySMILES even when CanonicalSMILES is requested.
                smiles = (props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
                          or props.get("IsomericSMILES"))
                if smiles:
                    _pubchem_hits[key] = smiles
                return smiles
            if resp.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(0.5 * (attempt + 1))
    return None


def suggest(name: str, n: int = 3) -> list[str]:
    """Closest known drug names, for misspellings like 'Asprin'."""
    cache = _cache()
    keys = list(cache) + list(SYNONYMS)
    matches = difflib.get_close_matches(name.strip().lower(), keys, n=n, cutoff=0.75)
    return [cache[m][0] if m in cache else m.title() for m in matches]


def resolve(name: str, use_pubchem: bool = True) -> dict:
    """
    Returns {"smiles": str | None, "source": "cache" | "pubchem" | None,
             "suggestions": [...] (only when not found)}.
    """
    key = name.strip().lower()
    key = SYNONYMS.get(key, key)
    hit = _cache().get(key)
    if hit:
        return {"smiles": hit[1], "source": "cache", "suggestions": []}
    if use_pubchem:
        smiles = pubchem_smiles(name)
        if smiles:
            return {"smiles": smiles, "source": "pubchem", "suggestions": []}
    return {"smiles": None, "source": None, "suggestions": suggest(name)}
