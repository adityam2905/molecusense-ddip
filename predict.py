"""
predict.py  —  CLI Inference
─────────────────────────────
Usage
─────
  # By SMILES
  python predict.py --smiles_a "CC(=O)Oc1ccccc1C(=O)O" --smiles_b "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

  # By drug name (fetches SMILES from PubChem)
  python predict.py --name_a Aspirin --name_b Ibuprofen

  # Specify a checkpoint directory
  python predict.py --name_a Warfarin --name_b Aspirin --checkpoint_dir checkpoints/
"""

import argparse
import sys
sys.path.insert(0, ".")

from utils.inference import DDIInference


def main():
    p = argparse.ArgumentParser(description="DDI-GNN prediction CLI")
    p.add_argument("--smiles_a",       default=None)
    p.add_argument("--smiles_b",       default=None)
    p.add_argument("--name_a",         default=None)
    p.add_argument("--name_b",         default=None)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--show_atoms",     action="store_true",
                   help="Print top attention atoms")
    args = p.parse_args()

    if not (args.smiles_a or args.name_a):
        p.error("Provide --smiles_a or --name_a for drug A")
    if not (args.smiles_b or args.name_b):
        p.error("Provide --smiles_b or --name_b for drug B")

    model  = DDIInference(checkpoint_dir=args.checkpoint_dir, device=args.device)
    result = model.predict(
        smiles_a=args.smiles_a, smiles_b=args.smiles_b,
        name_a=args.name_a,     name_b=args.name_b,
    )

    if result.get("error"):
        print(f"\n[Error] {result['error']}")
        sys.exit(1)

    prob  = result["probability"]
    risk  = result["risk"]
    emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk["level"]]

    print(f"\n{'─'*56}")
    print(f"  Drug A : {result['name_a']}")
    print(f"           {result['smiles_a'][:55]}")
    print(f"  Drug B : {result['name_b']}")
    print(f"           {result['smiles_b'][:55]}")
    print(f"{'─'*56}")
    print(f"  {emoji} Risk level  : {risk['level']}")
    if result["percentile"] is not None:
        print(f"  Percentile   : {result['percentile']:.1f}  (vs. pairs not known to interact)")
    print(f"  Probability  : {prob:.4f}  (assumes half of all pairs interact)")
    print(f"  Assessment   : {risk['description']}")
    print(f"{'─'*56}")

    if args.show_atoms:
        print("\n  Top attention atoms — Drug A:")
        for a in result["top_atoms_a"][:3]:
            print(f"    Atom {a['index']} ({a['symbol']})  score={a['attention']:.3f}"
                  f"  aromatic={a['is_aromatic']}")
        print("\n  Top attention atoms — Drug B:")
        for a in result["top_atoms_b"][:3]:
            print(f"    Atom {a['index']} ({a['symbol']})  score={a['attention']:.3f}"
                  f"  aromatic={a['is_aromatic']}")

    print("\n⚠  Research use only. Not a clinical recommendation.\n")


if __name__ == "__main__":
    main()
