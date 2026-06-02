"""Piste B: Create V11 = V2 + Model_3=0.10 + TAX_SMOOTHING post-processing.
Based on Nina EoS v9's taxonomic smoothing approach."""
import json
from pathlib import Path

SRC = Path(r'c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v2.ipynb')
DST = Path(r'c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v11.ipynb')

with open(str(SRC), 'r', encoding='utf-8') as f:
    nb = json.load(f)

# MOD 1: Change solutions dict
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if "solutions = {" in src and "type_add" in src:
        src = src.replace("weight':0.0", "weight':0.10")
        src = src.replace("weight':1.0", "weight':0.90")
        nb['cells'][i]['source'] = [src]
        print(f'Cell {i}: Model_3=0.10, Model_4=0.90')
        break

# MOD 2: Insert TAX_SMOOTHING cell after submission = f_add() (cell 14)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if src.strip().startswith('submission = f_add()'):
        tax_cell_code = """# V11: Taxonomic smoothing (genus + class level)
# Adapted from Nina EoS v9 TAX_SMOOTHING_POSTPROC
import pandas as pd, numpy as np, os

# Load taxonomy
tax_paths = [
    "/kaggle/input/competitions/birdclef-2026/taxonomy.csv",
    "/kaggle/input/birdclef-2026/taxonomy.csv",
]
tax = None
for p in tax_paths:
    if os.path.exists(p):
        tax = pd.read_csv(p)
        break

if tax is not None:
    species_to_genus = {}
    species_to_class = {}
    for _, r in tax.iterrows():
        label = str(r["primary_label"])
        sci = str(r.get("scientific_name", ""))
        cls = str(r.get("class_name", ""))
        genus = sci.split(" ")[0] if " " in sci else sci
        species_to_genus[label] = genus
        species_to_class[label] = cls

    cols = list(submission.columns)
    genus_groups = {}
    class_groups = {}
    for c in cols:
        g = species_to_genus.get(c, c)
        cl = species_to_class.get(c, "")
        genus_groups.setdefault(g, []).append(c)
        if cl:
            class_groups.setdefault(cl, []).append(c)

    multi_genus = {g: m for g, m in genus_groups.items() if len(m) > 1}
    multi_class = {cl: m for cl, m in class_groups.items() if len(m) > 1}
    print(f"Tax smoothing: {len(multi_genus)} multi-genus groups, {len(multi_class)} multi-class groups")

    alpha_genus = 0.15
    alpha_class = 0.05
    probs = submission[cols].values.copy()

    # Genus-level smoothing
    for members in multi_genus.values():
        idxs = [cols.index(m) for m in members if m in cols]
        if len(idxs) < 2:
            continue
        mean_probs = probs[:, idxs].mean(axis=1, keepdims=True)
        probs[:, idxs] = (1 - alpha_genus) * probs[:, idxs] + alpha_genus * mean_probs

    # Class-level smoothing  
    for members in multi_class.values():
        idxs = [cols.index(m) for m in members if m in cols]
        if len(idxs) < 2:
            continue
        mean_probs = probs[:, idxs].mean(axis=1, keepdims=True)
        probs[:, idxs] = (1 - alpha_class) * probs[:, idxs] + alpha_class * mean_probs

    submission[cols] = np.clip(probs, 0.0, 1.0)
    print("Taxonomic smoothing applied")
else:
    print("taxonomy.csv not found - skipping tax smoothing")
"""
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [tax_cell_code]
        }
        nb['cells'].insert(i + 1, new_cell)
        print(f'Inserted TAX_SMOOTHING cell after cell {i}')
        break

# Save
with open(str(DST), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print(f'V11 saved: {DST.name} ({len(nb["cells"])} cells)')
