import json
from pathlib import Path

# Piste A: V10 minimal - V2 with ONLY Model_3 weight changed
SRC = Path("c:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-v3/notebooks/eos_submit_v2.ipynb")
DST = Path("c:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-v3/notebooks/eos_submit_v10.ipynb")

print(f"Reading {SRC}")
with open(str(SRC), "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Found {len(nb['cells'])} cells")
found = False
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "solutions = {" in src and "type_add" in src:
        before = src
        src = src.replace("weight':0.0", "weight':0.10")
        src = src.replace("weight':1.0", "weight':0.90")
        nb["cells"][i]["source"] = [src]
        print(f"Cell {i}: changed solutions dict")
        print(f"  BEFORE: {before.strip()[:150]}")
        print(f"  AFTER:  {src.strip()[:150]}")
        found = True
        break

if not found:
    print("ERROR: Could not find solutions dict!")
else:
    with open(str(DST), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Saved to {DST}")
    print(f"File exists: {DST.exists()}, Size: {DST.stat().st_size}")
