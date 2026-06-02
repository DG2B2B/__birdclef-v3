"""Inspect pulled kernel notebook."""
import json

path = r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\_tmp_pull\birdclef-trainv19.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== METADATA ===")
print(json.dumps(nb.get("metadata", {}), indent=2))
print()
print("=== CELLS COUNT:", len(nb["cells"]))
print()

for i, cell in enumerate(nb["cells"][:3]):
    src = "".join(cell["source"])
    print(f"--- Cell {i} ({cell['cell_type']}) ---")
    print(src[:500])
    print()

if len(nb["cells"]) > 3:
    last = nb["cells"][-1]
    src = "".join(last["source"])
    print(f"--- Last Cell {len(nb['cells'])-1} ({last['cell_type']}) ---")
    print(src[:1000])
