"""Create V12 from V11 by adding Model_2 to solutions dict."""
import json
import shutil
from pathlib import Path

SRC = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v11.ipynb")
DST = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v12.ipynb")

# Copy V11 → V12
shutil.copy(str(SRC), str(DST))

# Load V12
with open(str(DST), "r", encoding="utf-8") as f:
    nb = json.load(f)

# Modify solutions dict in cell 2
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "solutions = {" in src and "type_add" in src:
        # Replace the 2-model block with 3-model block
        src = src.replace(
            "{'Model':'Model_3','subm':'subm_3.csv','weight':0.10,'xSED':['    ,    '],'LB':'0.928'},",
            "{'Model':'Model_2','subm':'subm_2.csv','weight':0.05,'xSED':['    ,    '],'LB':'0.917'},\n  {'Model':'Model_3','subm':'subm_3.csv','weight':0.10,'xSED':['    ,    '],'LB':'0.928'},"
        )
        src = src.replace("'weight':0.90", "'weight':0.85")
        nb["cells"][i]["source"] = [src]
        print(f"Cell {i}: Added Model_2 to solutions dict (3-way direct blend)")
        print(f"  Model_2=0.05, Model_3=0.10, Model_4=0.85")
        break

# Save
with open(str(DST), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

# Verify
with open(str(DST), "r", encoding="utf-8") as f:
    vnb = json.load(f)
src2 = "".join(vnb["cells"][2]["source"])
checks = [
    ("direct mode", "'direct'" in src2),
    ("Model_2 present", "Model_2" in src2),
    ("Model_3=0.10", "0.10" in src2),
    ("Model_4=0.85", "0.85" in src2),
    ("3 models", src2.count("'Model_'") >= 3),
]
all_ok = True
for name, ok in checks:
    status = "OK" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {status}: {name}")

print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
print(f"V12 saved to: {DST}")
print("NOT PUSHED - local only")
