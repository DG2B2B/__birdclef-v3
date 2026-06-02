"""V16 = V15 + cleaned markdown cells for better VS Code rendering."""
import json, shutil, re
from pathlib import Path

SRC = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v15.ipynb")
DST = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v16.ipynb")

shutil.copy(str(SRC), str(DST))
with open(str(DST), "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "markdown":
        continue
    src = "".join(cell["source"])

    # Clean HTML entities
    src = src.replace("&nbsp;", " ")
    src = src.replace("&lt;", "<")
    src = src.replace("&gt;", ">")
    src = src.replace("&amp;", "&")

    # Remove redundant empty table rows (lines with only | and spaces)
    lines = src.split("\n")
    cleaned = []
    prev_empty = False
    for line in lines:
        stripped = line.strip()
        # Detect empty table rows: only | and whitespace
        is_empty_row = bool(re.match(r'^\|[\s\|]*\|\s*$', stripped))
        if is_empty_row:
            if not prev_empty:
                cleaned.append("")  # single blank line
                prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False
    src = "\n".join(cleaned)

    # Collapse multiple blank lines
    src = re.sub(r'\n{3,}', '\n\n', src)

    # Ensure source is a list with one clean string
    nb["cells"][i]["source"] = [src]

print(f"Cleaned {sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')} markdown cells")

with open(str(DST), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"V16 saved: {DST.name} ({len(nb['cells'])} cells)")
print("NOT PUSHED")
