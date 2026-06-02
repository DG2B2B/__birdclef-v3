"""Piste A: Create V10 = V2 + only Model_3 weight change. Zero other modifications."""
import json
from pathlib import Path

src = Path(r'c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v2.ipynb')
dst = Path(r'c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v10.ipynb')

with open(src, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    cell_src = ''.join(cell['source'])
    if "solutions = {" in cell_src and "'type_add'" in cell_src:
        cell_src = cell_src.replace("'weight':0.0", "'weight':0.10")
        cell_src = cell_src.replace("'weight':1.0", "'weight':0.90")
        nb['cells'][i]['source'] = [cell_src]
        print(f'V10: Changed solutions in cell {i}')
        print(cell_src.strip()[:300])
        break

with open(dst, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print('V10 saved')
