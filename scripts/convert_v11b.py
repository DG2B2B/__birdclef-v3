"""Convert v11b to .ipynb"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
py_path = ROOT / 'notebooks' / 'eos_submit_v11b_extracted.py'
ipynb_path = ROOT / '_kaggle_push' / 'eos_v12' / 'eos_submit_v11b.ipynb'

content = py_path.read_text(encoding='utf-8')
cells_raw = re.split(r'^# === CELL \d+.*$\n?', content, flags=re.MULTILINE)
if cells_raw and not cells_raw[0].strip():
    cells_raw = cells_raw[1:]

nb = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[]}
for code in cells_raw:
    code = code.strip()
    if not code: continue
    nb['cells'].append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[code]})

ipynb_path.parent.mkdir(parents=True,exist_ok=True)
ipynb_path.write_text(json.dumps(nb,indent=1),encoding='utf-8')
print(f'OK: {len(nb["cells"])} cells -> V11b')
