"""One-shot: convert V24 and V25 .py to .ipynb"""
import json, re
from pathlib import Path

def py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    cells_raw = re.split(r'^# === CELL \d+.*$\n?', content, flags=re.MULTILINE)
    if cells_raw and not cells_raw[0].strip():
        cells_raw = cells_raw[1:]
    notebook = {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12.0'}},
        'cells': []
    }
    for code in cells_raw:
        code = code.strip()
        if not code: continue
        notebook['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [code]})
    Path(ipynb_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f'OK: {len(notebook["cells"])} cells -> {ipynb_path}')

ROOT = Path(__file__).resolve().parent.parent
py_to_ipynb(str(ROOT / 'notebooks' / 'eos_submit_v24.py'), str(ROOT / '_kaggle_push' / 'eos_v8' / 'eos_submit_v24.ipynb'))
py_to_ipynb(str(ROOT / 'notebooks' / 'eos_submit_v25.py'), str(ROOT / '_kaggle_push' / 'eos_v9' / 'eos_submit_v25.ipynb'))
print('Done V24+V25')
