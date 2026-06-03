"""Extract code cells from .ipynb, standardize to V22-style .py with CELL markers."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def ipynb_to_standard_py(ipynb_path, py_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    lines = []
    cell_num = 0
    
    for cell in nb['cells']:
        ct = cell.get('cell_type', 'code')
        source = ''.join(cell.get('source', []))
        
        # Skip pure markdown cells
        if ct == 'markdown':
            # Keep only if it looks like a section header (important context)
            stripped = source.strip()
            if stripped.startswith('# ') or stripped.startswith('## '):
                lines.append(f"# === {stripped}\n")
            continue
        
        # Skip empty code cells
        if not source.strip():
            continue
        
        # Skip cells that are pure comments/headers without code
        if all(line.strip().startswith('#') or not line.strip() for line in source.split('\n')):
            # Keep as comment block but don't count as CELL
            lines.append(source.rstrip() + '\n')
            continue
        
        cell_num += 1
        # Clean markdown artifacts (HTML entities, encoded chars)
        source = source.replace('\u200b', '')  # zero-width space
        source = source.replace('\u00a0', ' ')  # non-breaking space
        
        lines.append(f"# === CELL {cell_num} ===\n")
        lines.append(source.rstrip() + '\n')
        lines.append('\n')
    
    result = '\n'.join(lines)
    Path(py_path).write_text(result, encoding='utf-8')
    print(f"OK: {cell_num} code cells -> {py_path}")

if __name__ == '__main__':
    ipynb_to_standard_py(
        ROOT / 'notebooks' / 'eos_submit_v11.ipynb',
        ROOT / 'notebooks' / 'eos_submit_v11_extracted.py'
    )
    ipynb_to_standard_py(
        ROOT / 'notebooks' / 'perch_ssm_submit_v7.ipynb',
        ROOT / 'notebooks' / 'perch_ssm_submit_v7_extracted.py'
    )
    print('Done.')
